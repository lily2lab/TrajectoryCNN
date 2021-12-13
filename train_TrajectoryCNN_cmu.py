import tensorflow as tf
import os.path
import numpy as np
from nets import TrajectoryCNN as  TrajectoryCNN
from data_provider import datasets_factory_joints_cmu as datasets_factory
from utils import metrics
from utils import recovercmu_3d as recovercmu_3d
from utils import optimizer
import time
import scipy.io as io
import os,shutil
import pdb

FLAGS = tf.app.flags.FLAGS
# data path
tf.app.flags.DEFINE_string('dataset_name', 'skeleton',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           'data/moving-mnist-example/moving-mnist-train.npz',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           'data/moving-mnist-example/moving-mnist-valid.npz',
                           'validation data paths.')
tf.app.flags.DEFINE_string('test_data_paths',
                           'data/moving-mnist-example/moving-mnist-valid.npz',
                           'test data paths.')
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/mnist_predcnn',
                            'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_dir', 'results/mnist_predcnn',
                           'path to save generate results')
tf.app.flags.DEFINE_string('bak_dir', '',
                            'dir to backup result.')
# model parameter
tf.app.flags.DEFINE_string('pretrained_model','',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 10,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 35,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('joints_number', 25,
                            'the number of joints of a pose')
tf.app.flags.DEFINE_integer('joint_dims', 3,
                            'one joints dims.')

tf.app.flags.DEFINE_integer('stacklength', 8,
                            'stack trajblock number.')
#tf.app.flags.DEFINE_integer('numhidden', '100,100,100,100,100',
#                            'trajblock filter number.')
tf.app.flags.DEFINE_integer('filter_size', 3,
                            'filter size.')

# opt
tf.app.flags.DEFINE_float('lr', 0.0001,
                          'base learning rate.')
tf.app.flags.DEFINE_integer('batch_size', 8,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 100000,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 20,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 10000,
                            'number of iters saving models.')
tf.app.flags.DEFINE_integer('num_save_samples', 100000,
                            'number of sequences to be saved.')
tf.app.flags.DEFINE_integer('n_gpu', 4,
                            'how many GPUs to distribute the training across.')
num_hidden=[64,64,64,64,64]
print'!!! TrajectoryCNN:', num_hidden
class Model(object):
	def __init__(self):
		# inputs
		self.x = [tf.placeholder(tf.float32,[FLAGS.batch_size,
											FLAGS.seq_length+FLAGS.seq_length-FLAGS.input_length,
											FLAGS.joints_number,
											FLAGS.joint_dims])
				for i in range(FLAGS.n_gpu)]
		grads = []
		loss_train = []
		self.pred_seq = []
		self.tf_lr = tf.placeholder(tf.float32, shape=[])
		self.keep_prob = tf.placeholder(tf.float32)
		self.params = dict()
        

		for i in range(FLAGS.n_gpu):
			with tf.device('/gpu:%d' % i):
				with tf.variable_scope(tf.get_variable_scope(),
						reuse=True if i > 0 else None):
					# define a model
					output_list = TrajectoryCNN.TrajectoryCNN(
							self.x[i],
							self.keep_prob,
							FLAGS.seq_length,
							FLAGS.input_length,
							FLAGS.stacklength,
							num_hidden,
							FLAGS.filter_size)

					gen_ims = output_list[0]
					loss = output_list[1]
					pred_ims = gen_ims[:, FLAGS.input_length - FLAGS.seq_length:]
					loss_train.append(loss)   ###
					# gradients
					all_params = tf.trainable_variables()
					grads.append(tf.gradients(loss, all_params))
					self.pred_seq.append(pred_ims)

		if FLAGS.n_gpu == 1:
			self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
		else:
			# add losses and gradients together and get training updates
			with tf.device('/gpu:0'):
				for i in range(1, FLAGS.n_gpu):
					loss_train[0] += loss_train[i]
					for j in range(len(grads[0])):
						grads[0][j] += grads[i][j]
			# keep track of moving average
			ema = tf.train.ExponentialMovingAverage(decay=0.9995)
			maintain_averages_op = tf.group(ema.apply(all_params))
			self.train_op = tf.group(optimizer.adam_updates(
				all_params, grads[0], lr=self.tf_lr, mom1=0.95, mom2=0.9995),
				maintain_averages_op)

		self.loss_train = loss_train[0] / FLAGS.n_gpu

		# session
		variables = tf.global_variables()
		self.saver = tf.train.Saver(variables)
		init = tf.global_variables_initializer()
		configProt = tf.ConfigProto()
		configProt.gpu_options.allow_growth = True
		configProt.allow_soft_placement = True
		self.sess = tf.Session(config = configProt)
		self.sess.run(init)
		if FLAGS.pretrained_model:
			print 'pretrain model: ',FLAGS.pretrained_model
			self.saver.restore(self.sess, FLAGS.pretrained_model)

	def train(self, inputs, lr, keep_prob):
		feed_dict = {self.x[i]: inputs[i] for i in range(FLAGS.n_gpu)}
		feed_dict.update({self.tf_lr: lr})
		feed_dict.update({self.keep_prob: keep_prob})
		loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
		return loss

	def test(self, inputs, keep_prob):
		feed_dict = {self.x[i]: inputs[i] for i in range(FLAGS.n_gpu)}
		feed_dict.update({self.keep_prob: keep_prob})
		gen_ims = self.sess.run(self.pred_seq, feed_dict)
		return gen_ims

	def save(self, itr):
		checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
		self.saver.save(self.sess, checkpoint_path, global_step=itr)
		print('saved to ' + FLAGS.save_dir)


def main(argv=None):
	if not tf.gfile.Exists(FLAGS.save_dir):
		tf.gfile.MakeDirs(FLAGS.save_dir)
	if not tf.gfile.Exists(FLAGS.gen_dir):
		tf.gfile.MakeDirs(FLAGS.gen_dir)

	print 'start training !',time.strftime('%Y-%m-%d %H:%M:%S\n\n\n',time.localtime(time.time()))
	# load data
	train_input_handle, test_input_handle = datasets_factory.data_provider(
		FLAGS.dataset_name,FLAGS.train_data_paths, FLAGS.valid_data_paths,
		FLAGS.batch_size * FLAGS.n_gpu, FLAGS.joints_number,FLAGS.input_length,FLAGS.seq_length,is_training=True)

	print('Initializing models')
	model = Model()
	lr = FLAGS.lr
	train_time=0
	test_time_all=0
	folder=1
	path_bak=FLAGS.bak_dir
	min_err=10000.0
	errlist=[]

	Keep_prob = 0.75
	for itr in range(1, FLAGS.max_iterations + 1):
		if train_input_handle.no_batch_left():
			train_input_handle.begin(do_shuffle=True)
		'''
		if itr % 20000 == 0:
			lr = lr* 0.95
		'''
		
		start_time = time.time()
		ims = train_input_handle.get_batch()
		ims = ims[:,:,0:FLAGS.joints_number,:]
		pretrain_iter=0
		if itr<pretrain_iter:
			inputs1=ims
		else:
			inputs1=ims[:,0:FLAGS.input_length,:,:]
			tem=ims[:,FLAGS.input_length-1]
			tem=np.expand_dims(tem,axis=1)
			tem=np.repeat(tem,FLAGS.seq_length - FLAGS.input_length,axis=1)
			inputs1=np.concatenate((inputs1,tem),axis=1)
		#pdb.set_trace()
		inputs2=ims[:,FLAGS.input_length:]
		inputs=np.concatenate((inputs1,inputs2),axis=1)
		ims_list = np.split(inputs, FLAGS.n_gpu)
		cost = model.train(ims_list, lr, Keep_prob)
		# inverse the input sequence
		imv1=ims[:, ::-1]
		if itr>=pretrain_iter:
			imv_rev1=imv1[:,0:FLAGS.input_length,:,:]
			#pdb.set_trace()
			tem=imv1[:,FLAGS.input_length-1]
			tem=np.expand_dims(tem,axis=1)
			tem=np.repeat(tem,FLAGS.seq_length - FLAGS.input_length,axis=1)
			#pdb.set_trace()
			imv_rev1=np.concatenate((imv_rev1,tem),axis=1)
		else:
			imv_rev1 = imv1
		imv_rev2=imv1[:,FLAGS.input_length:]
		ims_rev1=np.concatenate((imv_rev1,imv_rev2),axis=1)
		ims_rev1 = np.split(ims_rev1, FLAGS.n_gpu)
		cost += model.train(ims_rev1, lr, Keep_prob)
		cost = cost/2
		
		end_time = time.time()
		t = end_time-start_time
		train_time += t

		if itr % FLAGS.display_interval == 0:
			print('itr: ' + str(itr)+' lr: '+str(lr)+' training loss: ' + str(cost))

		if itr % FLAGS.test_interval == 0:
			print('train time:'+ str(train_time))
			print('test...')
			str1 = 'basketball','basketball_signal','directing_traffic','jumping','running','soccer','walking','washwindow'
			res_path = os.path.join(FLAGS.gen_dir, str(itr))
			if  not tf.gfile.Exists(res_path):
				os.mkdir(res_path)
			avg_mse = 0
			batch_id = 0
			test_time=0
			joint_mse = np.zeros((25,38))
			joint_mae = np.zeros((25,38))
			mpjpe = np.zeros([1,FLAGS.seq_length - FLAGS.input_length])
			mpjpe_l = np.zeros([1,FLAGS.seq_length - FLAGS.input_length])
			img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []
			for i in range(FLAGS.seq_length - FLAGS.input_length):
				img_mse.append(0)
				fmae.append(0)
			f = 0
			for s in str1:
				start_time1 = time.time()
				batch_id = batch_id + 1
				mpjpe1=np.zeros([1,FLAGS.seq_length - FLAGS.input_length])
				tem = np.load(FLAGS.test_data_paths+'/test_cmu_'+str(FLAGS.seq_length)+'_'+s+'.npy')
				tem=np.repeat(tem,(FLAGS.batch_size*FLAGS.n_gpu)/8,axis=0)
				test_ims =tem[:,0:FLAGS.seq_length,:,:]
				test_ims1= test_ims
				test_ims=test_ims[:,:,0:FLAGS.joints_number,:]
				
				test_dat=test_ims[:,0:FLAGS.input_length,:,:]
				tem=test_dat[:,FLAGS.input_length-1]
				tem=np.expand_dims(tem,axis=1)
				tem=np.repeat(tem,FLAGS.seq_length - FLAGS.input_length,axis=1)
				test_dat1=np.concatenate((test_dat,tem),axis=1)
				test_dat2=test_ims[:,FLAGS.input_length:]
				test_dat=np.concatenate((test_dat1,test_dat2),axis=1)
				test_dat = np.split(test_dat, FLAGS.n_gpu)
				img_gen = model.test(test_dat,1)
				end_time1 = time.time()
				t1=end_time1-start_time1
				test_time += t1
				# concat outputs of different gpus along batch
				img_gen = np.concatenate(img_gen)
				gt_frm = test_ims1[:,FLAGS.input_length:]
				img_gen = recovercmu_3d.recovercmu_3d(gt_frm,img_gen)

				# MSE per frame
				for i in range(FLAGS.seq_length - FLAGS.input_length):
					x = gt_frm[:, i , :, ]
					gx = img_gen[:, i, :, ]
					fmae[i] += metrics.batch_mae_frame_float(gx, x)
					mse = np.square(x - gx).sum()
					for j in range(FLAGS.batch_size * FLAGS.n_gpu):
						tem1=0
						for k in range(gt_frm.shape[2]):
							tem1 += np.sqrt(np.square(x[j,k] - gx[j,k]).sum())
						mpjpe1[0,i] += tem1/(gt_frm.shape[2])
		
					img_mse[i] += mse
					avg_mse += mse 
					real_frm = x
					pred_frm = gx
					for j in range(gt_frm.shape[2]):
						xi = x[:,j]
						gxi = gx[:,j]
						joint_mse[i,j] += np.square(xi - gxi).sum()
						joint_mae[i,j] += metrics.batch_mae_frame_float1(gxi, xi)
				# save prediction examples
				path = os.path.join(res_path, s)
				if  not tf.gfile.Exists(path):
					os.mkdir(path)
				for ik in range(8):
					spath = os.path.join(path, str(ik))
					if  not tf.gfile.Exists(spath):
						os.mkdir(spath)
					for i in range(FLAGS.seq_length):
						name = 'gt' + str(i+1) + '.mat'
						file_name = os.path.join(spath, name)
						img_gt = test_ims1[ik*8, i, :, :]
						io.savemat(file_name, {'joint': img_gt})
					for i in range(FLAGS.seq_length-FLAGS.input_length):
						name = 'pd' + str(i+1+FLAGS.input_length) + '.mat'
						file_name = os.path.join(spath, name)
						img_pd = img_gen[ik*8, i, :, :]
						io.savemat(file_name, {'joint': img_pd})
				mpjpe1 = mpjpe1/(FLAGS.batch_size * FLAGS.n_gpu)

				print 'current action mpjpe: ',s
				for i in mpjpe1[0]:
					print i
				mpjpe +=mpjpe1
				if f<=3:
					print 'four actions',s
					mpjpe_l += mpjpe1
				f=f+1
			test_time_all += test_time
			joint_mae = np.asarray(joint_mae, dtype=np.float32) / batch_id
			joint_mse = np.asarray(joint_mse, dtype=np.float32)/(batch_id * FLAGS.batch_size * FLAGS.n_gpu)
			avg_mse = avg_mse / (batch_id * FLAGS.batch_size * FLAGS.n_gpu)
			print('mse per seq: ' + str(avg_mse))
			#for i in range(FLAGS.seq_length - FLAGS.input_length):
			#	print(img_mse[i] / (batch_id * FLAGS.batch_size * FLAGS.n_gpu))
			mpjpe=mpjpe/(batch_id)
			errlist.append(np.mean(mpjpe))
			print( 'mean per joints position error: '+str(np.mean(mpjpe)))
			for i in range(FLAGS.seq_length - FLAGS.input_length):
				print(mpjpe[0,i])
			mpjpe_l=mpjpe_l/4
			print('mean mpjpe for four actions: '+str(np.mean(mpjpe_l)))
			for i in range(FLAGS.seq_length - FLAGS.input_length):
				print(mpjpe_l[0,i])
			fmae = np.asarray(fmae, dtype=np.float32) / batch_id
			print('fmae per frame: ' + str(np.mean(fmae)))
			#for i in range(FLAGS.seq_length - FLAGS.input_length):
			#	print(fmae[i])
			print 'current test time:'+str(test_time)
			print 'all test time: '+str(test_time_all)
			filename = os.path.join(res_path, 'test_result')
			io.savemat(filename, {'joint_mse': joint_mse,'joint_mae': joint_mae,'mpjpe':mpjpe})

		if itr % FLAGS.snapshot_interval == 0 and min(errlist) < min_err:
			model.save(itr)
			min_err = min(errlist) 
			print 'model saving done! ', time.strftime('%Y-%m-%d %H:%M:%S\n\n\n',time.localtime(time.time()))
		if itr % FLAGS.snapshot_interval == 0:
			print 'current minimize error is: ', min_err

		train_input_handle.next()

if __name__ == '__main__':
	tf.app.run()










