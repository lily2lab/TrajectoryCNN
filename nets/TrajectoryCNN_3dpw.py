import tensorflow as tf
from layers.TrajBlock import TrajBlock as TB 
import pdb
import numpy as np

def TrajectoryCNN(images,keep_prob, seq_length, input_length, stacklength, num_hidden,filter_size):
	with tf.variable_scope('TrajectoryCNN', reuse=False):
		print 'TrajectoryCNN'
		#print 'is_training', is_training
		h = images[:,0:seq_length,:,:]
		gt_images=images[:,seq_length:]
		dims=gt_images.shape[-1]
		inputs = h
		inputs = tf.transpose(h, [0,2,3,1])

		out=[]
		loss = 0
		inputs = tf.layers.conv2d(inputs,num_hidden[0],1, padding='same',activation=tf.nn.leaky_relu,  
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='h0')
		for i in range(stacklength):
			inputs=TB('TrajBlock'+str(i),filter_size,num_hidden,keep_prob)(inputs)
		
		out = tf.layers.conv2d(inputs, seq_length-input_length, filter_size, padding='same',activation=tf.nn.leaky_relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='Decoder_conv1')
		out = tf.layers.conv2d(out, seq_length-input_length, 1, padding='same',activation=None,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='Decoder_conv2')
		#pdb.set_tracie()
		out = tf.transpose(out, [0,3,1,2])
		# loss
		gen_images=out
		loss += 1000 * tf.reduce_mean(tf.norm(gen_images-gt_images, axis=3, keep_dims=True, name='normal')) # mpjpe loss
		#loss=tf.norm((gen_images-gt_images),ord=1) # 	L1 loss	
		return [gen_images, loss]
