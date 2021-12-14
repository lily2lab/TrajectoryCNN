import numpy as np

def recoverjoints(imbat,x,y):
	batch_size = imbat.shape[0]
	seq_length = imbat.shape[1]
	joints = np.zeros([batch_size,seq_length,18,3])
	#print 'imbat size',imbat.shape
	for i in range(batch_size):
		seq = imbat[i]
		for j in range(seq_length):
			im = seq[j]
			frm_joint = np.zeros([18,3])
			for k in range(18):
				frm_joint[k,0] = im[x[k],y[k],0]
				frm_joint[k,1] = im[x[k],y[k],1]
				frm_joint[k,2] = im[x[k],y[k],2]
				#joints[i,j,k,1] = im[x[k],y[k],1]
				#joints[i,j,k,2] = im[x[k],y[k],2]
			#print frm_joint
			
			joints[i,j] = frm_joint

	return joints


