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
			# normalized joint coordinate
			j1=frm_joint[:,0]
			j2=frm_joint[:,1]
			j3=frm_joint[:,2]
			minvalue1=np.min(j1)
			maxvalue1=np.max(j1)
			j1=(j1-minvalue1)*1.0/(maxvalue1-minvalue1)
			minvalue2=np.min(j2)
			maxvalue2=np.max(j2)
			j2=(j2-minvalue2)*1.0/(maxvalue2-minvalue2)
			minvalue3=np.min(j3)
			maxvalue3=np.max(j3)
			j3=(j3-minvalue3)*1.0/(maxvalue3-minvalue3)
			frm_joint[:,0]=j1
			frm_joint[:,1]=j2
			frm_joint[:,2]=j3
			
			joints[i,j] = frm_joint

	return joints

def normjoints(imbat):
	batch_size = imbat.shape[0]
	joints = np.zeros([batch_size,18,3])
	for i in range(batch_size):
		frm_joint=imbat[i]
		j1=frm_joint[:,0]
		j2=frm_joint[:,1]
		j3=frm_joint[:,2]
		minvalue1=np.min(j1)
		maxvalue1=np.max(j1)
		j1=(j1-minvalue1)*1.0/(maxvalue1-minvalue1)
		minvalue2=np.min(j2)
		maxvalue2=np.max(j2)
		j2=(j2-minvalue2)*1.0/(maxvalue2-minvalue2)
		minvalue3=np.min(j3)
		maxvalue3=np.max(j3)
		j3=(j3-minvalue3)*1.0/(maxvalue3-minvalue3)
		frm_joint[:,0]=j1
		frm_joint[:,1]=j2
		frm_joint[:,2]=j3
		joints[i] = frm_joint

	return joints

