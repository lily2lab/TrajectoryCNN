import numpy as np
import pdb

def recoverh36m_3d(gt,pred):
	joint_to_ignore = np.array([22, 23, 24, 25, 26, 27])
	joint_equal = np.array([12, 7, 9, 12, 2, 0])
	unchange_joint=np.array([28,29,30,31])  # corresponding to original joints: 0,1,6,11
	tem=np.zeros([gt.shape[0],gt.shape[1],len(joint_to_ignore)+len(unchange_joint),gt.shape[-1]])
	#pdb.set_trace()
	pred_3d=np.concatenate((pred, tem), axis=2)
	pred_3d[:,:,joint_to_ignore]=pred_3d[:,:,joint_equal]
	pred_3d[:,:,unchange_joint]=gt[:,:,unchange_joint]

	return pred_3d
