import numpy as np
import pdb

def recovercmu_3d(gt,pred):
	joint_to_ignore = np.array([32,33,34,35,36,37,38])-1  
	joint_equal = np.array([16,16,16,9,9,4,4])-1  
	unchange_joint=np.array([26,27,28,29,30,31])-1  # corresponding to original joints: 0,1,2,7,8,13
	tem=np.zeros([gt.shape[0],gt.shape[1],len(joint_to_ignore)+len(unchange_joint),gt.shape[-1]])
	#pdb.set_trace()
	pred_3d=np.concatenate((pred, tem), axis=2)
	pred_3d[:,:,joint_to_ignore]=pred_3d[:,:,joint_equal]
	pred_3d[:,:,unchange_joint]=gt[:,:,unchange_joint]

	return pred_3d
