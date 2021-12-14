import numpy as np
import recoverjoints 
import pdb

rangx=np.array([4,7,7,7,7,7,7,7,7,7,10,13,13,13,16,16,19,19]) # representation 1
rangy=np.array([16,4,7,10,13,16,19,22,25,28,16,13,16,19,13,19,13,19])


pdb.set_trace()

testfile = 'Bowling_136.npy'
data=np.load(testfile)
tem=[]
tem.append(data[1:20])
tem.append(data[21:40])
tem.append(data[41:60])
tem=np.array(tem)

print tem.shape
joints=recoverjoints.recoverjoints(tem,rangx,rangy)

