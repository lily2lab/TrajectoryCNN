
# TrajectoryCNN
This is a TensorFlow implementation of TrajetorycCNN as described in the following paper: 

Xiaoli Liu, Jianqin Yin, Jin Liu, Pengxiang Ding, Jun Liu, and Huaping Liu. TrajectoryCNN: a new spatio-temporal feature learning network for human motion prediction[J]. IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2021, 31(6): 2133 - 2146.

## Setup
Required python libraries: tensorflow (>=1.0) + opencv + numpy.
Tested in ubuntu +  GTX 1080Ti with cuda (>=8.0) and cudnn (>=5.0).

## Datasets
Human3.6M, CMU-Mocap, 3DPW.
the processed data file will be available at: https://pan.baidu.com/s/1iVsvRC_PUeteY3Oi50teHA （password：123a）

# orders of joints in this paper
```shell
Human3.6M dataset: 30,29,27,26,25,17,18,19,21,22,15,14,13,12,7,8,9,10,2,3,4,5,  16,20,23,24,28,31,   0,1,6,11
CMU-Mocap dataset: 38,36,35,33,32,31,22,23,24,26,27,29,    20,19,18,16,15,   10,11,12,13, 4,5,6,7,    1,2,3,8,9,14,  17,21,30,25,28,34,37
3DPW dataset: 23,21,19,17,14,13,16,18,20,22,   15,12,9,6,3,  2,5,8,11, 1,4,7,10,  0
```

## Training/Testing
Use the `scripts/h36m/TrajectoryCNN_short_term_train.sh` or `scripts/h36m/TrajectoryCNN_long_term_train.sh` script to train/test the model on Human3.6M dataset for short-term or long-term predictions by the following commands:
```shell
cd scripts/h36m
sh TrajectoryCNN_short_term_train.sh  # for short-term prediction on Human3.6M
sh TrajectoryCNN_long_term_train # for long-term predictions on Human3.6M
```
You might want to change folders in `scripts` to train on CMU-Mocap or 3DPW datasets.


## Citation
If you use this code for your research, please consider citing:
```latex
@article{liu2020trajectorycnn,
  title={TrajectoryCNN: a new spatio-temporal feature learning network for human motion prediction},
  author={Liu, Xiaoli and Yin, Jianqin and Liu, Jin and Ding, Pengxiang and Liu, Jun and Liu, Huaping},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2020},
  publisher={IEEE}
}
```

## Contact
A part of code adopts from PredCNN at https://github.com/xzr12/PredCNN.git. 

