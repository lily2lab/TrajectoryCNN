
# TrajectoryCNN
This is a TensorFlow implementation of TrajetorycCNN as described in the following paper: 

Xiaoli Liu, Jianqin Yin, Jin Liu, Pengxiang Ding, Jun Liu, and Huaping Liu. TrajectoryCNN: a new spatio-temporal feature learning network for human motion prediction[J]. IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2021, 31(6): 2133 - 2146.

## Setup
Required python libraries: tensorflow (>=1.0) + opencv + numpy.
Tested in ubuntu +  GTX 1080Ti with cuda (>=8.0) and cudnn (>=5.0).

## Datasets
Human3.6M, CMU-Mocap, 3DPW.
the processed data file will be available at: https://pan.baidu.com/s/1iVsvRC_PUeteY3Oi50teHA （password：123a）

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
  author={Liu, Xiaoli and Yin, Jianqin and Liu, Jin and Ding, Pengxiang and Liu, Jun and Liub, Huaping},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2020},
  publisher={IEEE}
}
```

## Contact
A part of code adopts from PredCNN at https://github.com/xzr12/PredCNN.git. 

