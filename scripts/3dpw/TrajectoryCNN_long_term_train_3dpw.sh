#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ../..
savepath='results/3dpw/traj_3dpw_long/v2'
modelpath='checkpoints/3dpw/traj_3dpw_long/v2'
#pretrain_modelpath='checkpoints/3dpw/traj_3dpw_long/v1/model.ckpt-320'
logname='logs/3dpw/traj_3dpw_long.log'
nohup python -u train_TrajectoryCNN_3dpw.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths data/3dpw_ske/train_3dpw0_40.npy \
    --valid_data_paths data/3dpw_ske/test_3dpw0_40.npy \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --input_length 10 \
    --seq_length 40 \
    --stacklength 4 \
    --filter_size 3 \
    --lr 0.0001 \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 3000000 \
    --display_interval 10 \
    --test_interval 500 \
    --n_gpu 4 \
    --snapshot_interval 500  >>${logname}  2>&1 &

tail -f ${logname}

# --pretrained_model ${pretrain_modelpath}  \



