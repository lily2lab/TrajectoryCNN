#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ../..
savepath='results/cmu/traj_cmu_long_term/v3'
modelpath='checkpoints/cmu/traj_cmu_long_term/v3'
#pretrain_modelpath='checkpoints/cmu/traj_cmu_long_term/v1/model.ckpt-6720'
logname='logs/cmu/traj_cmu_long_term.log'
nohup python -u train_TrajectoryCNN_cmu.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths data/cmu_ske/train_cmu_35.npy \
    --valid_data_paths data/cmu_ske/train_cmu_35.npy \
    --test_data_paths  data/cmu_ske/    \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --input_length 10 \
    --seq_length 35 \
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



