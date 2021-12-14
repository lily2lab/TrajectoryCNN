#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ../..
savepath='results/cmu/traj_cmu_short_term/v3'
modelpath='checkpoints/cmu/traj_cmu_short_term/v3'
#pretrain_modelpath='checkpoints/cmu/traj_cmu_short_term/v2/model.ckpt-24100'
logname='logs/cmu/traj_cmu_short_term.log'
nohup python -u train_TrajectoryCNN_cmu.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths data/cmu_ske/train_cmu_20.npy \
    --valid_data_paths data/cmu_ske/train_cmu_20.npy \
    --test_data_paths  data/testset/    \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --input_length 10 \
    --seq_length 20 \
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

#  --pretrained_model checkpoints/ske_predcnn/model.ckpt-1000 \
# --pretrained_model ${pretrain_modelpath}  \



