#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ../..
savepath='results/h36m/v0'
modelpath='checkpoints/h36m/v0'
#pretrain_modelpath='/home/data2/lxldata/Trajectorylet_exp/h36m/models/traj_finetune_mpjpe_pretrain_stack4_droupout_call64_final_20_2/v5/model.ckpt-532000'
#sleep 4h
logname='logs/h36m/train_h36m.log'
nohup python -u train_TrajectoryCNN_h36m.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths data/h36m20/h36m20_train_3d.npy \
    --valid_data_paths data/h36m20/h36m20_val_3d.npy \
    --test_data_paths data/h36m20/testset \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --bak_dir ${bak_path}   \
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



