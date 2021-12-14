__author__ = 'yunbo'

import numpy as np
import pdb

def batch_mae_frame_float(gen_frames, gt_frames):
    # [batch, width, height]
    x = np.float32(gen_frames)
    y = np.float32(gt_frames)
    mae = np.sum(np.absolute(x - y), axis=(1,2), dtype=np.float32)
    return np.mean(mae)

def batch_mae_frame_float_all(gen_frames, gt_frames):
    # [batch, width, height]
    x = np.float32(gen_frames)
    y = np.float32(gt_frames)
    mae = np.sum(np.absolute(x - y), axis=(1,2), dtype=np.float32)
    return mae

def batch_mae_frame_float1(gen_frames, gt_frames):
    # [batch, width, height]
    x = np.float32(gen_frames)
    y = np.float32(gt_frames)
    mae = np.sum(np.absolute(x - y), axis=(1), dtype=np.float32)
    return np.mean(mae)


def batch_psnr(gen_frames, gt_frames):
    # [batch, width, height]
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse = np.sum((x - y)**2, axis=(1,2), dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)

def mpjpe(pred, gt):
    assert(pred.shape == gt.shape)
    n, seq_len, joint_num = gt.shape[0:3]
    result = []
    for i in range(n):
        for j in range(seq_len):
            s1_gt = gt[i,j, ...]
            s1_pred = pred[i,j, ...]
            temp = np.linalg.norm((s1_gt -s1_pred).flatten(), ord=2)/joint_num 
            result.append(temp)
    result = np.array(result).reshape([n, seq_len])
    per_frame = np.mean(result, axis=0)
    return per_frame

def np_reproduce(targ_p3d, pred_p3d):
    output_n = pred_p3d.shape[1]
    '''
    if output_n == 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    '''
    t_3d = np.zeros((1,output_n))
    #print(targ_p3d.shape)
    for k in range(output_n):
        #j = eval_frame[k]
        j = k
        t_3d[0,k] += np.mean(np.linalg.norm(
            targ_p3d[:, j, :, :].reshape((-1, 3)) - pred_p3d[:, j, :, :].reshape((-1, 3)), 2,
            1))
    return t_3d

