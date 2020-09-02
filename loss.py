# coding: utf-8

import torch
import numpy as np

def get_mask(tscale):

    """Generate mask of BM confidence map. """

    # mask.shape: Duration * Start Time
    mask = np.zeros([tscale, tscale], np.float32)

    # The proposals whose ending boundaries exceed the range of video are left(mask = 0).
    for i in range(tscale):
        for j in range(i, tscale):
            mask[i][j] = 1
    return torch.Tensor(mask)


def bmn_loss(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, bm_mask):
    """ 
    Loss of BMN, which consists of three parts: TEM, PEM-regression and PRM-classification.
    
    Arguements:
        1. Model output:
        pred_bm([2*D*T]): (M_c): BM confidence map which consist of 'regression' and 'binary classification'.
        pred_start([T]): Temporal boundary start probability sequence.
        pred_end([T]): Temporal boundary end probability sequence.
        2. Label from 'DataLoader':
        gt_iou_map([T*T]): (G_c): iou between certain period and all GT proposals. 
        gt_start([T]): G_(S, w): score sequence which presents ioa between certain temporal moment and expanded start periods(G_S). 
        gt_end([T]): G_(E, w): score sequence which presents ioa between certain temporal moment and expanded end periods(G_E). 
        3. Mask.
        bm_mask([T*T]): BM confidence map's mask.
    """
    # pred_bm.real_shape: batch_size * 2 * D * T 
    pred_bm_reg = pred_bm[:,0].contiguous()
    pred_bm_cls = pred_bm[:,1].contiguous()

    gt_iou_map = gt_iou_map * bm_mask

    tem_loss = tem_loss_func(pred_start, pred_end, gt_start, gt_end)
    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)

    loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss

    return loss, tem_loss, pem_reg_loss, pem_cls_loss
    

def tem_loss_func(pred_start, pred_end, gt_start, gt_end):

    """
    Adopt weighted binary logistic regression loss function for predicted and GT start/end score sequence. 
    
    Arguements:
        same as 'bmn_loss'.
    """

    def weighted_binary_logistic(pred_score, gt_label, threshold=0.5):

        # Flatten to 1d-array.
        pred_score, gt_label = pred_score.view(-1), gt_label.view(-1)

        threshold_mask = (gt_label > threshold).float()
        num_entries = len(threshold_mask)
        num_positive = torch.sum(num_entries)
        ratio = num_entries / num_positive

        # For positive one(above threshold), loss = num_entries / num_positive * log(p_i)
        # For negative one(below threshold), loss = num_entries / num_negative * log(1 - p_i)
        epsilon = 1e-6
        loss_positive = 0.5 * ratio * torch.log(pred_score + epsilon) * threshold_mask
        loss_negative = 0.5 * ratio / (ratio - 1) * torch.log(1.0 - pred_score + epsilon) * (1.0 - threshold_mask)
        loss = -1.0 * torch.mean(loss_positive + loss_negative)
        return loss
    
    loss_start = weighted_binary_logistic(pred_start, gt_start)
    loss_end = weighted_binary_logistic(pred_end, gt_end)
    tem_loss = loss_start + loss_end
    return tem_loss


def pem_reg_loss_func(pred_reg_score, gt_iou_map, bm_mask, high_threshold=0.7, low_threshold=0.3):

    """
    Use MSE + L2 loss to make each proposal's regression score approxiamte to proposal's IoU between GT.

    Arguements:
        pred_reg_score([T*T]): regression part of 'BM_confidence_map'.
        gt_iou_map([T*T]): (G_c): iou between certain period and all GT proposals. 
        bm_mask([T*T]): BM confidence map's mask.
        high_threshold(float[1]): high threshold of regression score.
        low_threshold(float[1]): low threshold of regression score.
    """

    gt_iou_map = gt_iou_map * bm_mask

    high_mask = (gt_iou_map > high_threshold).float()
    medium_mask = ((gt_iou_map <= high_threshold) & (gt_iou_map > low_threshold)).float()
    low_mask = ((gt_iou_map <= low_threshold) & (gt_iou_map >= 0.)).float()
    # ???
