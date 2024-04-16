from torch import nn
import torch
import numpy as np
import torchvision.ops as ops
import torch.nn.functional as F
        

def calculate_iou(boxes1, boxes2):
    """
    Compute intersection over union (IoU) between two sets of boxes.
    
    Args:
    - boxes1 (Tensor): First set of boxes.
    - boxes2 (Tensor): Second set of boxes.
    
    Returns:
    - iou (Tensor): IoU between the two sets of boxes.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    iou = inter / (area1[:, None] + area2 - inter)
    
    return iou

def calculate_bbox_deltas(proposals, matched_gt_boxes):
    proposals_cxcywh = ops.box_convert(proposals, in_fmt="xyxy", out_fmt="cxcywh")
    gt_boxes_cxcywh = ops.box_convert(matched_gt_boxes, in_fmt="xyxy", out_fmt="cxcywh")
    
    dx = (gt_boxes_cxcywh[:, 0] - proposals_cxcywh[:, 0]) / proposals_cxcywh[:, 2]
    dy = (gt_boxes_cxcywh[:, 1] - proposals_cxcywh[:, 1]) / proposals_cxcywh[:, 3]

    dw = torch.log(gt_boxes_cxcywh[:, 2] / proposals_cxcywh[:, 2])
    dh = torch.log(gt_boxes_cxcywh[:, 3] / proposals_cxcywh[:, 3])

    deltas = torch.stack([dx, dy, dw, dh], dim=1)
    return deltas

def assign_ground_truth_to_proposals(proposals, gt_boxes, gt_labels, bg_label=0, iou_threshold=0.5):
    bs = len(proposals)

    proposal_labels = []
    bbox_deltas = []
    
    for i in range(bs):
        batch_proposals = proposals[i]
        if batch_proposals.shape[0] == 0 or gt_boxes[i].shape[0] == 0:
            continue  # Skip if there are no proposals or ground truths for this image

        iou = ops.box_iou(batch_proposals, gt_boxes[i])
        max_iou, max_idx = iou.max(dim=1)

        pos_mask = max_iou >= iou_threshold
        if pos_mask.any():
            labels = torch.full((batch_proposals.shape[0],), bg_label, dtype=torch.long)
            matched_gt_labels = gt_labels[i][max_idx[pos_mask]]
            matched_gt_boxes = gt_boxes[i][max_idx[pos_mask]]
            labels[pos_mask] = matched_gt_labels
            proposal_labels.append(labels)
            bbox_deltas.append(calculate_bbox_deltas(batch_proposals[pos_mask], matched_gt_boxes))
        else:
            proposal_labels.append(torch.full((batch_proposals.shape[0],), bg_label, dtype=torch.long))
            bbox_deltas.append(torch.zeros((0, 4)))
    return proposal_labels, bbox_deltas


def assign_ground_truth_to_proposals_(proposals, gt_boxes, gt_labels, bg_label=0, iou_threshold=0.5):
    batch_size = proposals.size(0)
    for i in range(batch_size):
        iou = ops.box_iou(proposals[i], gt_boxes[i])
        max_iou, max_idx = iou.max(dim=1)

        proposal_labels = torch.full((proposals.size(1),), bg_label, dtype=torch.long, device=proposals.device)
        matched_gt_boxes = torch.zeros_like(proposals)
        bbox_deltas = torch.zeros((proposals.size(1), 4), device=proposals.device)
        
        # Assign ground truth labels to proposals with IoU above threshold
        pos_mask = max_iou >= iou_threshold
        if pos_mask.any():
            proposal_labels[pos_mask] = gt_labels[max_idx[pos_mask]]
            matched_gt_boxes[pos_mask] = gt_boxes[max_idx[pos_mask]]
            bbox_deltas[pos_mask] = calculate_bbox_deltas(proposals[pos_mask], matched_gt_boxes[pos_mask])
    

    return proposal_labels, bbox_deltas

def assign_ground_truth_to_anchors(anchors, gt_boxes, positive_threshold=0.7, negative_threshold=0.3):
    batch_size = anchors.size(0)
    num_anchors = anchors.size(1)

    anchor_labels = torch.zeros((batch_size, num_anchors), dtype=torch.long)
    anchor_gt_boxes = torch.zeros((batch_size, num_anchors, 4), dtype=gt_boxes[0].dtype)

    for i in range(batch_size):
        iou = ops.box_iou(anchors[i], gt_boxes[i])  # Shape: [num_anchors, num_gt_boxes]
        best_gt_iou, best_gt_idx = iou.max(dim=1)
        
        anchor_labels[i, best_gt_iou < negative_threshold] = 0
        anchor_labels[i, best_gt_iou > positive_threshold] = 1
        batch_gt_boxes = gt_boxes[i]
        anchor_gt_boxes[i] = batch_gt_boxes[best_gt_idx]
        anchor_gt_boxes[i, anchor_labels[i] == 0] = 0

    return anchor_labels, anchor_gt_boxes
    