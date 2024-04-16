import torch
from torch import nn
import torch.nn.functional as F  
import torchvision.ops as ops
import numpy as np
import time


class RegionProposalNetwork(nn.Module):
    def __init__(self, params):
        super(RegionProposalNetwork, self).__init__()

        cascae_rpn = params["cascade"]
        if cascae_rpn:
            raise ValueError("Cascade RPN is not yet supported")
        
        in_channels = params['in_channels']

        self.scales = params["scales"]
        self.bs = params["batch_size"]
        self.aspect_ratios = params["aspect_ratios"]
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)

        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)

        self.cls_head = nn.Conv2d(512, self.num_anchors, kernel_size=1)
        self.bbox_head = nn.Conv2d(512, self.num_anchors * 4, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        objectness_logits = self.cls_head(x)
        bbox_reg = self.bbox_head(x)
        self.device = x.device

        self.anchors = self.generate_anchors(x.shape[2:], batch_size=self.bs)
        proposals = self.generate_proposals(objectness_logits, bbox_reg, self.anchors)

        return objectness_logits, bbox_reg, proposals, self.anchors.to(self.device)
    
    def generate_anchors(self, feature_map_shape, feature_stride=16, batch_size=1):
        height, width = feature_map_shape
        num_anchors = len(self.scales) * len(self.aspect_ratios)

        base_anchors = np.zeros((num_anchors, 4), dtype=np.float32)

        index = 0
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                area = scale * scale
                w = np.round(np.sqrt(area / ratio))
                h = np.round(w * ratio)
                x_center = (w - 1) / 2
                y_center = (h - 1) / 2
                    
                base_anchors[index, :] = [-x_center, -y_center, x_center, y_center]
                index += 1
            
        # Generate grid centers
        shift_x = np.arange(0, width) * feature_stride + feature_stride / 2  # Adjusted shift calculation
        shift_y = np.arange(0, height) * feature_stride + feature_stride / 2  # Adjusted shift calculation
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

        A = base_anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))

        # Repeat the anchors for each sample in the batch
        all_anchors_batch = np.tile(all_anchors[None, ...], (batch_size, 1, 1))

        return torch.from_numpy(all_anchors_batch).float()

    
    def generate_proposals(self, objectness_logits, bbox_preds, anchors):
        anchors = anchors.to("cpu")
        objectness_logits = objectness_logits.to("cpu")
        bbox_preds = bbox_preds.to("cpu")

        objectness_prob = torch.sigmoid(objectness_logits)
        objectness_prob = objectness_prob.view(self.bs, -1)
        bbox_preds = bbox_preds.permute(0, 2, 3, 1).contiguous().view(self.bs, -1, 4)
        proposals_cxcywh = self._apply_deltas_to_anchors(anchors, bbox_preds)
        proposals = ops.box_convert(proposals_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")


        high_score_idxs = objectness_prob > 0.5
        final_proposals = []
        for i in range(self.bs):
            batch_high_score_idxs = high_score_idxs[i]
            batch_high_score_probs = objectness_prob[i][batch_high_score_idxs]
            batch_proposals = proposals[i][batch_high_score_idxs]
            keep_idxs = ops.nms(batch_proposals, batch_high_score_probs, iou_threshold=0.7)
            final_proposals.append(batch_proposals[keep_idxs].to(self.device))
        return final_proposals
    
    def _apply_deltas_to_anchors(self, anchors, deltas):
        widths = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x = anchors[:, :, 0] + 0.5 * widths
        ctr_y = anchors[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0]
        dy = deltas[:, :, 1]
        dw = deltas[:, :, 2]
        dh = deltas[:, :, 3]

        pred_ctr_x = dx * widths + ctr_x  # (batch_size, N, 1) 
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.zeros_like(deltas)  # Initialize with the same shape as deltas
        pred_boxes[:, :, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, :, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, :, 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, :, 3] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes