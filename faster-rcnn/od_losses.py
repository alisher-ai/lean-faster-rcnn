from torch import nn
import torch.nn.functional as F
import torch


class RPNLoss(nn.Module):
    def __init__(self, cls_loss_weight, bbox_loss_weight, batch_size) -> None:
        super(RPNLoss, self).__init__()
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.bs = batch_size

    def forward(self, rpn_cls_logits, rpn_bbox_reg, anchor_labels, anchor_gt_boxes):
            """
            Calculates the RPN loss.

            Args:
                rpn_cls_logits (Tensor): Raw classification scores from the RPN.
                rpn_bbox_reg (Tensor): Bounding box regression deltas from the RPN.
                anchor_labels (Tensor): Labels for anchors (0 for background, 1 for object).
                anchor_gt_boxes (Tensor): Ground truth boxes associated with each anchor.

            Returns:
                Tensor: The weighted sum of classification and bounding box regression losses.
            """
            rpn_cls_loss = F.binary_cross_entropy_with_logits(rpn_cls_logits.view(self.bs, -1), anchor_labels.float().to(rpn_cls_logits.device))

            rpn_bbox_reg_flat = rpn_bbox_reg.permute(0, 2, 3, 1).contiguous().view(self.bs, -1, 4)
            pos_indices = (anchor_labels == 1).nonzero(as_tuple=False).squeeze(1)

            if pos_indices.numel() == 0:
                rpn_bbox_loss = torch.tensor(0.0, dtype=torch.float32, device=rpn_cls_logits.device) 

            else:
                pos_indices = (anchor_labels == 1).nonzero(as_tuple=False)
                batch_indices = pos_indices[:, 0]
                anchor_indices = pos_indices[:, 1]

                rpn_bbox_reg_pos = rpn_bbox_reg_flat[batch_indices, anchor_indices]
                anchor_gt_boxes_pos = anchor_gt_boxes[batch_indices, anchor_indices]

                rpn_bbox_loss = F.smooth_l1_loss(rpn_bbox_reg_pos, anchor_gt_boxes_pos, reduction='sum') / pos_indices.numel()

            rpn_loss = self.cls_loss_weight * rpn_cls_loss + self.bbox_loss_weight * rpn_bbox_loss
            return rpn_loss


class ROILoss(nn.Module):
    def __init__(self, cls_loss_weight, bbox_loss_weight, batch_size) -> None:
        super(ROILoss, self).__init__()
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.bs = batch_size
    
    def forward(self, cls_logits, bbox_preds, proposal_labels, bbox_deltas):
        roi_loss_list = []
        for i in range(self.bs):
            batch_cls_logits = cls_logits[i]
            batch_proposal_labels = proposal_labels[i]
            batch_bbox_preds = bbox_preds[i]
            fg_mask = batch_proposal_labels > 0
            fg_bbox_preds = batch_bbox_preds[fg_mask]

            fg_proposal_labels = batch_proposal_labels[fg_mask]
            fg_bbox_targets = torch.zeros_like(fg_bbox_preds)
            if fg_bbox_preds.shape[0] == 0:
                roi_loss = torch.tensor(0.0, dtype=torch.float32)
                roi_loss_list.append(roi_loss)
                continue
            
            for j, label in enumerate(fg_proposal_labels):
                start_idx = label * 4
                end_idx = start_idx + 4
                fg_bbox_targets[j, start_idx:end_idx] = bbox_deltas[i][j]

            roi_bbox_loss = F.smooth_l1_loss(fg_bbox_preds, fg_bbox_targets)
            roi_cls_loss = F.cross_entropy(batch_cls_logits, batch_proposal_labels)

            roi_loss = self.cls_loss_weight * roi_cls_loss + self.bbox_loss_weight * roi_bbox_loss
            roi_loss_list.append(roi_loss)
        batch_roi_loss = torch.stack(roi_loss_list).mean()
        return batch_roi_loss