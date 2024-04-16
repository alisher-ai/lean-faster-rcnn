import torch
import torch.nn as nn
from torchvision import ops
import time


class ROIHeads(nn.Module):
    def __init__(self, params):
        super(ROIHeads, self).__init__()

        self.in_channels = params['in_channels']
        self.num_classes = params['num_classes']
        self.roi_pool_output_size = params['pool_output_size']
        self.bs = params['batch_size']
        self.fc_cls = nn.Linear(self.in_channels * self.roi_pool_output_size ** 2, self.num_classes)
        self.fc_bbox = nn.Linear(self.in_channels * self.roi_pool_output_size ** 2, self.num_classes * 4) 


    def forward(self, feature_maps, proposals_list):
        all_cls_logits = []
        all_bbox_preds = []

        for i in range(self.bs):
            proposals = proposals_list[i]
            batch_indices = torch.full((proposals.shape[0], 1), i, device=proposals.device, dtype=proposals.dtype)  
            proposals_with_batch_indices = torch.cat((batch_indices, proposals), dim=1)
            roi_features = ops.roi_align(feature_maps.to("cpu"), proposals_with_batch_indices.to("cpu"), output_size=self.roi_pool_output_size)
            x = roi_features.view(roi_features.size(0), -1).to(feature_maps.device)
            cls_logits = self.fc_cls(x)
            bbox_preds = self.fc_bbox(x)

            all_cls_logits.append(cls_logits)
            all_bbox_preds.append(bbox_preds)

        return all_cls_logits, all_bbox_preds
