from torch import nn
from region_proposal_network import RegionProposalNetwork
from backbone import Backbone
from region_of_interest_heads import ROIHeads


class ObjectDetectionModel(nn.Module):
    def __init__(self, params):
        """ model parts are initialized from the given parameters """
        super().__init__()
        self.num_classes = params['num_classes']
        self.backbone = Backbone(params)

        if "rpn" not in params:
            raise ValueError("RPN not specified in the parameters")
        params['rpn']['in_channels'] = self.backbone.out_channels
        if 'batch_size' not in params['rpn']:
            params['rpn']['batch_size'] = params['batch_size']
        
        if 'batch_size' not in params['roi']:
            params['roi']['batch_size'] = params['batch_size']
            
        self.rpn_heads = RegionProposalNetwork(params['rpn'])
        
        if "roi" not in params:
            raise ValueError("ROI not specified in the parameters")
        params['roi']['in_channels'] = self.backbone.out_channels
        self.roi_heads = ROIHeads(params['roi'])


    def forward(self, x):
        feature_maps = self.backbone(x)
        objectness_logits, bbox_reg, proposals, anchors = self.rpn_heads(feature_maps)
        cls_logits, bbox_preds = self.roi_heads(feature_maps, proposals)
        return objectness_logits, bbox_reg, proposals, cls_logits, bbox_preds, anchors
