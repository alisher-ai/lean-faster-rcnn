from torch import nn
import torchvision.models as models


class Backbone(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.freeze = params['backbone']['freeze']
        self.backbone = self.prepare_backbone(params)
        self.neck = self.prepare_neck(params)

    def forward(self, x):
        feature_map = self.backbone(x)
        if self.neck is not None:
            feature_map = self.neck(feature_map)
        return feature_map

    def prepare_neck(self, params):
        if "neck" not in params:
            return None

    def prepare_backbone(self, params):
        backbone_architecture = params['backbone']['arch']
        backbone_up_to = params['backbone']['layer'] 
        if backbone_architecture == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif backbone_architecture == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif backbone_architecture == 'resnet101':
            model = models.resnet101(pretrained=True)
        else:
            raise ValueError("Unsupported architecture specified")

        modified_layers = []
    
        for name, module in model.named_children():
            if name == backbone_up_to:
                break
            modified_layers.append(module)

        backbone = nn.Sequential(*modified_layers)
        if self.freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        self.out_channels = backbone[-1][-1].conv3.out_channels
        return backbone