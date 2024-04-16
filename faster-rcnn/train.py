import torch
import time
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from object_detection_model import ObjectDetectionModel
from od_losses import RPNLoss, ROILoss
import torchvision.transforms as T
from od_utils import assign_ground_truth_to_anchors, assign_ground_truth_to_proposals
from torchvision.datasets import CocoDetection
from dataset import ODCOCODataset, collate_fn


params = {
    "batch_size": 4,
    "num_classes": 2,  # foreground + background
    "coco_root": "/Users/alisher/Desktop/tx-dev-assignment/solution/generated_images/images/",
    "coco_path": "/Users/alisher/Desktop/tx-dev-assignment/solution/generated_images/labels/coco_annotations.json",
    "backbone": {'arch': 'resnet50', 'layer': 'layer4', 'freeze': True},

    "rpn": {'cascade': False, 'scales': [32, 64, 128], 'aspect_ratios': [0.5, 1, 2]},
    "roi": {'num_classes': 2, 'pool_output_size': 7}

}



learning_rate = 1e-6
num_epochs = 10
batch_size = params["batch_size"]

train_transforms = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ODCOCODataset(root=params["coco_root"], 
                             annotation=params["coco_path"],
                             transforms=train_transforms,
                             )

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                          collate_fn=collate_fn)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # MPS for Macbook M1
device = torch.device("cpu")

model = ObjectDetectionModel(params).to(device)

rpn_loss = RPNLoss(cls_loss_weight=1.0, bbox_loss_weight=1.0, batch_size=batch_size).to(device)
roi_loss = ROILoss(cls_loss_weight=1.0, bbox_loss_weight=1.0, batch_size=batch_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    iteration = 0
    for images, gt_bboxes, gt_labels in train_loader:
        iteration += 1
        images = images.to(device)

        objectness_logits, bbox_reg, proposals, cls_logits, bbox_preds, anchors = model(images)

        anchor_labels, anchor_gt_boxes = assign_ground_truth_to_anchors(anchors, gt_bboxes, positive_threshold=0.5, negative_threshold=0.3)
        rpn_loss_value = rpn_loss(objectness_logits, bbox_reg, anchor_labels, anchor_gt_boxes)
        
        proposal_labels, bbox_deltas = assign_ground_truth_to_proposals(proposals, gt_bboxes, gt_labels)
        roi_loss_value = roi_loss(cls_logits, bbox_preds, proposal_labels, bbox_deltas)

        if torch.isnan(rpn_loss_value):
            print("NaN detected @ RPN Loss")
        
        if torch.isnan(roi_loss_value):
            print("NaN detected @ ROI Loss")

        loss = rpn_loss_value + roi_loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1}/{num_epochs} >>> iteration: {iteration} / {len(train_loader)} >>>  loss: {loss.item()}")
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}")

print("Training complete.")

