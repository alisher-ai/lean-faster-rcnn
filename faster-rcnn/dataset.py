import os
import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO

class ODCOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
    
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        num_objs = len(coco_annotation)

        boxes = []
        ann_ = self._resize_anns(coco_annotation.copy(), 256. / 1000)
        for i in range(num_objs):
            xmin = ann_[i]['bbox'][0]
            ymin = ann_[i]['bbox'][1]
            xmax = xmin + ann_[i]['bbox'][2]
            ymax = ymin + ann_[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, boxes, labels
    
    def _resize_anns(self, anns, scale):
        for ann in anns:
            ann['bbox'] = [x * scale for x in ann['bbox']]
        return anns
    
    def __len__(self):
        return len(self.ids)
    
def collate_fn(data):
    images, boxes, labels = list(zip(*data))
    images = list(images)
    boxes = list(boxes)
    labels = list(labels)
    images = torch.stack(images, dim=0)
    return images, boxes, labels