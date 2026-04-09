import os
import torch
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="albumentations",  
        label_fields=["labels"],
        clip=True,
        min_visibility=0.5,
    ),
)

val_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="albumentations",
        label_fields=["labels"],
        clip=True,
    ),
)


class OxfordIIITPetBBoxDataset(Dataset):

    def __init__(self, images_dir, xml_dir, list_file, transform=None):
        self.images_dir = images_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.image_files = []

        with open(list_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 1:
                    continue

                image_name = parts[0]
                img_path = os.path.join(images_dir, image_name + ".jpg")
                xml_path = os.path.join(xml_dir, image_name + ".xml")

                if os.path.exists(img_path) and os.path.exists(xml_path):
                    self.image_files.append(image_name)

    def __len__(self):
        return len(self.image_files)

    def _parse_bbox(self, xml_path, img_w, img_h):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bbox_elem = root.find("object/bndbox")

        xmin = int(float(bbox_elem.find("xmin").text))
        ymin = int(float(bbox_elem.find("ymin").text))
        xmax = int(float(bbox_elem.find("xmax").text))
        ymax = int(float(bbox_elem.find("ymax").text))

        xmin = max(0, min(xmin, img_w - 1))
        ymin = max(0, min(ymin, img_h - 1))
        xmax = max(0, min(xmax, img_w - 1))
        ymax = max(0, min(ymax, img_h - 1))

        if xmax <= xmin:
            xmax = min(img_w - 1, xmin + 1)
        if ymax <= ymin:
            ymax = min(img_h - 1, ymin + 1)

        return xmin, ymin, xmax, ymax

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, image_name + ".jpg")
        xml_path = os.path.join(self.xml_dir, image_name + ".xml")

        image = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = image.shape[:2]

        xmin, ymin, xmax, ymax = self._parse_bbox(xml_path, img_w, img_h)

        bbox_norm = [
            xmin / img_w,
            ymin / img_h,
            xmax / img_w,
            ymax / img_h,
        ]

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=[bbox_norm],
                labels=[1],
            )

            image = transformed["image"]
            bboxes = transformed["bboxes"]

            if len(bboxes) == 0:
                x_min, y_min, x_max, y_max = bbox_norm
            else:
                x_min, y_min, x_max, y_max = bboxes[0]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            x_min, y_min, x_max, y_max = bbox_norm

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min

        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        w = float(np.clip(w, 0.0, 1.0))
        h = float(np.clip(h, 0.0, 1.0))

        bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)

        return image, bbox