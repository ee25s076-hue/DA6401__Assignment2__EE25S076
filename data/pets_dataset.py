import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=1.0),
    A.Rotate(limit=15, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


class OxfordIIITPetDataset(Dataset):
    def __init__(self, images_dir, list_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        self.labels_dict = {}
        self.image_files = []

        with open(list_file, "r") as file:
            for line in file:
                if line.startswith("#"):
                    continue

                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                image_name = parts[0] + ".jpg"
                label = int(parts[1]) - 1

                self.labels_dict[image_name] = label
                self.image_files.append(image_name)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))  

        if self.transform:
            augmented = self.transform(image=image)           
            image = augmented["image"]                          

        label = self.labels_dict[img_name]

        return image, label
