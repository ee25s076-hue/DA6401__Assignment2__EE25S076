import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from models.segmentation import VGG11UNet
import wandb

os.makedirs("weights", exist_ok=True)


train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.4),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


class OxfordIIITPetSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, list_file, transform=None):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.transform  = transform
        self.image_files = []

        with open(list_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                image_name = parts[0]
                mask_path  = os.path.join(masks_dir, image_name + ".png")
                if not os.path.exists(mask_path):
                    continue
                self.image_files.append(image_name)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name      = self.image_files[idx]
        img_path  = os.path.join(self.images_dir, name + ".jpg")
        mask_path = os.path.join(self.masks_dir,  name + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path))

        if self.transform:
            result = self.transform(image=image, mask=mask)
            image  = result["image"]  
            mask   = result["mask"]   

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask  = torch.from_numpy(mask.astype("int64")) - 1
        mask  = mask.clamp(0, 2)
        return image, mask


def get_criterion(device):
    """Weighted CrossEntropyLoss.
    Background dominates ~79% of pixels so it must be heavily downweighted.
    foreground (0): 3.0 | background (1): 0.3 | boundary (2): 2.0
    """
    weights = torch.tensor([3.0, 0.3, 2.0]).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def compute_dice(logits, masks, num_classes=3, eps=1e-6):
    """Compute mean Dice score across all classes.

    Args:
        logits: [B, C, H, W] raw model output
        masks:  [B, H, W]    ground-truth class indices (0..C-1)
    Returns:
        mean Dice score (scalar float)
    """
    preds = logits.argmax(dim=1) 
    dice_per_class = []
    for c in range(num_classes):
        pred_c   = (preds == c).float()
        target_c = (masks == c).float()
        intersection = (pred_c * target_c).sum()
        dice = (2.0 * intersection + eps) / (pred_c.sum() + target_c.sum() + eps)
        dice_per_class.append(dice.item())
    return sum(dice_per_class) / num_classes


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, config):
    model.train()
    total_loss = 0.0

    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        loss   = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 20 == 0:
            print(f"Epoch {epoch+1}/{config.epochs} | Batch {i}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")
            wandb.log({
                "batch/train_loss": loss.item(),
                "batch/step":       epoch * len(loader) + i,
            })

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, num_classes=3):
    """Returns avg loss and mean Dice score."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images)
            total_loss += criterion(logits, masks).item()
            total_dice += compute_dice(logits, masks, num_classes=num_classes)

    n = len(loader)
    return total_loss / n, total_dice / n


def main():
    wandb.init(
        project="oxford-pets-segmentation",
        config={
            "epochs":       100,
            "batch_size":   64,
            "lr":           1e-3,
            "num_classes":  3,
            "architecture": "VGG11UNet",
            "loss":         "WeightedCrossEntropyLoss",
            "optimizer":    "Adam",
            "scheduler":    "CosineAnnealingLR",
            "eta_min":      1e-5,
            "augmentation": "albumentations",
            "encoder_init": "classifier_weights",
        }
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_full = OxfordIIITPetSegDataset(
        images_dir="data/images",
        masks_dir="data/annotations/trimaps",
        list_file="data/annotations/trainval.txt",
        transform=train_transform,
    )
    val_full = OxfordIIITPetSegDataset(
        images_dir="data/images",
        masks_dir="data/annotations/trimaps",
        list_file="data/annotations/trainval.txt",
        transform=val_transform,
    )
    test_dataset = OxfordIIITPetSegDataset(
        images_dir="data/images",
        masks_dir="data/annotations/trimaps",
        list_file="data/annotations/test.txt",
        transform=val_transform,
    )

    total      = len(train_full)
    train_size = int(0.8 * total)
    generator  = torch.Generator().manual_seed(42)
    indices    = torch.randperm(total, generator=generator).tolist()

    train_dataset = Subset(train_full, indices[:train_size])
    val_dataset   = Subset(val_full,   indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size,
                              shuffle=False, num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size,
                              shuffle=False, num_workers=8, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = VGG11UNet(num_classes=config.num_classes).to(device)

    classifier_path = "weights/classifier_best.pth"
    if os.path.exists(classifier_path):
        ckpt = torch.load(classifier_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        encoder_state = {
            k.replace("encoder.", "encoder."): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        print(f"Loaded classifier encoder weights from {classifier_path}")
        print(f"  missing keys (decoder, expected): {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")
    else:
        print(f"WARNING: {classifier_path} not found — training encoder from scratch")

    criterion = get_criterion(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )

    torch.backends.cudnn.benchmark = True
    wandb.watch(model, log="gradients", log_freq=50)

    best_val_dice = 0.0
    weights_path  = "weights/segmentation_best.pth"

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        val_loss, val_dice = evaluate(
            model, val_loader, criterion, device, num_classes=config.num_classes
        )
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{config.epochs} Summary")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}\n")

        wandb.log({
            "epoch":      epoch + 1,
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "val/dice":   val_dice,
            "train/lr":   scheduler.get_last_lr()[0],
        })

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             val_loss,
                "val_dice":             best_val_dice,
            }, weights_path)
            print(f"  Saved best model → {weights_path}  (val Dice: {best_val_dice:.4f})")
            wandb.log({"best/val_dice": best_val_dice})

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best weights from epoch {checkpoint['epoch']}")

    test_loss, test_dice = evaluate(
        model, test_loader, criterion, device, num_classes=config.num_classes
    )
    print(f"Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f}")
    wandb.log({"test/loss": test_loss, "test/dice": test_dice})

    wandb.finish()


if __name__ == "__main__":
    main()
