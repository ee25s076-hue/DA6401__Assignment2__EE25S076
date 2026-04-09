import os
import torch
from torch.utils.data import DataLoader, Subset
from data.pets_bbox_dataset import OxfordIIITPetBBoxDataset, train_transform, val_transform
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss
import wandb

os.makedirs("weights", exist_ok=True)


def compute_iou(pred, target, eps=1e-6):
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    tgt_x1 = target[:, 0] - target[:, 2] / 2
    tgt_y1 = target[:, 1] - target[:, 3] / 2
    tgt_x2 = target[:, 0] + target[:, 2] / 2
    tgt_y2 = target[:, 1] + target[:, 3] / 2

    inter = (torch.min(pred_x2, tgt_x2) - torch.max(pred_x1, tgt_x1)).clamp(0) * \
            (torch.min(pred_y2, tgt_y2) - torch.max(pred_y1, tgt_y1)).clamp(0)

    pred_area = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
    tgt_area = (tgt_x2 - tgt_x1).clamp(0) * (tgt_y2 - tgt_y1).clamp(0)

    iou = inter / (pred_area + tgt_area - inter + eps)
    return iou.mean().item()


def compute_map(pred, target, iou_threshold=0.5, eps=1e-6):
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    tgt_x1 = target[:, 0] - target[:, 2] / 2
    tgt_y1 = target[:, 1] - target[:, 3] / 2
    tgt_x2 = target[:, 0] + target[:, 2] / 2
    tgt_y2 = target[:, 1] + target[:, 3] / 2

    inter = (torch.min(pred_x2, tgt_x2) - torch.max(pred_x1, tgt_x1)).clamp(0) * \
            (torch.min(pred_y2, tgt_y2) - torch.max(pred_y1, tgt_y1)).clamp(0)

    pred_area = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
    tgt_area = (tgt_x2 - tgt_x1).clamp(0) * (tgt_y2 - tgt_y1).clamp(0)

    iou = inter / (pred_area + tgt_area - inter + eps)
    return (iou >= iou_threshold).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, config):
    model.train()
    total_loss = total_iou = total_map = 0.0

    for i, (images, bboxes) in enumerate(loader):
        images = images.to(device)
        bboxes = bboxes.to(device)

        preds = model(images)
        loss = criterion(preds, bboxes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iou = compute_iou(preds.detach(), bboxes)
        map50 = compute_map(preds.detach(), bboxes)

        total_loss += loss.item()
        total_iou += iou
        total_map += map50

        if i % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{config.epochs} | Batch {i}/{len(loader)} "
                f"| Loss: {loss.item():.4f} | IoU: {iou:.4f} | mAP@50: {map50:.4f}"
            )

            wandb.log(
                {
                    "batch/train_loss": loss.item(),
                    "batch/train_iou": iou,
                    "batch/train_map": map50,
                    "batch/step": epoch * len(loader) + i,
                }
            )

    n = len(loader)
    return total_loss / n, total_iou / n, total_map / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_iou = total_map = 0.0

    with torch.no_grad():
        for images, bboxes in loader:
            images = images.to(device)
            bboxes = bboxes.to(device)

            preds = model(images)
            loss = criterion(preds, bboxes)

            total_loss += loss.item()
            total_iou += compute_iou(preds, bboxes)
            total_map += compute_map(preds, bboxes)

    n = len(loader)
    return total_loss / n, total_iou / n, total_map / n


def main():
    wandb.init(
        project="oxford-pets-localization",
        name="finetuned",
        config={
            "epochs": 150,
            "batch_size": 128,
            "learning_rate": 1e-4,
            "architecture": "VGG11Localizer",
            "loss": "IoULoss",
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "encoder_init": "classifier_weights",
            "freeze_encoder": False,
        },
    )

    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_full = OxfordIIITPetBBoxDataset(
        images_dir="data/images",
        xml_dir="data/annotations/xmls",
        list_file="data/annotations/trainval.txt",
        transform=train_transform,
    )

    val_full = OxfordIIITPetBBoxDataset(
        images_dir="data/images",
        xml_dir="data/annotations/xmls",
        list_file="data/annotations/trainval.txt",
        transform=val_transform,
    )

    total = len(train_full)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total, generator=generator).tolist()

    train_dataset = Subset(train_full, indices[:train_size])
    val_dataset = Subset(val_full, indices[train_size: train_size + val_size])
    test_dataset = Subset(val_full, indices[train_size + val_size:])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = VGG11Localizer().to(device)

    classifier_path = "weights/classifier_best.pth"
    if os.path.exists(classifier_path):
        ckpt = torch.load(classifier_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        encoder_state = {k: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)

        print(f"Loaded pretrained encoder from {classifier_path}")
        print(f"  encoder keys loaded : {len(encoder_state)}")
        print(f"  missing (regressor) : {len(missing)}")
        print(f"  unexpected keys      : {len(unexpected)}")
    else:
        print(f"WARNING: {classifier_path} not found - encoder stays random")

    for p in model.encoder.parameters():
        p.requires_grad = True

    criterion = IoULoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-5,
    )

    torch.backends.cudnn.benchmark = True
    wandb.watch(model, log="gradients", log_freq=50)

    best_val_map = 0.0
    weights_path = "weights/localizer_best.pth"

    for epoch in range(config.epochs):
        train_loss, train_iou, train_map = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )

        val_loss, val_iou, val_map = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{config.epochs} Summary")
        print(f"  Train - Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | mAP@50: {train_map:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}   | IoU: {val_iou:.4f}   | mAP@50: {val_map:.4f}\n")

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/iou": train_iou,
                "train/map50": train_map,
                "val/loss": val_loss,
                "val/iou": val_iou,
                "val/map50": val_map,
                "train/lr": scheduler.get_last_lr()[0],
            }
        )

        if val_map > best_val_map:
            best_val_map = val_map

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map": best_val_map,
                    "val_iou": val_iou,
                    "freeze_encoder": False,
                    "encoder_init": "classifier_weights",
                },
                weights_path,
            )

            print(f"  Saved best model -> {weights_path}  (val mAP@50: {best_val_map:.4f})")
            wandb.log({"best/val_map50": best_val_map})

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\nLoaded best weights from epoch {checkpoint['epoch']}")

    test_loss, test_iou, test_map = evaluate(model, test_loader, criterion, device)

    print(f"Test - Loss: {test_loss:.4f} | IoU: {test_iou:.4f} | mAP@50: {test_map:.4f}")

    wandb.log(
        {
            "test/loss": test_loss,
            "test/iou": test_iou,
            "test/map50": test_map,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
