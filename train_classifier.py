import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.pets_dataset import OxfordIIITPetDataset, train_transform, val_transform
from models.classification import VGG11Classifier
from sklearn.metrics import f1_score
import wandb

os.makedirs("weights", exist_ok=True)


def evaluate_with_f1(model, loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1


def main():
    wandb.init(
        project="oxford-pets-vgg11",
        name="classifier_training",
        config={
            "epochs":        150,
            "batch_size":    128,    
            "learning_rate": 0.001,
            "momentum":      0.9,
            "weight_decay":  1e-3,
            "num_classes":   37,
            "architecture":  "VGG11",
            "optimizer":     "Adam",
            "scheduler":     "CosineAnnealingLR",
        }
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainval_dataset = OxfordIIITPetDataset(
        images_dir="data/images",
        list_file="data/annotations/trainval.txt",
        transform=train_transform
    )
    test_dataset = OxfordIIITPetDataset(
        images_dir="data/images",
        list_file="data/annotations/test.txt",
        transform=val_transform
    )

    train_size = int(0.8 * len(trainval_dataset))
    val_size   = len(trainval_dataset) - train_size
    train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size,
                              shuffle=False, num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size,
                              shuffle=False, num_workers=8, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model     = VGG11Classifier(num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    betas=(0.9, 0.999),   
    eps=1e-8,
    weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.epochs,
    eta_min=1e-5
    )

    torch.backends.cudnn.benchmark = True
    wandb.watch(model, log="gradients", log_freq=50)

    best_val_f1  = 0.0   
    weights_path = "weights/classifier_best.pth"

    for epoch in range(config.epochs):

        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total   += labels.size(0)

            if i % 20 == 0:
                batch_acc = 100.0 * (predicted == labels).sum().item() / labels.size(0)
                print(f"Epoch {epoch+1}/{config.epochs} | Batch {i}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f} | Acc: {batch_acc:.1f}%")
                wandb.log({
                    "batch/train_loss": loss.item(),
                    "batch/train_acc":  batch_acc,
                    "batch/step":       epoch * len(train_loader) + i,
                })

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = 100.0 * train_correct / train_total

        val_loss, val_acc, val_f1 = evaluate_with_f1(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{config.epochs} Summary")
        print(f"  Train — Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2f}%")
        print(f"  Val   — Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}\n")

        wandb.log({
            "epoch":          epoch + 1,
            "train/loss":     avg_train_loss,
            "train/accuracy": avg_train_acc,
            "val/loss":       val_loss,
            "val/accuracy":   val_acc,
            "val/macro_f1":   val_f1,
            "train/lr":       scheduler.get_last_lr()[0],
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":               best_val_f1,
                "val_acc":              val_acc,
            }, weights_path)
            print(f"  Saved best model (val F1: {best_val_f1:.4f})")
            wandb.log({"best/val_f1": best_val_f1})

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best weights from epoch {checkpoint['epoch']}")

    test_loss, test_acc, test_f1 = evaluate_with_f1(model, test_loader, criterion, device)
    print(f"Test — Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | Macro F1: {test_f1:.4f}")
    wandb.log({
        "test/loss":     test_loss,
        "test/accuracy": test_acc,
        "test/macro_f1": test_f1,
    })

    wandb.finish()


if __name__ == "__main__":
    main()
