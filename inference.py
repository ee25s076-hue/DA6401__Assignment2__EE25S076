import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from PIL import Image
from torchvision import transforms
import wandb

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


BREED_NAMES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx", "American Bulldog", "American Pit Bull Terrier",
    "Basset Hound", "Beagle", "Boxer", "Chihuahua", "English Cocker Spaniel",
    "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese",
    "Japanese Chin", "Keeshond", "Leonberger", "Miniature Pinscher",
    "Newfoundland", "Pomeranian", "Pug", "Saint Bernard", "Samoyed",
    "Scottish Terrier", "Shiba Inu", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier",
]

SEG_COLORS = np.array([
    [72, 209, 104],  
    [20, 20, 20],     
    [220, 60, 60],    
], dtype=np.uint8)

SEG_NAMES = ["Foreground (pet)", "Background", "Boundary"]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_state(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    print(f"Loaded {weights_path} (epoch: {epoch})")
    return model


def load_models(cls_path, loc_path, seg_path, device):
    cls_model = VGG11Classifier(num_classes=37)
    loc_model = VGG11Localizer()
    seg_model = VGG11UNet(num_classes=3)

    cls_model = load_state(cls_model, cls_path, device)
    loc_model = load_state(loc_model, loc_path, device)
    seg_model = load_state(seg_model, seg_path, device)

    return cls_model, loc_model, seg_model


def bbox_norm_cxcywh_to_xyxy_pixels(bbox, img_size=224):
    cx, cy, w, h = bbox
    x1 = (cx - w / 2.0) * img_size
    y1 = (cy - h / 2.0) * img_size
    x2 = (cx + w / 2.0) * img_size
    y2 = (cy + h / 2.0) * img_size

    x1 = max(0, min(img_size - 1, x1))
    y1 = max(0, min(img_size - 1, y1))
    x2 = max(0, min(img_size - 1, x2))
    y2 = max(0, min(img_size - 1, y2))

    return x1, y1, x2, y2


def run_three_models(image_path, cls_model, loc_model, seg_model, device):
    original = Image.open(image_path).convert("RGB")
    display_img = original.resize((224, 224), Image.BILINEAR)

    tensor = preprocess(original).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits = cls_model(tensor)[0]
        loc_pred = loc_model(tensor)[0]
        seg_logits = seg_model(tensor)[0]

    probs = torch.softmax(cls_logits, dim=0)
    breed_idx = probs.argmax().item()
    confidence = probs[breed_idx].item() * 100.0
    breed_name = BREED_NAMES[breed_idx]

    bbox = loc_pred.detach().cpu().numpy()
    bbox = np.clip(bbox, 0.0, 1.0)

    seg_mask = torch.argmax(seg_logits, dim=0).cpu().numpy()
    seg_color = SEG_COLORS[seg_mask]

    return display_img, breed_name, confidence, bbox, seg_mask, seg_color


def make_panel(display_img, breed_name, confidence, bbox, seg_mask, seg_color):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(display_img)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(display_img)
    axes[1].set_title("Classification")
    axes[1].axis("off")
    axes[1].text(
        112, 210, f"{breed_name}\n{confidence:.1f}%",
        ha="center", va="bottom", fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.65),
    )

    axes[2].imshow(display_img)
    axes[2].set_title("Localization")
    axes[2].axis("off")

    x1, y1, x2, y2 = bbox_norm_cxcywh_to_xyxy_pixels(bbox, img_size=224)
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="#48D168", facecolor="none",
    )
    axes[2].add_patch(rect)
    axes[2].text(
        max(0, x1), max(0, y1 - 4), "pet",
        fontsize=8, color="#48D168",
        path_effects=[pe.withStroke(linewidth=2, foreground="black")],
    )

    overlay = np.array(display_img, dtype=np.float32) / 255.0
    mask_f = seg_color.astype(np.float32) / 255.0
    blended = np.clip(0.55 * overlay + 0.45 * mask_f, 0, 1)

    axes[3].imshow(blended)
    axes[3].set_title("Segmentation")
    axes[3].axis("off")

    total = seg_mask.size
    stats = "  |  ".join(
        f"{SEG_NAMES[i]}: {100 * (seg_mask == i).sum() / total:.1f}%"
        for i in range(3)
    )
    fig.text(0.5, 0.01, stats, ha="center", fontsize=8, color="gray")

    legend_patches = [
        patches.Patch(color=SEG_COLORS[i] / 255.0, label=SEG_NAMES[i])
        for i in range(3)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04)
    )

    plt.tight_layout()
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    panel = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()

    plt.close(fig)
    return panel


def main():
    parser = argparse.ArgumentParser(description="W&B showcase using 3 separate task weights")
    parser.add_argument("--images", nargs="+", required=True,
                        help="Paths to 3 novel pet images")
    parser.add_argument("--cls_weights", default="weights/classifier_best.pth",
                        help="Path to classifier checkpoint")
    parser.add_argument("--loc_weights", default="weights/localizer_frozen_best.pth",
                        help="Path to localizer checkpoint")
    parser.add_argument("--seg_weights", default="weights/segmentation_best.pth",
                        help="Path to segmentation checkpoint")
    parser.add_argument("--project", default="oxford-pets-final-showcase",
                        help="W&B project name")
    parser.add_argument("--run_name", default="three-separate-models-showcase",
                        help="W&B run name")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference")
    args = parser.parse_args()

    if len(args.images) < 3:
        raise ValueError("Please provide exactly 3 novel pet images using --images")

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    cls_model, loc_model, seg_model = load_models(
        args.cls_weights, args.loc_weights, args.seg_weights, device
    )

    wandb.init(
        project=args.project,
        name=args.run_name,
        config={
            "classifier_weights": args.cls_weights,
            "localizer_weights": args.loc_weights,
            "segmentation_weights": args.seg_weights,
            "num_images": len(args.images),
            "setup": "3 separate trained models, same input image",
        }
    )

    logged_images = []
    table = wandb.Table(columns=[
        "image_name", "predicted_breed", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "fg_pct", "bg_pct", "boundary_pct"
    ])

    for image_path in args.images[:3]:
        display_img, breed_name, confidence, bbox, seg_mask, seg_color = run_three_models(
            image_path, cls_model, loc_model, seg_model, device
        )

        panel = make_panel(display_img, breed_name, confidence, bbox, seg_mask, seg_color)

        x1, y1, x2, y2 = bbox_norm_cxcywh_to_xyxy_pixels(bbox, img_size=224)

        total = seg_mask.size
        fg_pct = 100 * (seg_mask == 0).sum() / total
        bg_pct = 100 * (seg_mask == 1).sum() / total
        boundary_pct = 100 * (seg_mask == 2).sum() / total

        caption = (
            f"{os.path.basename(image_path)} | "
            f"Breed: {breed_name} ({confidence:.1f}%) | "
            f"BBox(px): [{int(round(x1))}, {int(round(y1))}, {int(round(x2))}, {int(round(y2))}]"
        )

        logged_images.append(wandb.Image(panel, caption=caption))

        table.add_data(
            os.path.basename(image_path),
            breed_name,
            round(confidence, 2),
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
            round(float(fg_pct), 2),
            round(float(bg_pct), 2),
            round(float(boundary_pct), 2),
        )

        print(f"\nImage: {image_path}")
        print(f"  Classification : {breed_name} ({confidence:.1f}%)")
        print(f"  Bounding box px: x1={int(round(x1))}, y1={int(round(y1))}, x2={int(round(x2))}, y2={int(round(y2))}")
        print(f"  Segmentation   : FG={fg_pct:.1f}% BG={bg_pct:.1f}% Boundary={boundary_pct:.1f}%")

    wandb.log({
        "final_showcase/images": logged_images,
        "final_showcase/prediction_table": table,
    })

    wandb.finish()
    print("\nLogged 3 final showcase outputs to W&B.")


if __name__ == "__main__":
    main()