import os
import gdown
import torch
import torch.nn as nn

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3):
        super().__init__()

        os.makedirs("checkpoints", exist_ok=True)

        classifier_path = "checkpoints/classifier.pth"
        localizer_path = "checkpoints/localizer.pth"
        unet_path = "checkpoints/unet.pth"

        gdown.download(id="1h_ElEDvqggV1yamS0MlzQEVJKcz2zIcx", output=classifier_path, quiet=False)
        gdown.download(id="1y1XkrCTUQixrWjwsE5L3GSbH0mQyETOq", output=localizer_path, quiet=False)
        gdown.download(id="1k9Onp8UubeYU_RgFmEZPvvgcP8_xOqIo", output=unet_path, quiet=False)

        self.classifier_model = VGG11Classifier(
            num_classes=num_breeds,
            in_channels=in_channels
        )

        self.localizer_model = VGG11Localizer(
            in_channels=in_channels
        )

        self.segmenter_model = VGG11UNet(
            num_classes=seg_classes,
            in_channels=in_channels
        )

        self._load_weights(self.classifier_model, classifier_path, "classifier")
        self._load_weights(self.localizer_model, localizer_path, "localizer")
        self._load_weights(self.segmenter_model, unet_path, "segmenter")

    def _load_weights(self, model: nn.Module, path: str, name: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Loaded {name} weights from {path}")

    def forward(self, x: torch.Tensor):
        cls_out = self.classifier_model(x)

        loc_out = self.localizer_model(x)
        H, W = x.shape[-2], x.shape[-1]
        scale = torch.tensor([W, H, W, H], device=loc_out.device, dtype=loc_out.dtype)
        loc_out = loc_out * scale

        seg_out = self.segmenter_model(x)

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }