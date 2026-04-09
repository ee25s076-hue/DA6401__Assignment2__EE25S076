import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.encoder = VGG11Encoder(
            in_channels=in_channels,
            use_batchnorm=use_batchnorm,
        )

        def bn1d(c: int):
            return nn.BatchNorm1d(c) if use_batchnorm else nn.Identity()

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 1024),
            bn1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(1024, 512),
            bn1d(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        return x