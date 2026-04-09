import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout 


class VGG11Localizer(nn.Module):

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.flatten = nn.Flatten()

        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5), 
            nn.Linear(1024, 4),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = self.flatten(features)
        return self.regressor(features)


