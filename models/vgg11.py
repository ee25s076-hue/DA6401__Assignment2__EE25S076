from typing import Dict, Tuple, Union
import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, use_batchnorm: bool = True):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        def bn2d(c: int):
            return nn.BatchNorm2d(c) if self.use_batchnorm else nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            bn2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            bn2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            bn2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            bn2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            bn2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            bn2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            bn2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            bn2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    @property
    def block1(self):
        return nn.Sequential(self.conv1, self.pool1)

    @property
    def block2(self):
        return nn.Sequential(self.conv2, self.pool2)

    @property
    def block3(self):
        return nn.Sequential(self.conv3, self.pool3)

    @property
    def block4(self):
        return nn.Sequential(self.conv4, self.pool4)

    @property
    def block5(self):
        return nn.Sequential(self.conv5, self.pool5)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        f1 = self.conv1(x)
        f2 = self.conv2(self.pool1(f1))
        f3 = self.conv3(self.pool2(f2))
        f4 = self.conv4(self.pool3(f3))
        f5 = self.conv5(self.pool4(f4))
        out = self.pool5(f5)

        if return_features:
            features = {
                "fwd1": f1,
                "fwd2": f2,
                "fwd3": f3,
                "fwd4": f4,
                "fwd5": f5,
            }
            return out, features

        return out