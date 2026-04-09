import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


class DecoderBlock(nn.Module):

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class VGG11UNet(nn.Module):

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.dec5 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=512)
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=128, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64,  skip_channels=64,  out_channels=32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, features = self.encoder(x, return_features=True)
        f1 = features["fwd1"]   
        f2 = features["fwd2"]   
        f3 = features["fwd3"]   
        f4 = features["fwd4"]   
        f5 = features["fwd5"]   
        
        d5 = self.dec5(out, f5) 
        d4 = self.dec4(d5,  f4)  
        d3 = self.dec3(d4,  f3)  
        d2 = self.dec2(d3,  f2)  
        d1 = self.dec1(d2,  f1)  

        return self.final_conv(d1)  
