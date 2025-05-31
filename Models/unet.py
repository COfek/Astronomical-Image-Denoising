# Models/unet.py

import torch
import torch.nn as nn
from torch import Tensor

class UNet(nn.Module):
    """
    A simplified 3-level U-Net architecture for grayscale image denoising.

    Architecture:
    - 3 encoding levels with max pooling
    - Bottleneck with 2 conv layers
    - 3 decoding levels with transposed convs and skip connections
    - Final 1x1 convolution to map to single-channel output

    Input:
        Tensor of shape (B, 1, H, W)

    Output:
        Tensor of shape (B, 1, H, W) (same spatial size)
    """
    def __init__(self) -> None:
        super(UNet, self).__init__()

        def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            """
            Returns a block with two 3x3 convolutions + ReLU.
            """
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        # Encoder blocks
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = conv_block(256, 512)

        # Decoder blocks
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the U-Net.

        Args:
            x (Tensor): Input image tensor of shape (B, 1, H, W)

        Returns:
            Tensor: Output tensor of same shape (B, 1, H, W)
        """
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = self.dec3(torch.cat((dec3, enc3), dim=1))

        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))

        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))

        return self.final(dec1)
