import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

# === ViT Components ===
class ViTDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8), num_layers=2
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)         # [B, 64, H, W]
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        x = self.transformer(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)
        x = self.decoder(x)
        return x

