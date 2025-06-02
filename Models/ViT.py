import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

# === ViT Components ===
class ViTBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

# === ViT Denoiser ===
class ViTDenoiser(nn.Module):
    def __init__(self, img_size=256, patch_size=16, dim=256, depth=6):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(1, dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.transformer = nn.Sequential(*[ViTBlock(dim) for _ in range(depth)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 128, 4, stride=2, padding=1),  # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 64→128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),     # 128→256
            nn.Sigmoid()
        )


    def forward(self, x):
        patches = self.patch_embed(x)  # (B, dim, H/ps, W/ps)
        B, D, H, W = patches.shape
        x = patches.flatten(2).transpose(1, 2)  # (B, N, D)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, D, H, W)
        out = self.decoder(x)
        return out



