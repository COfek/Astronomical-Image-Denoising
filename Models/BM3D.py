import torch
import numpy as np
from bm3d import bm3d
import torch.nn as nn

class BM3DDenoiser(nn.Module):
    def __init__(self, sigma: float = 25.0):
        """
        Args:
            sigma (float): Standard deviation of assumed Gaussian noise, in [0, 255] range.
        """
        super().__init__()
        self.sigma = sigma / 255.0  # Normalize to [0, 1] domain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Image tensor of shape [1, 1, H, W] in [0, 1] range

        Returns:
            torch.Tensor: BM3D-denoised image of same shape
        """
        x_np = x.squeeze().cpu().numpy()  # H x W
        x_np = np.clip(x_np, 0.0, 1.0).astype(np.float32)

        # Apply BM3D denoising
        denoised_np = bm3d(x_np, sigma_psd=self.sigma)

        # Convert back to tensor
        denoised_tensor = torch.tensor(denoised_np, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        return denoised_tensor
