import torch
import numpy as np
from skimage.restoration import denoise_tv_chambolle
import torch.nn as nn

class TVDenoiser(nn.Module):
    def __init__(self, weight: float = 0.1, n_iter: int = 5):
        """
        Args:
            weight (float): Regularization strength (higher → more smoothing)
            n_iter (int): Number of Chambolle iterations
        """
        super().__init__()
        self.weight = weight
        self.n_iter = n_iter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Image tensor of shape [1, 1, H, W] in [0, 1]

        Returns:
            torch.Tensor: Denoised image of same shape
        """
        x_np = x.squeeze().cpu().numpy()  # shape: H x W
        x_denoised = denoise_tv_chambolle(x_np, weight=self.weight, max_num_iter=self.n_iter)
        return torch.tensor(x_denoised, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
