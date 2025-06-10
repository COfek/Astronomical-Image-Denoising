import torch
from torch import Tensor
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_psnr(img1: Tensor, img2: Tensor) -> float:
    """
    Compute PSNR between two images (values in [0, 1]).

    Args:
        img1 (Tensor): Shape [1, 1, H, W] or [1, H, W]
        img2 (Tensor): Shape [1, 1, H, W] or [1, H, W]

    Returns:
        float: PSNR in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse.item() == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)
    return psnr.item()


def compute_ssim(img1: Tensor, img2: Tensor) -> float:
    """
    Compute SSIM between two single images of shape [1, 1, H, W] or [1, H, W].

    Returns:
        float: SSIM score
    """
    img1_np = img1.squeeze().detach().cpu().numpy()
    img2_np = img2.squeeze().detach().cpu().numpy()
    return ssim(img1_np, img2_np, data_range=1.0)


def compute_ssim_batch(batch1: Tensor, batch2: Tensor) -> float:
    """
    Compute mean SSIM over a batch of grayscale image pairs.

    Args:
        batch1, batch2: shape (N, 1, H, W), values in [0, 1]

    Returns:
        float: average SSIM across batch
    """
    batch1_np = batch1.squeeze(1).detach().cpu().numpy()
    batch2_np = batch2.squeeze(1).detach().cpu().numpy()

    return np.mean([
        ssim(batch1_np[i], batch2_np[i], data_range=1.0)
        for i in range(batch1_np.shape[0])
    ])
