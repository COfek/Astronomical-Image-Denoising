import torch
from torch import Tensor
from typing import Union
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_psnr(img1: Tensor, img2: Tensor) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (Tensor): First image tensor with values in [0, 1].
        img2 (Tensor): Second image tensor with values in [0, 1].

    Returns:
        float: The PSNR value in decibels (dB). Returns infinity if the images are identical.
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def compute_ssim_batch(batch1: Tensor, batch2: Tensor) -> float:
    """
    Compute the mean Structural Similarity Index Measure (SSIM) over a batch of image pairs.

    Args:
        batch1 (Tensor): Batch of first images, shape (N, 1, H, W), values in [0, 1].
        batch2 (Tensor): Batch of second images, shape (N, 1, H, W), values in [0, 1].

    Returns:
        float: The mean SSIM score across the batch.
    """
    batch1_np = batch1.squeeze(1).detach().cpu().numpy()
    batch2_np = batch2.squeeze(1).detach().cpu().numpy()
    ssim_sum = 0.0
    for i in range(len(batch1_np)):
        ssim_sum += ssim(batch1_np[i], batch2_np[i], data_range=1.0)
    return ssim_sum / len(batch1_np)
