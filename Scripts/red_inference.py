import sys
from pathlib import Path
import os

# Add project root to sys.path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from Scripts.utils import compute_psnr, compute_ssim_batch


def apply_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Applies blur operator H to input x."""
    return F.conv2d(x, kernel, padding="same")


def apply_blur_T(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Applies transpose of blur operator Hᵀ to input x."""
    return F.conv2d(x, torch.flip(kernel, dims=[2, 3]), padding="same")


def red_sd(
    y: torch.Tensor,
    denoiser: torch.nn.Module,
    kernel: torch.Tensor,
    lambda_: float = 0.05,
    alpha: float = 0.1,
    max_iter: int = 30,
    verbose: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    x_clean: torch.Tensor = None,  # ✅ NEW: clean reference image
) -> torch.Tensor:
    """
    Perform Regularization by Denoising (RED) image restoration.

    Args:
        y (torch.Tensor): Noisy observed image (shape: [1, 1, H, W]).
        denoiser (torch.nn.Module): Pretrained denoising model (e.g., UNet).
        kernel (torch.Tensor): Convolution kernel for forward model (H).
        lambda_ (float): Regularization strength.
        alpha (float): Step size for gradient update.
        max_iter (int): Number of RED iterations.
        verbose (bool): Whether to print PSNR/SSIM during optimization.
        device (str): 'cuda' or 'cpu'.
        x_clean (torch.Tensor, optional): Clean ground truth image to compare against.

    Returns:
        torch.Tensor: Restored image after RED iterations.
    """
    x = y.clone().detach().to(device)
    y = y.to(device)
    kernel = kernel.to(device)
    denoiser = denoiser.to(device)
    denoiser.eval()

    if x_clean is not None:
        x_clean = x_clean.to(device)

    for i in range(max_iter):
        Hx = apply_blur(x, kernel)
        grad_data = apply_blur_T(Hx - y, kernel)

        with torch.no_grad():
            denoised = denoiser(x)

        grad_prior = x - denoised
        grad = grad_data + lambda_ * grad_prior
        x = x - alpha * grad

        if verbose and (i % 5 == 0 or i == max_iter - 1):
            target = x_clean if x_clean is not None else y
            psnr = compute_psnr(x, target)
            ssim = compute_ssim_batch(x.detach(), target.detach())
            label = "clean" if x_clean is not None else "noisy"
            print(f"[{i+1}/{max_iter}] PSNR ({label}): {psnr:.2f}, SSIM: {ssim:.4f}")

    return x
