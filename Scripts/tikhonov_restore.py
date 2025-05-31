import torch
import torch.nn.functional as F
from Scripts.utils import compute_psnr, compute_ssim_batch

def tikhonov_restore(y: torch.Tensor, kernel: torch.Tensor, lambda_: float = 0.01,
                    alpha: float = 0.1, max_iter: int = 50, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                    verbose: bool = True, ground_truth: torch.Tensor = None) -> torch.Tensor:
    """
    Restore image using Tikhonov regularization (L2).

    Args:
        y (torch.Tensor): Noisy observation [1, 1, H, W]
        kernel (torch.Tensor): Blur kernel [1, 1, kH, kW]
        lambda_ (float): Regularization weight
        alpha (float): Step size
        max_iter (int): Number of iterations
        device (str): 'cuda' or 'cpu'
        verbose (bool): Print PSNR/SSIM
        ground_truth (torch.Tensor): Clean reference image for metrics

    Returns:
        torch.Tensor: Restored image
    """
    x = y.clone().detach().to(device)
    y = y.to(device)
    kernel = kernel.to(device)

    for i in range(max_iter):
        Hx = F.conv2d(x, kernel, padding='same')
        grad_data = F.conv2d(Hx - y, torch.flip(kernel, [2, 3]), padding='same')
        grad_prior = lambda_ * x
        x = x - alpha * (grad_data + grad_prior)

        if verbose and (i % 5 == 0 or i == max_iter - 1):
            if ground_truth is not None:
                psnr = compute_psnr(x, ground_truth.to(device))
                ssim = compute_ssim_batch(x.detach(), ground_truth.to(device))
                print(f"[{i+1}/{max_iter}] PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

    return x
