import torch
import torch.nn.functional as F

def admm_reconstruct(
    y: torch.Tensor,
    denoiser: torch.nn.Module,
    kernel: torch.Tensor,
    lambda_: float = 0.05,
    rho: float = 0.1,  # <- renamed from beta
    max_iter: int = 100,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False
) -> torch.Tensor:
    """
    ADMM-based image reconstruction using a pretrained denoiser model.

    Args:
        y (torch.Tensor): Observed noisy+blurred image, shape [1, 1, H, W]
        denoiser (nn.Module): Pretrained denoising model
        kernel (torch.Tensor): PSF kernel, shape [1, 1, kh, kw]
        lambda_ (float): Regularization weight (unused)
        rho (float): ADMM penalty parameter
        max_iter (int): Maximum number of iterations
        device (torch.device): CUDA or CPU
        verbose (bool): Print iteration info

    Returns:
        torch.Tensor: Reconstructed image, same shape as y
    """
    denoiser.eval()
    y = y.to(device)
    kernel = kernel.to(device)
    pad = kernel.shape[-1] // 2

    # Define H and H^T
    H = lambda x: F.conv2d(x, kernel, padding=pad)
    Ht = lambda x: F.conv2d(x, kernel.flip(-1).flip(-2), padding=pad)

    # Initialize variables
    x = Ht(y)  # start with pseudo-inverse
    v = x.clone()
    u = torch.zeros_like(x)

    # Precompute HtH (approximate normalization)
    ones = torch.ones_like(y)
    HtH = F.conv2d(ones, kernel**2, padding=pad) + 1e-6

    for i in range(max_iter):
        # x-update
        rhs = Ht(y) + rho * (v - u)
        x = rhs / (HtH + rho)

        # v-update (denoising)
        with torch.no_grad():
            v = denoiser((x + u).clamp(0, 1))

        # u-update
        u = u + x - v

        if verbose and (i % 10 == 0 or i == max_iter - 1):
            err = torch.norm(x - v).item()
            print(f"[ADMM Iter {i+1}/{max_iter}] ‖x−v‖={err:.4f}")

    return x.clamp(0, 1)
