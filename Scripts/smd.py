import torch

def smd_denoise(model, noisy_img, eta, noise_variance, n_steps=1, device="cpu"):
    """
    Perform Score-Matching Denoising (SMD) using a pretrained denoiser model.

    Args:
        model: PyTorch denoiser (e.g., UNet)
        noisy_img (torch.Tensor): shape (1, 1, H, W), normalized to [0,1]
        eta (float): step size
        noise_variance (float): variance of Gaussian noise used in training
        n_steps (int): number of SMD update iterations
        device (str): device for computation

    Returns:
        torch.Tensor: denoised image (same shape as input)
    """
    x = noisy_img.clone().to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(n_steps):
            denoised = model(x)
            score = (denoised - x) / noise_variance
            x = x + eta * score
            x = torch.clamp(x, 0.0, 1.0)

    return x
import torch

def smd_denoise(model, noisy_img, eta, noise_variance, n_steps=1, device="cpu"):
    """
    Perform Score-Matching Denoising (SMD) using a pretrained denoiser model.

    Args:
        model: PyTorch denoiser (e.g., UNet)
        noisy_img (torch.Tensor): shape (1, 1, H, W), normalized to [0,1]
        eta (float): step size
        noise_variance (float): variance of Gaussian noise used in training
        n_steps (int): number of SMD update iterations
        device (str): device for computation

    Returns:
        torch.Tensor: denoised image (same shape as input)
    """
    x = noisy_img.clone().to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(n_steps):
            denoised = model(x)
            score = (denoised - x) / noise_variance
            x = x + eta * score
            x = torch.clamp(x, 0.0, 1.0)

    return x
