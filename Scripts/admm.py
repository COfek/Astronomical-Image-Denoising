import torch
import torch.fft
import torch.nn.functional as F

def fft_conv2d(x, kernel_fft):
    return torch.real(torch.fft.ifft2(torch.fft.fft2(x) * kernel_fft))

def admm_reconstruct(
    y: torch.Tensor,
    denoiser: torch.nn.Module,
    kernel: torch.Tensor,
    lambda_: float = 0.05,
    rho: float = 0.1,
    max_iter: int = 50,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False
) -> torch.Tensor:
    B, C, H, W = y.shape
    y = y.to(device)
    denoiser = denoiser.to(device).eval()
    kernel = kernel.to(device)

    # === Prepare FFT of PSF ===
    psf = F.pad(kernel, [0, W - kernel.shape[-1], 0, H - kernel.shape[-2]])
    H_fft = torch.fft.fft2(psf)
    H_conj_fft = torch.conj(H_fft)
    Ht_y_fft = H_conj_fft * torch.fft.fft2(y)

    # === Init Variables ===
    x = y.clone()
    v = x.clone()
    u = torch.zeros_like(x)

    for i in range(max_iter):
        # === x-update in Fourier domain ===
        rhs_fft = Ht_y_fft + rho * torch.fft.fft2(v - u)
        denom = H_conj_fft * H_fft + rho
        x = torch.real(torch.fft.ifft2(rhs_fft / denom)).clamp(0, 1)

        # === v-update: denoising step ===
        with torch.no_grad():
            v = denoiser((x + u).clamp(0, 1))

        # === u-update ===
        u = u + x - v

        if verbose and (i % 10 == 0 or i == max_iter - 1):
            err = torch.norm(x - v).item()
            print(f"[ADMM Iter {i+1}/{max_iter}] ‖x−v‖={err:.4f}")

    return x.clamp(0, 1)
