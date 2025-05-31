import torch
import os
import matplotlib.pyplot as plt
from pathlib import Path


def plot_classic_denoising(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    tikhonov: torch.Tensor,
    tv: torch.Tensor,
    save_path: str = "output/compare_tikhonov_tv.png"
) -> None:
    """
    Plot and save a side-by-side comparison of noisy, clean, Tikhonov, and TV restored images.

    Args:
        noisy (torch.Tensor): Noisy input image, shape [1, 1, H, W] or [1, H, W].
        clean (torch.Tensor): Ground truth clean image.
        tikhonov (torch.Tensor): Image restored via Tikhonov regularization.
        tv (torch.Tensor): Image restored via Total Variation (TV) regularization.
        save_path (str): Path to save the comparison figure.
    """
    # Ensure directory exists
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    # Detach and move to CPU if needed
    noisy_np = noisy.squeeze().detach().cpu().numpy()
    clean_np = clean.squeeze().detach().cpu().numpy()
    tikhonov_np = tikhonov.squeeze().detach().cpu().numpy()
    tv_np = tv.squeeze().detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(noisy_np, cmap='gray')
    plt.title("Noisy")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(clean_np, cmap='gray')
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(tikhonov_np, cmap='gray')
    plt.title("Tikhonov")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(tv_np, cmap='gray')
    plt.title("Total Variation")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"✅ Comparison figure saved to {save_path}")



def plot_unet_denoising(
    noisy_img: torch.Tensor,
    clean_img: torch.Tensor,
    restored_img: torch.Tensor,
    titles: tuple[str, str, str] = ("Noisy Image", "Ground Truth Clean Image", "Restored Image"),
    save_path: str = "output/restoration_comparison.png",
    verbose: bool = True
) -> None:
    """
    Plot side-by-side comparison of noisy, clean, and restored images using UNet.

    Args:
        noisy_img (torch.Tensor): Noisy input image, shape [1, 1, H, W] or [1, H, W].
        clean_img (torch.Tensor): Ground truth clean image, shape [1, 1, H, W] or [1, H, W].
        restored_img (torch.Tensor): Restored image from UNet, shape [1, 1, H, W] or [1, H, W].
        titles (tuple[str, str, str]): Titles for the three subplots.
        save_path (str): Path to save the output image.
        verbose (bool): Whether to print success message.
    """
    # Convert tensors to NumPy arrays
    noisy_np = noisy_img.squeeze().detach().cpu().numpy()
    clean_np = clean_img.squeeze().detach().cpu().numpy()
    restored_np = restored_img.squeeze().detach().cpu().numpy()

    # Plot side-by-side
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(noisy_np, cmap='gray')
    plt.title(titles[0])
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(clean_np, cmap='gray')
    plt.title(titles[1])
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(restored_np, cmap='gray')
    plt.title(titles[2])
    plt.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    if verbose:
        print(f"✅ Plot saved to {save_path}")
