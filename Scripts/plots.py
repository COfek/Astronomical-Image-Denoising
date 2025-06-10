import matplotlib.pyplot as plt
from pathlib import Path
from torch import Tensor
from typing import Dict, Tuple
import math

def side_by_side_plot(
    y: Tensor,
    clean_img: Tensor,
    results: Dict[str, Tensor],
    metrics: Dict[str, Tuple[float, float]],
    save_path: str = None
) -> None:
    """
    Display denoising results side by side with PSNR and SSIM annotations.

    Args:
        y (Tensor): Noisy image, shape [1, 1, H, W]
        clean_img (Tensor): Ground truth clean image, shape [1, 1, H, W]
        results (Dict[str, Tensor]): Denoised results by method name.
        metrics (Dict[str, Tuple[float, float]]): PSNR and SSIM values keyed by method name.
        save_path (str, optional): If set, saves the figure to this path.
    """
    titles = ["Noisy Input", "Clean Image"] + list(results.keys())
    images = [y, clean_img] + [results[k] for k in results.keys()]
    total = len(images)

    cols = 4
    rows = math.ceil(total / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        img_np = img.squeeze().detach().cpu().numpy()
        plt.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')

        if title in metrics:
            psnr, ssim = metrics[title]
            plt.title(f"{title}\nPSNR: {psnr:.2f}, SSIM: {ssim:.3f}")
        else:
            plt.title(title)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"📷 Saved side-by-side comparison to: {save_path}")

    plt.show()
