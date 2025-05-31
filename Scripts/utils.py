from skimage.metrics import structural_similarity as ssim
import torch


def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def compute_ssim_batch(batch1, batch2):
    batch1 = batch1.squeeze(1).cpu().numpy()
    batch2 = batch2.squeeze(1).cpu().numpy()
    ssim_sum = 0
    for i in range(len(batch1)):
        ssim_sum += ssim(batch1[i], batch2[i], data_range=1.0)
    return ssim_sum / len(batch1)