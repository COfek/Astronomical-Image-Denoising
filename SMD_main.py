# === Standard Libraries ===
from pathlib import Path
import numpy as np
import torch
import json

# === PyTorch & TorchVision ===
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# === Custom Dataset & Models ===
from Data.dataloader import AstroDenoisingDataset
from Data.simulate_astronomy_dataset import download_and_process_sdss
from Models.unet import UNet
from Models.ViT import ViTDenoiser
from Models.BM3D import BM3DDenoiser  # optional fallback baseline

# === Plotting & Utilities ===
from Scripts.plots import side_by_side_plot
from Scripts.utils import compute_psnr, compute_ssim
from Scripts.smd import smd_denoise  # <<< NEW import

# === CONFIGURATION ===
VERBOSE = True
DOWNLOAD_AND_PROCESS_SDSS = False
MODEL_TO_USE = "UNet"
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(f"output/{MODEL_TO_USE}/best_{MODEL_TO_USE}.pth")
NOISE_STD = 0.05
NOISE_VARIANCE = NOISE_STD ** 2
SMD_ETA = 0.0003
SMD_STEPS = 10

# === MAIN FUNCTION ===
def main():
    if DOWNLOAD_AND_PROCESS_SDSS:
        download_and_process_sdss()

    dataset = AstroDenoisingDataset("Data/clean", "Data/noisy", transform=transforms.ToTensor())
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # === Load Denoising Model ===
    if MODEL_TO_USE == "UNet":
        model = UNet().to(DEVICE)
    elif MODEL_TO_USE == "ViT":
        model = ViTDenoiser().to(DEVICE)

    else:
        raise ValueError(f"Unsupported model: {MODEL_TO_USE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded pretrained model: {MODEL_PATH}")

    # === Load PSF (optional, not used in SMD directly) ===
    psf_np = np.load("Data/psf.npy")
    psf_tensor = torch.tensor(psf_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    results = {}
    metrics = {}

    for idx, (noisy_img, clean_img) in enumerate(test_loader):
        noisy_img = noisy_img.to(DEVICE)
        clean_img = clean_img.to(DEVICE)

        smd_restored = smd_denoise(
            model=model,
            noisy_img=noisy_img,
            eta=SMD_ETA,
            noise_variance=NOISE_VARIANCE,
            n_steps=SMD_STEPS,
            device=DEVICE
        )

        name = f"SMD-{MODEL_TO_USE}"
        results[name] = smd_restored
        psnr = compute_psnr(smd_restored, clean_img)
        ssim = compute_ssim(smd_restored, clean_img)
        metrics[name] = (psnr, ssim)

        print(f"📈 [{idx+1}] {name} — PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        break  # Only run on one sample for now

    # === Visualize ===
    side_by_side_plot(noisy_img, clean_img, results, metrics,
                    save_path=f"output/smd_results/{MODEL_TO_USE}_smd_comparison.png")

    with open(f"output/smd_results/{MODEL_TO_USE}_metrics.json", "w") as f:
        json.dump({k: [float(v[0]), float(v[1])] for k, v in metrics.items()}, f, indent=2)

if __name__ == "__main__":
    main()
