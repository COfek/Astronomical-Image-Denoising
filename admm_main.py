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
from Models.DnCNN import DnCNN
from Models.BM3D import BM3DDenoiser
from Models.tikhonov import LinearTikhonovDenoiser
from Models.TV import TVDenoiser

# === Training Functions ===
from Scripts.train_val_test import train_validate_test

# === Plotting & Utilities ===
from Scripts.plots import side_by_side_plot
from Scripts.utils import compute_psnr, compute_ssim
from Scripts.admm import admm_reconstruct

# === CONFIGURATION ===
VERBOSE = True
DO_TRAIN = False
MODEL_TO_TRAIN = "UNet"  # Options: "UNet", "DnCNN", "ViT"
DOWNLOAD_AND_PROCESS_SDSS = False
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(f"output/{MODEL_TO_TRAIN}/best_{MODEL_TO_TRAIN}.pth")

# === MAIN FUNCTION ===
def main():
    if DOWNLOAD_AND_PROCESS_SDSS:
        download_and_process_sdss()

    dataset = AstroDenoisingDataset("Data/clean", "Data/noisy", transform=transforms.ToTensor())
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # === Select and load model ===
    if MODEL_TO_TRAIN == "UNet":
        model = UNet().to(DEVICE)
    elif MODEL_TO_TRAIN == "DnCNN":
        model = DnCNN().to(DEVICE)
    elif MODEL_TO_TRAIN == "ViT":
        model = ViTDenoiser().to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TO_TRAIN}")

    if DO_TRAIN:
        raise NotImplementedError("Training is disabled in this ADMM-only version.")
    else:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Loaded pre-trained {MODEL_TO_TRAIN} from {MODEL_PATH}")

    noisy_img, clean_img = next(iter(test_loader))

    # === Setup Denoisers ===
    denoisers = {
        f"ADMM-{MODEL_TO_TRAIN}": model,
        "ADMM-BM3D": BM3DDenoiser(sigma=25),
        "ADMM-TV": TVDenoiser(weight=0.1, n_iter=5),
        "ADMM-Tikhonov": LinearTikhonovDenoiser(beta=0.05)
    }

    results = {}
    metrics = {}

    print("\n Running ADMM with all denoisers...")
    psf_np = np.load("Data/psf.npy")
    psf_np /= psf_np.sum()
    psf_tensor = torch.tensor(psf_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    for name, denoiser in denoisers.items():
        print(f"\n Running {name}")
        x_restored = admm_reconstruct(
            y=noisy_img.to(DEVICE),
            denoiser=denoiser,
            kernel=psf_tensor.to(DEVICE),
            lambda_=0.001,
            rho=0.1,
            max_iter=100,
            verbose=True,
            device=DEVICE
        )
        results[name] = x_restored
        psnr = compute_psnr(x_restored, clean_img.to(DEVICE))
        ssim = compute_ssim(x_restored, clean_img.to(DEVICE))
        metrics[name] = (psnr, ssim)
        print(f"✅ {name} — PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

    # === Visualize Comparison ===
    side_by_side_plot(noisy_img, clean_img, results, metrics,
                    save_path=f"output/admm_results/{MODEL_TO_TRAIN}_admm_comparison.png")

    with open(f"output/admm_results/{MODEL_TO_TRAIN}_metrics.json", "w") as f:
        json.dump({k: [float(v[0]), float(v[1])] for k, v in metrics.items()}, f, indent=2)

if __name__ == "__main__":
    main()
