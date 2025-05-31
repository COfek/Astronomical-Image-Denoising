from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from Data.dataloader import AstroDenoisingDataset
from Models.unet import UNet
from Scripts import train_validate_test, red_restore, tikhonov_restore, tv_restore, plot_unet_denoising, plot_classic_denoising
from Data.simulate_astronomy_dataset import gaussian_kernel, download_and_process_sdss  

# === CONFIGURATION ===
VERBOSE: bool = True # Set to False to suppress print statements
DO_TRAIN: bool = False  # Set to False to skip training and load pre-trained model
DOWNLOAD_AND_PROCESS_SDSS = False # Set to False to skip downloading and processing SDSS data
MODEL_PATH = Path("output/unet/best_unet.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8

if __name__ == "__main__":
    if DOWNLOAD_AND_PROCESS_SDSS:
        download_and_process_sdss()

    # === Load Dataset ===
    dataset = AstroDenoisingDataset("Data/clean", "Data/noisy", transform=transforms.ToTensor())
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # RED works on single images

    # === Load or Train UNet ===
    model = UNet().to(DEVICE)
    if DO_TRAIN:
        train_validate_test(model, train_loader, val_loader, test_loader)
    else:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        if VERBOSE:
            print(f"✅ Loaded pre-trained UNet from {MODEL_PATH}")

    # === RED Inference on One Test Image ===
    noisy_img, clean_img = next(iter(test_loader))
    psf_np = np.load("Data/psf.npy")  # Load the exact kernel used
    psf_tensor = torch.tensor(psf_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    restored_image = red_restore(
        y=noisy_img.to(DEVICE),
        denoiser=model,
        kernel=psf_tensor.to(DEVICE),
        lambda_=0.1,
        alpha=0.1,
        max_iter=1000,
        verbose=True,
        device=DEVICE,
        x_clean=clean_img.to(DEVICE)  # ✅ pass clean reference
    )

    plot_unet_denoising(
        noisy_img, clean_img, restored_image,
        save_path="output/rid_results/unet_denoising_comparison.png"
    )

    # === Compare Classic Denoising Methods ===
    if VERBOSE:
        print("Comparing classic denoising methods...")
    restored_tikhonov = tikhonov_restore(noisy_img, kernel=psf_tensor, ground_truth=clean_img)
    restored_tv = tv_restore(noisy_img, kernel=psf_tensor, ground_truth=clean_img)
    
    plot_classic_denoising(
        noisy_img, clean_img, restored_tikhonov, restored_tv,
        save_path="output/rid_results/classic_denoising_comparison.png"
    )