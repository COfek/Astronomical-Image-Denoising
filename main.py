# === Standard Libraries ===
from pathlib import Path
import numpy as np
import torch

# === PyTorch & TorchVision ===
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# === Custom Dataset & Models ===
from Data.dataloader import AstroDenoisingDataset
from Data.simulate_astronomy_dataset import download_and_process_sdss, download_and_process_div2k
from Models.unet import UNet
from Models.ViT import ViTDenoiser
from Models.DnCNN import DnCNN

# === Training Functions ===
from Scripts.train_unet import train_validate_test as train_unet
from Scripts.train_ViT import train_validate_test_vit
from Scripts.train_DnCNN import train_validate_test_dncnn

# === Plotting & Classic Denoisers ===
from Scripts.plots import plot_classic_denoising, plot_denoising
from Scripts.red_inference import red_restore
from Scripts.tikhonov_restore import tikhonov_restore
from Scripts.TV_restore import tv_restore

# === CONFIGURATION ===
VERBOSE = True
DO_TRAIN = True
MODEL_TO_TRAIN = "UNet"  # Options: "UNet", "DnCNN", "ViT"
DOWNLOAD_AND_PROCESS_SDSS = False
DOWNLOAD_AND_PROCESS_DIV2K = True  # Not implemented in this script
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(f"output/{MODEL_TO_TRAIN}/best_{MODEL_TO_TRAIN}.pth")

# === MAIN FUNCTION ===
def main():
    # Optional: download and process data
    if DOWNLOAD_AND_PROCESS_SDSS:
        download_and_process_sdss()
    if DOWNLOAD_AND_PROCESS_DIV2K:
        download_and_process_div2k()

    # === Load Dataset ===
    dataset = AstroDenoisingDataset("Data/clean", "Data/noisy", transform=transforms.ToTensor())
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # RED expects single image


    # === Select Model ===
    if MODEL_TO_TRAIN == "UNet":
        model = UNet().to(DEVICE)
    elif MODEL_TO_TRAIN == "DnCNN":
        model = DnCNN().to(DEVICE)
    elif MODEL_TO_TRAIN == "ViT":
        model = ViTDenoiser().to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TO_TRAIN}")

    # === Train or Load Model ===
    if DO_TRAIN:
        if MODEL_TO_TRAIN == "UNet":
            train_unet(model, train_loader, val_loader, test_loader)
        elif MODEL_TO_TRAIN == "DnCNN":
            train_validate_test_dncnn(model, train_loader, val_loader, test_loader, epochs=10)
        elif MODEL_TO_TRAIN == "ViT":
            train_validate_test_vit(model, train_loader, val_loader, test_loader, epochs=20)
    else:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        if VERBOSE:
            print(f"✅ Loaded pre-trained {MODEL_TO_TRAIN} from {MODEL_PATH}")

    # === Load One Test Image & PSF Kernel ===
    noisy_img, clean_img = next(iter(test_loader))
    psf_np = np.load("Data/psf.npy")
    psf_tensor = torch.tensor(psf_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape [1, 1, H, W]

    # === RED Inference ===
    if VERBOSE:
        print("Performing RED inference on one test image...")
    restored_image = red_restore(
        y=noisy_img.to(DEVICE),
        denoiser=model,
        kernel=psf_tensor.to(DEVICE),
        lambda_=0.1,
        alpha=0.1,
        max_iter=1000,
        verbose=True,
        device=DEVICE,
        x_clean=clean_img.to(DEVICE)
    )

    # === Plot RED Result ===
    plot_denoising(
        noisy_img, clean_img, restored_image,
        save_path=f"output/rid_results/{MODEL_TO_TRAIN}_denoising_comparison.png"
    )

    # === Compare Classic Methods ===
    if VERBOSE:
        print("Comparing classic denoising methods...")
    restored_tikhonov = tikhonov_restore(noisy_img, kernel=psf_tensor, ground_truth=clean_img)
    restored_tv = tv_restore(noisy_img, kernel=psf_tensor, ground_truth=clean_img)

    plot_classic_denoising(
        noisy_img, clean_img, restored_tikhonov, restored_tv,
        save_path="output/rid_results/classic_denoising_comparison.png"
    )


# === ENTRY POINT ===
if __name__ == "__main__":
    main()
