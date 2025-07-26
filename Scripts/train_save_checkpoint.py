# === Standard Libraries ===
from pathlib import Path
import numpy as np
import torch
import json

import sys
import os
sys.path.append(str(Path(__file__).resolve().parents[1]))

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

# === Plotting & Classic Denoisers ===
from Scripts.plots import side_by_side_plot
from Scripts.red_inference import red_sd
from Scripts.utils import compute_psnr, compute_ssim

# === CONFIGURATION ===
VERBOSE = True
DO_TRAIN = True
MODEL_TO_TRAIN = "ViT"  # Options: "UNet", "DnCNN", "ViT"
DOWNLOAD_AND_PROCESS_SDSS = False
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(f"output/{MODEL_TO_TRAIN}/best_{MODEL_TO_TRAIN}.pth")

def train_save_chkpt():
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
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # === Select Model ===
    if MODEL_TO_TRAIN == "UNet":
        model = UNet().to(DEVICE)
    elif MODEL_TO_TRAIN == "DnCNN":
        model = DnCNN().to(DEVICE)
    elif MODEL_TO_TRAIN == "ViT":
        model = ViTDenoiser().to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TO_TRAIN}")

    train_validate_test(model, train_loader, val_loader, test_loader, MODEL_TO_TRAIN,
                        epochs=8, batch_size=BATCH_SIZE, verbose=VERBOSE)
    
train_save_chkpt()