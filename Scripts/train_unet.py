import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm
from Data.dataloader import AstroDenoisingDataset
from Models.unet import UNet
from Scripts.utils import compute_psnr, compute_ssim_batch

# === Config ===
VERBOSE = True
EPOCHS = 10
BATCH_SIZE = 8
SEED = 42
OUTPUT_DIR = Path("output/unet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Reproducibility ===
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def train_validate_test(model, train_loader, val_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if VERBOSE:
        print(f"Using device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []
    train_ssims, val_ssims = [], []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        train_ssim = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for noisy, clean in loop:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = loss_fn(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_psnr += compute_psnr(output, clean)
            train_ssim += compute_ssim_batch(output.detach(), clean.detach())
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        avg_train_psnr = train_psnr / len(train_loader)
        avg_train_ssim = train_ssim / len(train_loader)
        train_losses.append(avg_train_loss)
        train_psnrs.append(avg_train_psnr)
        train_ssims.append(avg_train_ssim)

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for noisy, clean in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                val_loss += loss_fn(output, clean).item()
                val_psnr += compute_psnr(output, clean)
                val_ssim += compute_ssim_batch(output.detach(), clean.detach())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_val_psnr)
        val_ssims.append(avg_val_ssim)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_unet.pth")
            if VERBOSE:
                print(f"✅ Saved better model at epoch {epoch+1}")

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, PSNR = {avg_train_psnr:.2f}, SSIM = {avg_train_ssim:.4f} | Val Loss = {avg_val_loss:.6f}, PSNR = {avg_val_psnr:.2f}, SSIM = {avg_val_ssim:.4f}")

    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "loss_plot.png")

    # Plot PSNR curves
    plt.figure()
    plt.plot(train_psnrs, label="Train PSNR")
    plt.plot(val_psnrs, label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Training vs Validation PSNR")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "psnr_plot.png")

    # Plot SSIM curves
    plt.figure()
    plt.plot(train_ssims, label="Train SSIM")
    plt.plot(val_ssims, label="Val SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Training vs Validation SSIM")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "ssim_plot.png")

    # Test Phase
    model.eval()
    test_loss = 0.0
    test_psnr = 0.0
    test_ssim = 0.0
    with torch.no_grad():
        for noisy, clean in tqdm(test_loader, desc="Testing", leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            test_loss += loss_fn(output, clean).item()
            test_psnr += compute_psnr(output, clean)
            test_ssim += compute_ssim_batch(output.detach(), clean.detach())

    print(f"Test Loss: {test_loss / len(test_loader):.6f}, PSNR: {test_psnr / len(test_loader):.2f} dB, SSIM: {test_ssim / len(test_loader):.4f}")

    # Save example output triplet (noisy, output, clean) with labels
    noisy, clean = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        output = model(noisy.to(device)).cpu()

    fig, axes = plt.subplots(3, BATCH_SIZE, figsize=(BATCH_SIZE * 2, 6))
    row_titles = ["Noisy", "Denoised", "Clean"]
    for row, title in enumerate(row_titles):
        for col in range(BATCH_SIZE):
            axes[row, col].imshow([
                noisy, output, clean
            ][row][col][0], cmap='gray')
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=12)

    fig.suptitle("First row: Noisy | Second row: Denoised | Third row: Clean", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_results.png")
    plt.close()
