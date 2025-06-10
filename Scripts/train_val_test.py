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


def train_validate_test(model: nn.Module,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        test_loader: DataLoader,
                        model_name: str,
                        epochs: int = 10,
                        batch_size: int = 8,
                        verbose: bool = False) -> None:
    """
    Trains a denoising model (e.g., UNet) using provided training, validation, and test data loaders.

    Args:
        model (Module): The PyTorch model to train and evaluate.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        NoReturn: This function does not return anything. It saves training logs, model weights,
        evaluation metrics, and sample output images to the 'output/<model_name>' directory.
    """   
    OUTPUT_DIR = Path(f"output/{model_name.lower()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / f"best_{model_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []
    train_ssims, val_ssims = [], []

    for epoch in range(epochs):
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
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
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

    fig, axes = plt.subplots(3, batch_size, figsize=(batch_size * 2, 6))
    row_titles = ["Noisy", "Denoised", "Clean"]
    for row, title in enumerate(row_titles):
        for col in range(batch_size):
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
