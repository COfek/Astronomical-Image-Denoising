from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from Scripts.utils import compute_psnr, compute_ssim_batch

def train_validate_test_dncnn(model: nn.Module, train_loader, val_loader, test_loader, 
                            epochs=10, lr=1e-3, output_dir=Path("output/DnCNN"), verbose=True):

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pth"
    best_model_path = output_dir / "best_dncnn.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    start_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']
        if verbose:
            print(f"🔁 Resuming from epoch {start_epoch + 1}, best_val_loss = {best_val_loss:.6f}")

    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []
    train_ssims, val_ssims = [], []

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_psnr, total_ssim = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for noisy, clean in loop:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = loss_fn(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_psnr += compute_psnr(output, clean)
            total_ssim += compute_ssim_batch(output.detach(), clean.detach())
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        avg_train_psnr = total_psnr / len(train_loader)
        avg_train_ssim = total_ssim / len(train_loader)
        train_losses.append(avg_train_loss)
        train_psnrs.append(avg_train_psnr)
        train_ssims.append(avg_train_ssim)

        model.eval()
        val_loss, val_psnr, val_ssim = 0, 0, 0
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
            torch.save(model.state_dict(), best_model_path)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            if verbose:
                print(f"✅ Saved checkpoint and best model at epoch {epoch+1}")

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, PSNR={avg_train_psnr:.2f}, SSIM={avg_train_ssim:.4f} | "
            f"Val Loss={avg_val_loss:.6f}, PSNR={avg_val_psnr:.2f}, SSIM={avg_val_ssim:.4f}")

    # === Plotting ===
    def plot_metric(data, label, ylabel):
        plt.figure()
        plt.plot(data[0], label=f"Train {label}")
        plt.plot(data[1], label=f"Val {label}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Train vs Val {label}")
        plt.legend()
        plt.savefig(output_dir / f"{label.lower()}_plot.png")
        plt.close()

    plot_metric((train_losses, val_losses), "Loss", "MSE Loss")
    plot_metric((train_psnrs, val_psnrs), "PSNR", "PSNR (dB)")
    plot_metric((train_ssims, val_ssims), "SSIM", "SSIM")

    # === Final Test Phase ===
    model.eval()
    test_loss, test_psnr, test_ssim = 0, 0, 0
    with torch.no_grad():
        for noisy, clean in tqdm(test_loader, desc="Testing", leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            test_loss += loss_fn(output, clean).item()
            test_psnr += compute_psnr(output, clean)
            test_ssim += compute_ssim_batch(output.detach(), clean.detach())

    print(f"Test Loss: {test_loss / len(test_loader):.6f}, PSNR: {test_psnr / len(test_loader):.2f} dB, SSIM: {test_ssim / len(test_loader):.4f}")

    # === Save Sample Output ===
    noisy, clean = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        output = model(noisy.to(device)).cpu()

    fig, axes = plt.subplots(3, noisy.shape[0], figsize=(noisy.shape[0] * 2, 6))
    titles = ["Noisy", "Denoised", "Clean"]
    for i in range(3):
        for j in range(noisy.shape[0]):
            img = [noisy, output, clean][i][j][0]
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(titles[i], fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "sample_results.png")
    plt.close()
