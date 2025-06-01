import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Dataset ===
class AstroDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.files = sorted(os.listdir(noisy_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.files[idx])
        clean_path = os.path.join(self.clean_dir, self.files[idx])
        noisy = np.array(Image.open(noisy_path).convert("L"), dtype=np.float32) / 255.0
        clean = np.array(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0
        noisy = torch.tensor(noisy).unsqueeze(0)  # (1, H, W)
        clean = torch.tensor(clean).unsqueeze(0)
        return noisy, clean

# === ViT Components ===
class ViTBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

# === ViT Denoiser ===
class ViTDenoiser(nn.Module):
    def __init__(self, img_size=256, patch_size=16, dim=256, depth=6):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(1, dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.transformer = nn.Sequential(*[ViTBlock(dim) for _ in range(depth)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 128, 4, stride=2, padding=1),  # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 64→128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),     # 128→256
            nn.Sigmoid()
        )


    def forward(self, x):
        patches = self.patch_embed(x)  # (B, dim, H/ps, W/ps)
        B, D, H, W = patches.shape
        x = patches.flatten(2).transpose(1, 2)  # (B, N, D)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, D, H, W)
        out = self.decoder(x)
        return out

# === Training ===
def train_model(model, dataloader, epochs, device, lr=1e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for noisy, clean in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(dataloader):.4f}")

# === Visualize Results ===
def show_sample(model, dataset, device, idx=0):
    model.eval()
    noisy, clean = dataset[idx]
    noisy = noisy.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(noisy).cpu().squeeze().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(noisy.squeeze().cpu().numpy(), cmap='gray')
    axs[0].set_title("Noisy")
    axs[1].imshow(clean.squeeze().numpy(), cmap='gray')
    axs[1].set_title("Clean")
    axs[2].imshow(output, cmap='gray')
    axs[2].set_title("Denoised")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# === Main Script ===
if __name__ == "__main__":
    data_root = "Data/astro_simulated_dataset_all"
    noisy_dir = os.path.join(data_root, "noisy")
    clean_dir = os.path.join(data_root, "clean")

    dataset = AstroDataset(noisy_dir, clean_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = ViTDenoiser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, dataloader, epochs=50, device=device)
    show_sample(model, dataset, device, idx=0)
