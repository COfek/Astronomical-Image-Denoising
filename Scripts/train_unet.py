import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from Data.dataloader import AstroDenoisingDataset
from Models.unet import UNet

VERBOSE = False  # Toggle print statements


def train_validate_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if VERBOSE:
        print(f"Using device: {device}")
        
    dataset = AstroDenoisingDataset("Data/clean", "Data/noisy", transform=transforms.ToTensor())
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for noisy, clean in loop:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = loss_fn(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                val_loss += loss_fn(output, clean).item()

        print(f"Epoch {epoch+1}: Train Loss = {train_loss / len(train_loader):.6f}, Val Loss = {val_loss / len(val_loader):.6f}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for noisy, clean in tqdm(test_loader, desc="Testing", leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            test_loss += loss_fn(output, clean).item()

    print(f"Test Loss: {test_loss / len(test_loader):.6f}")

if __name__ == '__main__':
    train_validate_test()
