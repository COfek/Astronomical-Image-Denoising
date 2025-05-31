import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path

class AstroDenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.clean_paths = sorted(list(Path(clean_dir).glob("*.png")))
        self.noisy_paths = sorted(list(Path(noisy_dir).glob("*.png")))
        assert len(self.clean_paths) == len(self.noisy_paths), "Mismatch between clean and noisy images"
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_img = Image.open(self.clean_paths[idx]).convert("L")
        noisy_img = Image.open(self.noisy_paths[idx]).convert("L")

        clean_tensor = self.transform(clean_img)
        noisy_tensor = self.transform(noisy_img)

        return noisy_tensor, clean_tensor  # input, target
