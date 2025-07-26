import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Tuple


class AstroDenoisingDataset(Dataset):
    """
    Custom PyTorch Dataset for astronomical image denoising.

    This dataset loads paired grayscale images: clean targets and noisy inputs.
    The files must be PNG images stored in two parallel folders with identical filenames.

    Args:
        clean_dir (str or Path): Directory containing clean (target) images.
        noisy_dir (str or Path): Directory containing noisy (input) images.
        transform (callable, optional): Optional transformation to apply to both images (default: ToTensor).

    Returns:
        A tuple (noisy_tensor, clean_tensor) where:
            - noisy_tensor: input image with noise (shape: [1, H, W])
            - clean_tensor: ground truth image (shape: [1, H, W])
    """
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str,
        transform: Optional[Callable] = None
    ) -> None:
        self.clean_paths = sorted(Path(clean_dir).glob("*.png"))
        self.noisy_paths = sorted(Path(noisy_dir).glob("*.png"))
        assert len(self.clean_paths) == len(self.noisy_paths), "Mismatch between clean and noisy images"
        self.transform = transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_img = Image.open(self.clean_paths[idx]).convert("L")
        noisy_img = Image.open(self.noisy_paths[idx]).convert("L")

        clean_tensor = self.transform(clean_img)
        noisy_tensor = self.transform(noisy_img)

        return noisy_tensor, clean_tensor
