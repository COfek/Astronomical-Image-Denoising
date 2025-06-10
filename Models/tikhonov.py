import torch
import torch.nn as nn

class LinearTikhonovDenoiser(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.scale = 1 / (1 + beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x