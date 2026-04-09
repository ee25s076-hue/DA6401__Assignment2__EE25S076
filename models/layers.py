import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0 <= p <= 1):
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        if self.p == 1:
            return torch.zeros_like(x)

        mask = (torch.rand_like(x) > self.p).float()

        return x * mask / (1 - self.p)