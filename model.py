import torch
from torch import nn


class SpinItModel(nn.Module):
    def __init__(self, params_count) -> None:
        super().__init__()
        self._linear = nn.Linear(params_count, params_count)
        self._relu = nn.ReLU(inplace=True)
    
    def forward(self, beta, tree_tensor):
        return torch.clamp(self._relu(self._linear(beta)), max=1)