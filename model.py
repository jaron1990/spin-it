from torch import nn
from torch.nn.parameter import Parameter

class SpinItModel(nn.Module):
    def __init__(self, beta) -> None:
        super().__init__()
        self._beta = Parameter(beta)
    
    def forward(self, s_internal_unstable, s_rest):
        s_internal_sum = (s_internal_unstable.T*self._beta).sum(axis=1)
        s_total = s_rest + s_internal_sum
        return s_total