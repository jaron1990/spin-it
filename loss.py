import torch
import torch.nn as nn
from utils import SVector

class SpinItLoss(nn.Module):
    def __init__(self, phi: float, gamma_i: float, gamma_c: float, calc_type: str, constraints_weights: list[float]) -> None:
        super().__init__()
        self._calc_type = calc_type
        self._gamma_i = gamma_i
        self._gamma_c = gamma_c
        self._constraints_weights = constraints_weights[:-2]
        phi = torch.tensor([phi])
        cos = torch.cos(phi)
        sin = torch.sin(phi)
        self._R = torch.tensor([[cos, -sin],
                                [sin, cos]])
        self._phi_constraint = lambda x, y: (cos * sin * x + (cos**2 - sin**2) * y) * constraints_weights[-2]
        self._beta_weight = constraints_weights[-1]
        self.relu = nn.ReLU()

    def _calc_constraints_loss(self, total_s, beta):
        constraints_vals = [total_s[SVector.X]**2, total_s[SVector.Y]**2, total_s[SVector.XZ]**2, total_s[SVector.YZ]**2]
        phi_contraint_val = (total_s[SVector.XX] - total_s[SVector.YY], total_s[SVector.XY])
        weighted_constaints = sum([v * w for v, w in zip(constraints_vals, self._constraints_weights)])
        beta_constraints = (max(self.relu(-beta)) + max(self.relu(beta-1))) * self._beta_weight
        return weighted_constaints + self._phi_constraint(*phi_contraint_val) + beta_constraints
    
    def forward(self, model_outputs, beta) -> torch.Tensor:
        s = model_outputs
    
        I = torch.tensor([[s[SVector.YY] + s[SVector.ZZ], -s[SVector.XY], -s[SVector.XZ]],
                          [-s[SVector.XY], s[SVector.XX] + s[SVector.ZZ], -s[SVector.YZ]],
                          [-s[SVector.XZ], -s[SVector.YZ], s[SVector.XX] + s[SVector.YY]]])
        I_CoM = I[:2,:2] - (s[SVector.Z]**2 / s[SVector.ONE]) * torch.eye(2)
        I_CoM_Rot = self._R * I_CoM * self._R.T
        I_a = I_CoM_Rot[0, 0]
        I_b = I_CoM_Rot[1, 1]
        I_c = s[SVector.XX] + s[SVector.YY]
        f_yoyo = self._gamma_i * ((I_a / I_c)**2 + (I_b / I_c)**2)
        
        if self._calc_type == "yoyo":
            return f_yoyo
        f_top = self._gamma_c * (s[SVector.Z]**2) + f_yoyo
        return f_top + self._calc_constraints_loss(s, beta)