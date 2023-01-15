import torch
import torch.nn as nn
from utils import SVector, Constraints
from octree import OctreeTensorHandler


class SpinItLoss(nn.Module):
    def __init__(self, phi: float, gamma_i: float, gamma_c: float, calc_type: str, constraints_weights: list[float]) -> None:
        super().__init__()
        self._calc_type = calc_type
        self._gamma_i = gamma_i
        self._gamma_c = gamma_c
        self._constraints_weights = constraints_weights[:-1]
        phi = torch.tensor([phi])
        cos = torch.cos(phi)
        sin = torch.sin(phi)
        self._R = torch.tensor([[cos, -sin],
                                [sin, cos]])
        self._phi_constraint = lambda x, y: (cos * sin * x + (cos**2 - sin**2) * y) * constraints_weights[-1]
    
    def _calc_total_s(self, model_outputs: torch.Tensor, tree_tensor: torch.Tensor, stable_beta_mask: torch.Tensor,
                      unstable_beta_mask: torch.Tensor) -> torch.Tensor:
        s_unstable_total = (OctreeTensorHandler.get_s_vector(tree_tensor)[unstable_beta_mask] * model_outputs[..., None]
                            ).sum(axis=0)
        s_stable_total = OctreeTensorHandler.get_s_vector(tree_tensor)[stable_beta_mask].sum(axis=0)
        s_boundary = OctreeTensorHandler.get_boundary_s_vector(tree_tensor).sum(axis=0)
        return s_unstable_total + s_stable_total + s_boundary
    
    def _calc_constraints_loss(self, total_s):
        constraints_vals = [total_s[SVector.X], total_s[SVector.Y], total_s[SVector.XZ], total_s[SVector.YZ]]
        phi_contraint_val = (total_s[SVector.XX] - total_s[SVector.YY], total_s[SVector.XY])
        weighted_constaints = sum([v * w for v, w in zip(constraints_vals, self._constraints_weights)])
        return weighted_constaints + self._phi_constraint(*phi_contraint_val)
    
    def forward(self, model_outputs, tree_tensor, stable_beta_mask, unstable_beta_mask) -> torch.Tensor:
        s = self._calc_total_s(model_outputs, tree_tensor, stable_beta_mask, unstable_beta_mask)
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
        return f_top + self._calc_constraints_loss(s)