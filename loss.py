import torch
import torch.nn as nn
from utils import SVector
from octree import OctreeTensorHandler


class SpinItLoss(nn.Module):
    def __init__(self, phi, gamma_i, gamma_c, calc_type) -> None:
        super().__init__()
        self._calc_type = calc_type
        self._gamma_i = gamma_i
        self._gamma_c = gamma_c
        phi = torch.tensor([phi])
        self._R = torch.tensor([[torch.cos(phi), -torch.sin(phi)],
                                [torch.sin(phi), torch.cos(phi)]])
    
    def _calc_total_s(self, s_internal: torch.Tensor, s_boundary: torch.Tensor, internal_beta: torch.Tensor
                      ) -> torch.Tensor:
        s_internal_total = (s_internal * internal_beta.unsqueeze(-1)).sum(axis=0)
        s_boundary_total = s_boundary.sum(axis=0)
        return s_internal_total + s_boundary_total
    
    def forward(self, internal_beta, tree_tensor) -> torch.Tensor:
        s_internal = OctreeTensorHandler.get_internal_s_vector(tree_tensor)
        s_boundary = OctreeTensorHandler.get_boundary_s_vector(tree_tensor)
        s = self._calc_total_s(s_internal, s_boundary, internal_beta)
        
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
        # print(f'iter={self.iter}')
        # self.iter+=1
        # print(f'min_beta={internal_beta.min()}, max_beta={internal_beta.max()}')
        # print(f'Ia/Ic={I_a/I_c}, Ib/Ic={I_b/I_c}')
        # print(f'f_top={f_top}')
        return f_top