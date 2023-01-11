import torch
import torch.nn as nn
from octree import OctreeTensorMapping


class SVector:
    ONE = OctreeTensorMapping.S_1 - OctreeTensorMapping.S_1
    X = OctreeTensorMapping.S_X - OctreeTensorMapping.S_1
    Y = OctreeTensorMapping.S_Y - OctreeTensorMapping.S_1
    Z = OctreeTensorMapping.S_Z - OctreeTensorMapping.S_1
    XY = OctreeTensorMapping.S_XY - OctreeTensorMapping.S_1
    XZ = OctreeTensorMapping.S_XZ - OctreeTensorMapping.S_1
    YZ = OctreeTensorMapping.S_YZ - OctreeTensorMapping.S_1
    XX = OctreeTensorMapping.S_XX - OctreeTensorMapping.S_1
    YY = OctreeTensorMapping.S_YY - OctreeTensorMapping.S_1
    ZZ = OctreeTensorMapping.S_ZZ - OctreeTensorMapping.S_1


class SpinItLoss(nn.Module):
    def __init__(self, phi, gamma_i, gamma_c, calc_type) -> None:
        super().__init__()
        self._calc_type = calc_type
        self._gamma_i = gamma_i
        self._gamma_c = gamma_c
        self._R = torch.tensor([[torch.cos(phi), -torch.sin(phi)],
                                [torch.sin(phi), torch.cos(phi)]])
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        I = torch.tensor([[s[SVector.YY], s[SVector.ZZ], -s[SVector.XY], -s[SVector.XZ]],
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