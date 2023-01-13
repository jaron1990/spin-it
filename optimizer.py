import torch
from octree import Octree, OctreeTensorHandler
from scipy.optimize import minimize
import numpy as np
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
from torch.optim import Adam
import nlopt
import wandb
from functools import partial
from utils import SVector


class QPOptimizer:
    def __init__(self, algorithm, phi, gamma_i, gamma_c, calc_type, tolerance) -> None:
        self._opt = partial(nlopt.opt, getattr(nlopt, algorithm))

        phi = torch.tensor([phi])
        self._R = torch.tensor([[torch.cos(phi), -torch.sin(phi)],
                                [torch.sin(phi), torch.cos(phi)]])
        self._gamma_i = gamma_i
        self._gamma_c = gamma_c
        self._calc_type = calc_type
        self._tolerance = tolerance

    def _constraint_s_x(self, internal_beta, grad):
        s_internal = OctreeTensorHandler.get_internal_s_vector(self.tree_tensor)
        s_boundary = OctreeTensorHandler.get_boundary_s_vector(self.tree_tensor)
        s = self._calc_total_s(s_internal, s_boundary, internal_beta)
        return s[SVector.X].item()
    def _constraint_s_y(self, internal_beta, grad):
        s_internal = OctreeTensorHandler.get_internal_s_vector(self.tree_tensor)
        s_boundary = OctreeTensorHandler.get_boundary_s_vector(self.tree_tensor)
        s = self._calc_total_s(s_internal, s_boundary, internal_beta)
        return s[SVector.Y].item()
    
    def _constraint_s_xz(self, internal_beta, grad):
        s_internal = OctreeTensorHandler.get_internal_s_vector(self.tree_tensor)
        s_boundary = OctreeTensorHandler.get_boundary_s_vector(self.tree_tensor)
        s = self._calc_total_s(s_internal, s_boundary, internal_beta)
        return s[SVector.XZ].item()
    
    def _constraint_s_yz(self, internal_beta, grad):
        s_internal = OctreeTensorHandler.get_internal_s_vector(self.tree_tensor)
        s_boundary = OctreeTensorHandler.get_boundary_s_vector(self.tree_tensor)
        s = self._calc_total_s(s_internal, s_boundary, internal_beta)
        return s[SVector.YZ].item()
    
    def _calc_total_s(self, s_internal: torch.Tensor, s_boundary: torch.Tensor, internal_beta: torch.Tensor
                      ) -> torch.Tensor:
        s_internal_total = (torch.tensor(internal_beta).unsqueeze(1) * s_internal).sum(axis=0)
        s_boundary_total = s_boundary.sum(axis=0)
        return s_internal_total + s_boundary_total
    
    def _loss(self, internal_beta, grad):
        s_internal = OctreeTensorHandler.get_internal_s_vector(self.tree_tensor)
        s_boundary = OctreeTensorHandler.get_boundary_s_vector(self.tree_tensor)
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

        # wandb.log({
        #     'min_beta': internal_beta.min(),
        #     'max_beta': internal_beta.max(),
        #     'I_a': I_a,
        #     'I_b': I_b,
        #     'I_c': I_c,
        #     'beta[0]': internal_beta[0],
        #     'beta[1]': internal_beta[1],
        #     'beta[2]': internal_beta[2],
        #     'beta[3]': internal_beta[3],
        #     'beta[4]': internal_beta[4],
        #     's_1': s[SVector.ONE],
        #     's_x': s[SVector.X],
        #     's_y': s[SVector.Y],
        #     's_z': s[SVector.Z],
        #     's_xx': s[SVector.XX],
        #     's_yy': s[SVector.YY],
        #     's_zz': s[SVector.ZZ],
        #     's_xy': s[SVector.XY],
        #     's_xz': s[SVector.XZ],
        #     's_yz': s[SVector.YZ],
        #     'f_yoyo': f_yoyo,
        #     'f_top': f_top,
        #     })

        return f_top.item()        
    
    def __call__(self, beta: torch.Tensor, tree_tensor: torch.Tensor) -> torch.Tensor:
        # wandb.init(project='spinit', entity="spinit", config={'optimizer': 'nlopt'})
        self.tree_tensor = tree_tensor
        
        opt = self._opt(len(beta))
        opt.set_lower_bounds(np.zeros(beta.shape))
        opt.set_upper_bounds(np.ones(beta.shape))

        opt.add_equality_constraint(self._constraint_s_x, self._tolerance)
        opt.add_equality_constraint(self._constraint_s_y, self._tolerance)
        opt.add_equality_constraint(self._constraint_s_xz, self._tolerance)
        opt.add_equality_constraint(self._constraint_s_yz, self._tolerance)

        opt.set_min_objective(self._loss)
        opt.set_maxeval(len(beta)+100)
        # opt.set_xtol_abs(0.1)
        return torch.tensor(opt.optimize(beta.numpy()))


