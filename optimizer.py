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
    def __init__(self, name, args) -> None:
        self.iter=0
        self._name=name
        if name == "Adam":
            self._opt = partial(Adam, **args)
        elif name == "nlopt":
            algorithm = getattr(nlopt, args["algorithm"])
            self._opt = partial(nlopt.opt, algorithm)
        else:
            raise NotImplementedError("Only Adam implemented")

    def _constraint_s_x(self, s_total):
        return s_total[SVector.X]
    
    def _constraint_s_y(self, s_total):
        return s_total[SVector.Y]
    
    def _constraint_s_xz(self, s_total):
        return s_total[SVector.XZ]
    
    def _constraint_s_yz(self, s_total):
        return s_total[SVector.YZ]
    
    def _calc_total_s(self, s_internal: torch.Tensor, s_boundary: torch.Tensor, internal_beta: torch.Tensor
                      ) -> torch.Tensor:
        s_internal_total = (s_internal * internal_beta.unsqueeze(-1)).sum(axis=0)
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
        # print(f'iter={self.iter}')
        # self.iter+=1
        # print(f'min_beta={internal_beta.min()}, max_beta={internal_beta.max()}')
        # print(f'Ia/Ic={I_a/I_c}, Ib/Ic={I_b/I_c}')
        # print(f'f_top={f_top}')
        return f_top
    
    def _run_nlopt(self, beta: torch.Tensor, tree_tensor: torch.Tensor, loss_func: torch.Tensor):
        self.tree_tensor = tree_tensor
        phi = 0
        phi = torch.tensor([phi])
        self._R = torch.tensor([[torch.cos(phi), -torch.sin(phi)],
                                [torch.sin(phi), torch.cos(phi)]])
        self._gamma_i = 0.4
        self._gamma_c = 0.5
        self._calc_type = "top"
        
        opt = self._opt(len(beta))
        tolerance = 1e-6
        opt.set_lower_bounds(np.zeros(beta.shape))
        opt.set_upper_bounds(np.ones(beta.shape))

        opt.add_equality_constraint(self._constraint_s_x, tolerance)
        opt.add_equality_constraint(self._constraint_s_y, tolerance)
        opt.add_equality_constraint(self._constraint_s_xz, tolerance)
        opt.add_equality_constraint(self._constraint_s_yz, tolerance)

        opt.set_min_objective(self._loss)
        opt.set_maxeval(len(beta) + 50)
        # opt.set_xtol_abs(0.1)
        optimal_beta = opt.optimize(beta.numpy())
        print('finished optimization step')

    def _run_adam(self, beta, s_total, loss_score):
        optim = self._opt(beta)
        optim.zero_grad()
        optim.step()
    
    def __call__(self, beta: torch.Tensor, tree_tensor: torch.Tensor, loss_func):
        if self._name=="Adam":
            self._run_adam(beta, tree_tensor, loss_func)
        elif self._name == "nlopt":
            self._run_nlopt(beta, tree_tensor, loss_func)
        else:
            raise NotImplementedError('unknown optimizer')

        # wandb.init(project='spinit', entity="spinit", config={'optimizer': 'nlopt', 'tolerance': self.tolerance})

        # wandb.log({
        #             'min_beta': internal_beta.min(),
        #             'max_beta': internal_beta.max(),
        #             'I_a': I_a,
        #             'I_b': I_b,
        #             'I_c': I_c,
        #             's_1': s_total['s_1'],
        #             's_x': s_total['s_x'],
        #             's_y': s_total['s_y'],
        #             's_z': s_total['s_z'],
        #             's_xx': s_total['s_xx'],
        #             's_yy': s_total['s_yy'],
        #             's_zz': s_total['s_zz'],
        #             's_xy': s_total['s_xy'],
        #             's_xz': s_total['s_xz'],
        #             's_yz': s_total['s_yz'],
        #             'f_yoyo': f_yoyo,
        #             'f_top': f_top,
        #             })