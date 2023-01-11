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


class QPOptimizer:
    def __init__(self, name, args) -> None:
        self.iter=0
        self.name=name
        if name == "Adam":
            self._opt = partial(Adam,**args)
        elif name == "nlopt":
            self._opt = partial(nlopt.opt, nlopt.LN_COBYLA)
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
    
    def _run_nlopt(self, beta: torch.Tensor, s_total: torch.Tensor, loss_score: torch.Tensor):
        opt =self._opt(len(beta))
        tolerance = 1e-6
        opt.set_lower_bounds(np.zeros(beta.shape))
        opt.set_upper_bounds(np.ones(beta.shape))

        opt.add_equality_constraint(self._constraint_s_x, tolerance)
        opt.add_equality_constraint(self._constraint_s_y, tolerance)
        opt.add_equality_constraint(self._constraint_s_xz, tolerance)
        opt.add_equality_constraint(self._constraint_s_yz, tolerance)

        # self.octree = octree

        opt.set_min_objective(self.loss)
        opt.set_maxeval(len(beta)+50)
        # opt.set_xtol_abs(0.1)
        optimal_beta = opt.optimize(beta)
        print('finished optimization step')

    
    def __call__(self, beta: torch.Tensor, s_total: torch.Tensor, loss_score: torch.Tensor):
        if self.name=="Adam":
            raise NotImplementedError('implement Adam')
        elif self.name=="nlopt":
            self._run_nlopt(beta, s_total, loss_score)
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