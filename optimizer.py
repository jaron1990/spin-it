import torch
from octree import Octree, OctreeTensorHandler
from scipy.optimize import minimize
import numpy as np
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
from torch.optim import Adam


class QPOptimizer:
    def __init__(self, name, args) -> None:
        self.iter=0
        if name == "Adam":
            self._opt = Adam(**args)
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
    
    def __call__(self, beta: torch.Tensor, s_total: torch.Tensor, loss_score: torch.Tensor):
        bounds = [(0, 1)] * len(beta)
        problem = [{'type': 'eq', 'fun': self._constraint_s_x, 'args':(s_total,)},           
                    {'type': 'eq', 'fun': self._constraint_s_y, 'args':(s_total,)},
                    {'type': 'eq', 'fun': self._constraint_s_xz, 'args':(s_total,)},
                    {'type': 'eq', 'fun': self._constraint_s_yz, 'args':(s_total,)}]

        result = minimize(self.loss, beta, bounds=bounds, constraints=problem, method='SLSQP', args=(tree_tensor,), options={'disp': True})
        print(result)
        tree_tensor = OctreeTensorHandler.set_internal_beta(tree_tensor, result.x)
