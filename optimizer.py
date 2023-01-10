import torch
from octree import Octree, OctreeTensorHandler
from scipy.optimize import minimize
import numpy as np
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt


class QPOptimizer:
    def __init__(self, calc_type: str, gamma_l: float, gamma_i: float, gamma_c: float = None) -> None:
        self._gamma_l = gamma_l
        self._gamma_i = gamma_i
        self._gamma_c = gamma_c
        self._calc_type = calc_type

        self._R = np.eye(2)
        self.iter=0
        self.loss_list = []

    def _calculate_s_total(self, internal_beta, tree_tensor):
        s_internal = OctreeTensorHandler.get_internal_s_vector(tree_tensor)
        s_internal_total = (s_internal.mul(internal_beta, axis=0)).sum()

        s_boundary = OctreeTensorHandler.get_boundary_s_vector(tree_tensor)
        s_boundary_total = s_boundary.sum()

        return s_internal_total + s_boundary_total


    def loss(self, internal_beta, tree_tensor):
        tree_tensor = OctreeTensorHandler.set_s_vector(tree_tensor)

        s_total = self._calculate_s_total(internal_beta, tree_tensor)

        I = np.array([[s_total['s_yy'] + s_total['s_zz'], -s_total['s_xy'], -s_total['s_xz']],
                      [-s_total['s_xy'], s_total['s_xx'] + s_total['s_zz'], -s_total['s_yz']],
                      [-s_total['s_xz'], -s_total['s_yz'], s_total['s_xx'] + s_total['s_yy']]])
        I_CoM = I[:2,:2] - (s_total['s_z']**2 / s_total['s_1']) * np.eye(2)

        I_CoM_Rot = self._R * I_CoM * self._R.T

        I_a = I_CoM_Rot[0, 0]
        I_b = I_CoM_Rot[1, 1]
        I_c = s_total['s_xx'] + s_total['s_yy']


        f_yoyo = self._gamma_i * ((I_a / I_c)**2 + (I_b / I_c)**2)
        f_top = self._gamma_c * (s_total['s_z']**2) + f_yoyo

        # print(f'iter={self.iter}')
        # self.iter+=1
        # print(f'min_beta={internal_beta.min()}, max_beta={internal_beta.max()}')
        # print(f'Ia/Ic={I_a/I_c}, Ib/Ic={I_b/I_c}')
        # print(f'f_top={f_top}')

        self.loss_list.append(f_top)
        return f_top


        # TODO - choose between top and yoyo
        # f_to_min = f_top + self._gamma_l*0.5*

        # I_a = octree.get_I_a()
        # I_b = octree.get_I_b()
        # I_c = octree.get_I_c()
        # M = octree.get_mass()
        # l = octree.calc_center_of_mass() # TODO: height only

        # f_yoyo = self._gamma_i * ((I_a / I_c)**2 + (I_b / I_c)**2)
        # if self._calc_type == "top":
        #     f = self._gamma_c * (l * M)**2 + f_yoyo
        # elif self._calc_type == "yoyo":
        #     f = f_yoyo
            
        # return (f + self._gamma_l * 0.5 * beta.transpose() @ L @ beta)


    def _constraint_s_x(self, beta, tree_tensor):
        return self._calculate_s_total(beta, tree_tensor)['s_x']
    
    def _constraint_s_y(self, beta, tree_tensor):
        return self._calculate_s_total(beta, tree_tensor)['s_y']
    
    def _constraint_s_xz(self, beta, tree_tensor):
        return self._calculate_s_total(beta, tree_tensor)['s_xz']
    
    def _constraint_s_yz(self, beta, tree_tensor):
        return self._calculate_s_total(beta, tree_tensor)['s_yz']
    
    def __call__(self, tree_tensor: torch.Tensor, roh: float):
        beta = OctreeTensorHandler.get_internal_beta(tree_tensor)
        tree_tensor = OctreeTensorHandler.set_s_vector(tree_tensor, roh)

        bounds = [(0, 1)] * len(beta)
        problem = [{'type': 'eq', 'fun': self._constraint_s_x, 'args':(tree_tensor,)},           
                    {'type': 'eq', 'fun': self._constraint_s_y, 'args':(tree_tensor,)},
                    {'type': 'eq', 'fun': self._constraint_s_xz, 'args':(tree_tensor,)},
                    {'type': 'eq', 'fun': self._constraint_s_yz, 'args':(tree_tensor,)}]

        result = minimize(self.loss, beta, bounds=bounds, constraints=problem, method='SLSQP', args=(tree_tensor,), options={'disp': True})
        print(result)
        tree_tensor = OctreeTensorHandler.set_internal_beta(tree_tensor, result.x)
