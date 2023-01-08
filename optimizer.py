from octree import Octree
from scipy.optimize import minimize
import numpy as np


class QPOptimizer:
    def __init__(self, calc_type: str, gamma_l: float, gamma_i: float, gamma_c: float = None) -> None:
        self._gamma_l = gamma_l
        self._gamma_i = gamma_i
        self._gamma_c = gamma_c
        self._calc_type = calc_type

        self._R = np.eye(2)

    def loss(self, beta, octree):
        octree.set_s_vector()

        internal_beta = octree.get_internal_beta()
        s_internal = octree.get_internal_s_vector()
        s_internal_total = (internal_beta*s_internal).sum()

        s_boundary = octree.get_boundary_s_vector()
        s_boundary_total = s_boundary.sum()

        s_total = s_internal_total + s_boundary_total

        I = np.array([[s_yy+s_zz, -s_xy, -s_xz],
                        [-s_xy, s_xx+s_zz, -s_yz],
                        [-s_xz, -s_yz, s_xx+s_yy]])
        I_CoM = I[:2,:2] - (s_z**2/s_1) * np.eye(2)

        I_CoM_Rot = R*I_CoM*R_t

        I_a = I_CoM_Rot[0, 0]
        I_b = I_CoM_Rot[1, 1]

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



    
    def __call__(self, octree: Octree):
        beta = octree.get_internal_beta()
        result = minimize(self.loss, beta, (octree,))

        next_beta = result.x
