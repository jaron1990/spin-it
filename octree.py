from trimesh import Trimesh
import torch
import numpy as np
import pandas as pd
from utils import is_vertex_in_bbox, Location, OctreeTensorMapping
from igl import fast_winding_number_for_meshes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d



class OctreeTensorHandler:
    @staticmethod
    def stack_base_data(lvl: int | torch.Tensor, cell_start: torch.Tensor, cell_end: torch.Tensor, loc: int | torch.Tensor
                    ) -> torch.Tensor:
        if type(lvl) == int:
            lvl = torch.tensor([[lvl]])
        if type(loc) == int:
            loc = torch.tensor([[loc]])
        if len(cell_start.shape) == 1:
            cell_start = cell_start[None]
        if len(cell_end.shape) == 1:
            cell_end = cell_end[None]
        return torch.hstack((lvl, cell_start, cell_end, loc))
    
    @staticmethod
    def get_bbox_start(tree_tensor: torch.Tensor) -> torch.Tensor:
        return tree_tensor[:, torch.LongTensor([OctreeTensorMapping.BBOX_X0, 
                                                OctreeTensorMapping.BBOX_Y0, 
                                                OctreeTensorMapping.BBOX_Z0])]
    
    @staticmethod
    def get_bbox_end(tree_tensor: torch.Tensor) -> torch.Tensor:
        return tree_tensor[:, torch.LongTensor([OctreeTensorMapping.BBOX_X1, 
                                                OctreeTensorMapping.BBOX_Y1, 
                                                OctreeTensorMapping.BBOX_Z1])]
    
    @staticmethod
    def get_bbox_center(tree_tensor: torch.Tensor) -> torch.Tensor:
        return (OctreeTensorHandler.get_bbox_end(tree_tensor) + OctreeTensorHandler.get_bbox_start(tree_tensor)) / 2
    
    @staticmethod
    def get_bbox_size(tree_tensor: torch.Tensor) -> torch.Tensor:
        return OctreeTensorHandler.get_bbox_end(tree_tensor) - OctreeTensorHandler.get_bbox_start(tree_tensor)
    
    @staticmethod
    def get_lvl(tree_tensor: torch.Tensor) -> torch.Tensor:
        return tree_tensor[:, torch.LongTensor([OctreeTensorMapping.LVL])]
    
    @staticmethod
    def get_loc(tree_tensor: torch.Tensor) -> torch.Tensor:
        return tree_tensor[:, torch.LongTensor([OctreeTensorMapping.LOC])]
    
    @staticmethod
    def set_loc(tree_tensor: torch.Tensor, row_mask: torch.Tensor, new_loc: torch.Tensor) -> torch.Tensor:
        new_loc = new_loc.to(tree_tensor)
        col_mask = torch.zeros_like(tree_tensor[0]).bool()
        col_mask[OctreeTensorMapping.LOC] = True
        tree_tensor[row_mask, col_mask] = new_loc
        return tree_tensor
    
    @staticmethod
    def get_boundary(tree_tensor: torch.Tensor) -> torch.Tensor:
        return tree_tensor[(OctreeTensorHandler.get_loc(tree_tensor) == Location.BOUNDARY).squeeze(-1)]
    
    @staticmethod
    def get_interior(tree_tensor: torch.Tensor) -> torch.Tensor:
        return tree_tensor[(OctreeTensorHandler.get_loc(tree_tensor) == Location.INSIDE).squeeze(-1)]

    @staticmethod
    def get_exterior(tree_tensor: torch.Tensor) -> torch.Tensor:
        return tree_tensor[(OctreeTensorHandler.get_loc(tree_tensor) == Location.OUTSIDE).squeeze(-1)]

    @staticmethod
    def get_internal_beta(tree_tensor: torch.Tensor) -> torch.Tensor:
        return OctreeTensorHandler.get_interior(tree_tensor)[:, OctreeTensorMapping.BETA]

    @staticmethod
    def set_internal_beta(tree_tensor, internal_beta) -> torch.Tensor:
        return OctreeTensorHandler.set_beta(OctreeTensorHandler.get_interior(tree_tensor), internal_beta)

    @staticmethod
    def get_s_vector(tree_tensor: torch.Tensor) -> torch.Tensor:
        cols = torch.tensor([OctreeTensorMapping.S_1,
                             OctreeTensorMapping.S_X, OctreeTensorMapping.S_Y, OctreeTensorMapping.S_Z,
                             OctreeTensorMapping.S_XY, OctreeTensorMapping.S_XZ, OctreeTensorMapping.S_YZ,
                             OctreeTensorMapping.S_XX, OctreeTensorMapping.S_YY, OctreeTensorMapping.S_ZZ])
        cols_mask = torch.zeros_like(tree_tensor[0], dtype=bool)
        cols_mask[cols] = True
        return tree_tensor[:, cols_mask]

    @staticmethod
    def get_internal_s_vector(tree_tensor: torch.Tensor) -> torch.Tensor:
        return OctreeTensorHandler.get_s_vector(OctreeTensorHandler.get_interior(tree_tensor))

    @staticmethod
    def get_boundary_s_vector(tree_tensor: torch.Tensor) -> torch.Tensor:
        return OctreeTensorHandler.get_s_vector(OctreeTensorHandler.get_boundary(tree_tensor))

    @staticmethod
    def calc_inner_outter_location(mesh_obj: Trimesh, tree_tensor: torch.Tensor) -> torch.Tensor:
        is_unknown = (OctreeTensorHandler.get_loc(tree_tensor) == Location.UNKNOWN).squeeze(-1)
        centers = OctreeTensorHandler.get_bbox_center(tree_tensor[is_unknown])
        is_inner = fast_winding_number_for_meshes(np.array(mesh_obj.vertices), 
                                                  np.array(mesh_obj.faces), 
                                                  centers.numpy()) > 0.5
        new_loc = torch.where(torch.tensor(is_inner), torch.tensor([Location.INSIDE]), torch.tensor([Location.OUTSIDE]))
        return OctreeTensorHandler.set_loc(tree_tensor, is_unknown, new_loc)

    @staticmethod
    def set_beta(tree_tensor: torch.Tensor, beta_vals: None | torch.Tensor = None) -> torch.Tensor:
        is_outside = OctreeTensorHandler.get_loc(tree_tensor) == Location.OUTSIDE
        if beta_vals is None:
            beta_vals = torch.where(is_outside, torch.tensor([0.]), torch.tensor([1.]))
            assert tree_tensor[0].shape[0] == OctreeTensorMapping.BETA
            return torch.cat((tree_tensor, beta_vals), axis=-1)
        tree_tensor[:, OctreeTensorMapping.BETA] = torch.tensor(beta_vals)
        return tree_tensor

    @staticmethod
    def calc_s_vector(tree_tensor, roh):
        p0 = OctreeTensorHandler.get_bbox_start(tree_tensor)
        p1 = OctreeTensorHandler.get_bbox_end(tree_tensor)
        size = OctreeTensorHandler.get_bbox_size(tree_tensor)
        size_x = size[:, 0]
        size_y = size[:, 1]
        size_z = size[:, 2]
        
        integral = lambda x1, x0: (x1**2 -x0**2) / 2
        integral_x = integral(p1[:, 0], p0[:, 0])
        integral_y = integral(p1[:, 1], p0[:, 1])
        integral_z = integral(p1[:, 2], p0[:, 2])
        
        integral = lambda x1, x0: (x1**3 -x0**3) / 3
        integral_xx = integral(p1[:, 0], p0[:, 0])
        integral_yy = integral(p1[:, 1], p0[:, 1])
        integral_zz = integral(p1[:, 2], p0[:, 2])
        
        s_1 = roh * size_x * size_y * size_z
        s_x = roh * size_y * size_z * integral_x
        s_y = roh * size_x * size_z * integral_y
        s_z = roh * size_x * size_y * integral_z
        s_xy = roh * size_z * integral_x * integral_y
        s_xz = roh * size_y * integral_x * integral_z
        s_yz = roh * size_x * integral_y * integral_z
        s_xx = roh * size_y * size_z * integral_xx
        s_yy = roh * size_x * size_z * integral_yy
        s_zz = roh * size_x * size_y * integral_zz
        
        return torch.cat((tree_tensor[:, :OctreeTensorMapping.S_1],
                          torch.stack((s_1, 
                                       s_x, s_y, s_z, 
                                       s_xy, s_xz, s_yz, 
                                       s_xx, s_yy, s_zz), axis=-1)), dim=-1)


class Octree:
    def __init__(self, init_resolution: int, max_resolution_levels: int) -> None:
        self._init_res = init_resolution
        self._max_level = max_resolution_levels
    
    def _build_init_res(self, vertices: torch.Tensor):
        obj_start = vertices.min(axis=0).values
        obj_end = vertices.max(axis=0).values
        lvl = -1
        loc = Location.UNKNOWN
        return self._create_leaves_tensor(self._init_res, vertices, 
                                          OctreeTensorHandler.stack_base_data(lvl, obj_start, obj_end, loc))
    
    def _create_leaves_tensor(self, split_size: int, vertices: torch.Tensor, tree_tensor: torch.Tensor) -> torch.Tensor:
        cell_size = OctreeTensorHandler.get_bbox_size(tree_tensor) / split_size
        
        split_range = torch.arange(split_size)
        idxs = torch.cartesian_prod(*([split_range] * 3))
        lvl = (OctreeTensorHandler.get_lvl(tree_tensor) + 1).repeat(idxs.shape[0], 1)
        bbox_start = OctreeTensorHandler.get_bbox_start(tree_tensor).repeat(idxs.shape[0], 1)
        cell_start = bbox_start + (cell_size[None] * idxs.unsqueeze(1)).reshape(-1, 3)
        cell_end = bbox_start + (cell_size[None] * (idxs + 1).unsqueeze(1)).reshape(-1 ,3)
        
        is_in_bbox = is_vertex_in_bbox(vertices, cell_start, cell_end)
        loc = torch.where(is_in_bbox, torch.tensor([Location.BOUNDARY]), torch.tensor([Location.UNKNOWN])).unsqueeze(1)
        return OctreeTensorHandler.stack_base_data(lvl, cell_start, cell_end, loc)
    
    def build_from_mesh(self, mesh_obj: Trimesh):
        mesh_obj.vertices
        vertices = torch.tensor(mesh_obj.vertices)
        vertices_max_abs = vertices.absolute().max()
        mesh_obj.vertices /= vertices_max_abs
        vertices = torch.tensor(mesh_obj.vertices)
        
        tree_tensor = self._build_init_res(vertices)
        for _ in range(1, self._max_level):
            is_bound = OctreeTensorHandler.get_loc(tree_tensor).squeeze(-1) == Location.BOUNDARY
            tree_tensor = torch.vstack((tree_tensor[~is_bound], 
                                        self._create_leaves_tensor(2, vertices, tree_tensor[is_bound])))
        tree_tensor = OctreeTensorHandler.calc_inner_outter_location(mesh_obj, tree_tensor)
        tree_tensor = OctreeTensorHandler.set_beta(tree_tensor)

        # self._plot(tree_tensor)
        return tree_tensor

    def _plot(self, tree_tensor):
        inside_tensor = OctreeTensorHandler.get_interior(tree_tensor)
        boundary_tensor = OctreeTensorHandler.get_boundary(tree_tensor)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for ind in range(inside_tensor.shape[0]):
            x0, y0, z0 = [v.item() for v in OctreeTensorHandler.get_bbox_start(inside_tensor)[ind].split(1)]
            x1, y1, z1 = [v.item() for v in OctreeTensorHandler.get_bbox_end(inside_tensor)[ind].split(1)]
            
            x, y = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
            z = np.ones(x.shape) * z0
            ax.plot_surface(x, y, z, color='b', alpha=0.6)

            x, y = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
            z = np.ones(x.shape) * z1
            ax.plot_surface(x, y, z, color='b', alpha=0.6)

            x, z = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(z0, z1, 2))
            y = np.ones(x.shape) * y0
            ax.plot_surface(x, y, z, color='b', alpha=0.6)

            x, z = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(z0, z1, 2))
            y = np.ones(x.shape) * y1
            ax.plot_surface(x, y, z, color='b', alpha=0.6)

            y, z = np.meshgrid(np.linspace(y0, y1, 2), np.linspace(z0, z1, 2))
            x = np.ones(y.shape) * x0
            ax.plot_surface(x, y, z, color='b', alpha=0.6)

            y, z = np.meshgrid(np.linspace(y0, y1, 2), np.linspace(z0, z1, 2))
            x = np.ones(y.shape) * x1
            ax.plot_surface(x, y, z, color='b')
        
        for ind in range(boundary_tensor.shape[0]):
            x0, y0, z0 = [v.item() for v in OctreeTensorHandler.get_bbox_start(boundary_tensor)[ind].split(1)]
            x1, y1, z1 = [v.item() for v in OctreeTensorHandler.get_bbox_end(boundary_tensor)[ind].split(1)]
            
            x, y = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
            z = np.ones(x.shape) * z0
            ax.plot_surface(x, y, z, color='r', alpha=0.3)

            x, y = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
            z = np.ones(x.shape) * z1
            ax.plot_surface(x, y, z, color='r', alpha=0.3)

            x, z = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(z0, z1, 2))
            y = np.ones(x.shape) * y0
            ax.plot_surface(x, y, z, color='r', alpha=0.3)

            x, z = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(z0, z1, 2))
            y = np.ones(x.shape) * y1
            ax.plot_surface(x, y, z, color='r', alpha=0.3)

            y, z = np.meshgrid(np.linspace(y0, y1, 2), np.linspace(z0, z1, 2))
            x = np.ones(y.shape) * x0
            ax.plot_surface(x, y, z, color='r', alpha=0.3)

            y, z = np.meshgrid(np.linspace(y0, y1, 2), np.linspace(z0, z1, 2))
            x = np.ones(y.shape) * x1
            ax.plot_surface(x, y, z, color='r')
        plt.savefig("boxes.png")
        