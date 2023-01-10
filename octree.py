from trimesh import Trimesh
import torch
import numpy as np
import pandas as pd
from utils import is_vertex_in_bbox, Location #, is_bbox_inside_mesh
from igl import fast_winding_number_for_meshes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d


class OctreeTensorMapping:
    LVL = 0
    BBOX_X0 = 1
    BBOX_Y0 = 2
    BBOX_Z0 = 3
    BBOX_X1 = 4
    BBOX_Y1 = 5
    BBOX_Z1 = 6
    LOC = 7
    BETA = 8
    S_1 = 9
    S_X = 10
    S_Y = 11
    S_Z = 12
    S_XY = 13
    S_XZ = 14
    S_YZ = 15
    S_XX = 16
    S_YY = 17
    S_ZZ = 18


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
        return tree_tensor[OctreeTensorHandler.get_loc(tree_tensor) == Location.BOUNDARY]
    
    @staticmethod
    def get_interior(tree_tensor: torch.Tensor) -> torch.Tensor:
        return tree_tensor[OctreeTensorHandler.get_loc(tree_tensor) == Location.INSIDE]
    
    @staticmethod
    def get_internal_beta(tree_tensor: torch.Tensor) -> torch.Tensor:
        return OctreeTensorHandler.get_interior(tree_tensor)[:, OctreeTensorMapping.BETA]

    @staticmethod
    def set_internal_beta(internal_beta): # TODO: fix
        self._tree_tensor.loc[(self._tree_tensor['loc'] == Location.INSIDE), 'beta'] = internal_beta

    @staticmethod
    def get_s_vector(tree_tensor: torch.Tensor) -> torch.Tensor:
        cols = torch.tensor([OctreeTensorMapping.S_1,
                             OctreeTensorMapping.S_X, OctreeTensorMapping.S_Y, OctreeTensorMapping.S_Z,
                             OctreeTensorMapping.S_XY, OctreeTensorMapping.S_XZ, OctreeTensorMapping.S_YZ,
                             OctreeTensorMapping.S_XX, OctreeTensorMapping.S_YY, OctreeTensorMapping.S_ZZ])
        cols_mask = torch.zeros_like(tree_tensor[0])
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
    def set_beta(tree_tensor: torch.Tensor) -> torch.Tensor:
        is_outside = OctreeTensorHandler.get_loc(tree_tensor) == Location.OUTSIDE
        beta_vals = torch.where(is_outside, torch.tensor([0.]), torch.tensor([0.]))
        assert tree_tensor[0].shape[0] == OctreeTensorMapping.BETA
        return torch.cat((tree_tensor, beta_vals), axis=-1)


class Octree:
    def __init__(self, init_resolution: int, max_resolution_levels: int, roh: float) -> None:
        self._init_res = init_resolution
        self._max_level = max_resolution_levels
        self._roh = roh
    
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
        vertices = torch.tensor(mesh_obj.vertices)
        tree_tensor = self._build_init_res(vertices)
        for _ in range(1, self._max_level):
            is_bound = OctreeTensorHandler.get_loc(tree_tensor).squeeze(-1) == Location.BOUNDARY
            tree_tensor = torch.vstack((tree_tensor[~is_bound], 
                                        self._create_leaves_tensor(2, vertices, tree_tensor[is_bound])))
        tree_tensor = OctreeTensorHandler.calc_inner_outter_location(mesh_obj, tree_tensor)
        tree_tensor = OctreeTensorHandler.set_beta(tree_tensor)

        self._plot()
        return tree_tensor

    def _plot(self):
        inside_df = self.get_interior()
        boundary_df = self.get_boundary()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for ind in inside_df.index:
            (x0, y0, z0, x1, y1, z1) = (inside_df['bbox_x0'][ind], inside_df['bbox_y0'][ind], inside_df['bbox_z0'][ind], inside_df['bbox_x1'][ind], inside_df['bbox_y1'][ind], inside_df['bbox_z1'][ind])
            
            
            x, y = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
            z = np.ones(x.shape) * z0
            ax.plot_surface(x, y, z, color='b')

            x, y = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
            z = np.ones(x.shape) * z1
            ax.plot_surface(x, y, z, color='b')

            x, z = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(z0, z1, 2))
            y = np.ones(x.shape) * y0
            ax.plot_surface(x, y, z, color='b')

            x, z = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(z0, z1, 2))
            y = np.ones(x.shape) * y1
            ax.plot_surface(x, y, z, color='b')

            y, z = np.meshgrid(np.linspace(y0, y1, 2), np.linspace(z0, z1, 2))
            x = np.ones(y.shape) * x0
            ax.plot_surface(x, y, z, color='b')

            y, z = np.meshgrid(np.linspace(y0, y1, 2), np.linspace(z0, z1, 2))
            x = np.ones(y.shape) * x1
            ax.plot_surface(x, y, z, color='b')
        for ind in boundary_df.index:
            (x0, y0, z0, x1, y1, z1) = (boundary_df['bbox_x0'][ind], boundary_df['bbox_y0'][ind], boundary_df['bbox_z0'][ind], boundary_df['bbox_x1'][ind], boundary_df['bbox_y1'][ind], boundary_df['bbox_z1'][ind])
            
            
            x, y = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
            z = np.ones(x.shape) * z0
            ax.plot_surface(x, y, z, color='r')

            x, y = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
            z = np.ones(x.shape) * z1
            ax.plot_surface(x, y, z, color='r')

            x, z = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(z0, z1, 2))
            y = np.ones(x.shape) * y0
            ax.plot_surface(x, y, z, color='r')

            x, z = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(z0, z1, 2))
            y = np.ones(x.shape) * y1
            ax.plot_surface(x, y, z, color='r')

            y, z = np.meshgrid(np.linspace(y0, y1, 2), np.linspace(z0, z1, 2))
            x = np.ones(y.shape) * x0
            ax.plot_surface(x, y, z, color='r')

            y, z = np.meshgrid(np.linspace(y0, y1, 2), np.linspace(z0, z1, 2))
            x = np.ones(y.shape) * x1
            ax.plot_surface(x, y, z, color='r')

        

        plt.savefig("boxes.png")

    def set_s_vector(self):
        p0 = self._get_bbox_start(self._tree_tensor).to_numpy()
        p1 = self._get_bbox_end(self._tree_tensor).to_numpy()
        size = self._get_bbox_size(self._tree_tensor)
        size_x = size[:, 0]
        size_y = size[:, 1]
        size_z = size[:, 2]
        integral_x = (p1[:, 0]**2 - p0[:, 0]**2) / 2
        integral_y = (p1[:, 1]**2 - p0[:, 1]**2) / 2
        integral_z = (p1[:, 2]**2 - p0[:, 2]**2) / 2
        integral_xx = (p1[:, 0]**3 - p0[:, 0]**3) / 3
        integral_yy = (p1[:, 1]**3 - p0[:, 1]**3) / 3
        integral_zz = (p1[:, 2]**3 - p0[:, 2]**3) / 3
        s_1 = self._roh * size_x * size_y * size_z
        s_x = self._roh * size_y * size_z * integral_x
        s_y = self._roh * size_x * size_z * integral_y
        s_z = self._roh * size_x * size_y * integral_z
        s_xy = self._roh * size_z * integral_x * integral_y
        s_xz = self._roh * size_y * integral_x * integral_z
        s_yz = self._roh * size_x * integral_y * integral_z
        s_xx = self._roh * size_y * size_z * integral_xx
        s_yy = self._roh * size_x * size_z * integral_yy
        s_zz = self._roh * size_x * size_y * integral_zz

        if 's_1' in self._tree_tensor.columns:
            self._tree_tensor[['s_1', 's_x', 's_y', 's_z', 's_xy', 's_xz', 's_yz', 's_xx', 's_yy', 's_zz']] = np.stack((s_1, s_x, s_y, s_z, s_xy, s_xz, s_yz, s_xx, s_yy, s_zz), axis=-1)
        else:
            s_vector = pd.DataFrame(np.stack((s_1, s_x, s_y, s_z, s_xy, s_xz, s_yz, s_xx, s_yy, s_zz), 
                                    axis=-1), 
                                    columns=['s_1', 's_x', 's_y', 's_z', 's_xy', 's_xz', 's_yz', 's_xx', 's_yy', 's_zz'])
            self._tree_tensor = pd.concat([self._tree_tensor, s_vector], axis=1)
        