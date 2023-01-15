from trimesh import Trimesh
import torch
import numpy as np
import pandas as pd
from utils import is_vertex_in_bbox, Location, OctreeTensorMapping
from igl import fast_winding_number_for_meshes
import matplotlib.pyplot as plt
from mesh_obj import MeshObj
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d



class OctreeTensorHandler:
    @staticmethod
    def stack_base_data(lvl: int | torch.Tensor, cell_start: torch.Tensor, cell_end: torch.Tensor, loc: int | torch.Tensor,
                        beta: torch.Tensor | None = None, s_vector: torch.Tensor | None = None) -> torch.Tensor:
        if type(lvl) == int:
            lvl = torch.tensor([[lvl]])
        if type(loc) == int:
            loc = torch.tensor([[loc]])
        if len(cell_start.shape) == 1:
            cell_start = cell_start[None]
        if len(cell_end.shape) == 1:
            cell_end = cell_end[None]
        if beta is None:
            beta = torch.ones_like(lvl) * 0.5
        if s_vector is None:
            s_vector = torch.zeros_like(lvl).repeat(1, OctreeTensorMapping.S_ZZ - OctreeTensorMapping.BETA)
        return torch.hstack((lvl, cell_start, cell_end, loc, beta, s_vector))
    
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
        return tree_tensor[:, OctreeTensorMapping.LOC]
    
    @staticmethod
    def set_loc(tree_tensor: torch.Tensor, row_mask: torch.Tensor, new_loc: torch.Tensor) -> torch.Tensor:
        new_loc = new_loc.to(tree_tensor)
        col_mask = torch.zeros_like(tree_tensor[0]).bool()
        col_mask[OctreeTensorMapping.LOC] = True
        tree_tensor[row_mask, col_mask] = new_loc
        return tree_tensor
    
    @staticmethod
    def get_boundary(tree_tensor: torch.Tensor) -> torch.Tensor:
        return tree_tensor[(OctreeTensorHandler.get_loc(tree_tensor) == Location.BOUNDARY)]
    
    @staticmethod
    def get_interior(tree_tensor: torch.Tensor) -> torch.Tensor:
        return tree_tensor[(OctreeTensorHandler.get_loc(tree_tensor) == Location.INSIDE)]

    @staticmethod
    def get_exterior(tree_tensor: torch.Tensor) -> torch.Tensor:
        return tree_tensor[(OctreeTensorHandler.get_loc(tree_tensor) == Location.OUTSIDE)]

    @staticmethod
    def get_internal_beta(tree_tensor: torch.Tensor) -> torch.Tensor:
        return OctreeTensorHandler.get_interior(tree_tensor)[:, OctreeTensorMapping.BETA]

    @staticmethod
    def set_internal_beta(tree_tensor: torch.Tensor, internal_beta: torch.Tensor) -> torch.Tensor:
        is_inside = OctreeTensorHandler.get_loc(tree_tensor) == Location.INSIDE
        tree_tensor[is_inside, OctreeTensorMapping.BETA] = internal_beta
        return tree_tensor

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
    def calc_inner_outter_location(mesh_obj: MeshObj, tree_tensor: torch.Tensor) -> torch.Tensor:
        is_unknown = OctreeTensorHandler.get_loc(tree_tensor) == Location.UNKNOWN
        centers = OctreeTensorHandler.get_bbox_center(tree_tensor[is_unknown])
        is_inner = fast_winding_number_for_meshes(mesh_obj.vertices, mesh_obj.faces, centers.numpy()) > 0.5
        new_loc = torch.where(torch.tensor(is_inner), torch.tensor([Location.INSIDE]), torch.tensor([Location.OUTSIDE]))
        return OctreeTensorHandler.set_loc(tree_tensor, is_unknown, new_loc)

    @staticmethod
    def set_beta(tree_tensor: torch.Tensor, beta_vals: None | torch.Tensor = None) -> torch.Tensor:
        if beta_vals is None:
            is_outside = OctreeTensorHandler.get_loc(tree_tensor) == Location.OUTSIDE
            curr_beta = tree_tensor[:, OctreeTensorMapping.BETA]
            beta_vals = torch.where(is_outside, torch.tensor([0.]), curr_beta)
            
            is_boundary = OctreeTensorHandler.get_loc(tree_tensor) == Location.BOUNDARY
            beta_vals = torch.where(is_boundary, torch.tensor([1.]), beta_vals)
        tree_tensor[:, OctreeTensorMapping.BETA] = beta_vals
        return tree_tensor

    # @staticmethod
    # def create_voxel_grid(tree_tensor: torch.Tensor):
    #     inside_tensor = OctreeTensorHandler.get_interior(tree_tensor)
    #     inside_tensor = inside_tensor.where(inside_tensor[:,OctreeTensorMapping.BETA]>0.5)
    #     boundary_tensor = OctreeTensorHandler.get_boundary(tree_tensor)
    #     # OctreeTensorHandler.

    #     # return grid
    
    @staticmethod
    def plot_slices(tree_tensor: torch.Tensor):
        OctreeTensorHandler.plot_2D_x(tree_tensor, 0)
        OctreeTensorHandler.plot_2D_y(tree_tensor, 0)
        OctreeTensorHandler.plot_2D_z(tree_tensor, 0)

    @staticmethod
    def plot_2D_x(self, tree_tensor, x_val):
        inside_tensor = OctreeTensorHandler.get_interior(tree_tensor)
        boundary_tensor = OctreeTensorHandler.get_boundary(tree_tensor)

        min_y = min(inside_tensor[:,OctreeTensorMapping.BBOX_Y0].min(), boundary_tensor[:,OctreeTensorMapping.BBOX_Y0].min())
        max_y = max(inside_tensor[:,OctreeTensorMapping.BBOX_Y1].max(), boundary_tensor[:,OctreeTensorMapping.BBOX_Y1].max())
        min_z = min(inside_tensor[:,OctreeTensorMapping.BBOX_Z0].min(), boundary_tensor[:,OctreeTensorMapping.BBOX_Z0].min())
        max_z = max(inside_tensor[:,OctreeTensorMapping.BBOX_Z1].max(), boundary_tensor[:,OctreeTensorMapping.BBOX_Z1].max())

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.xlim([min_y, max_y])
        plt.ylim([min_z, max_z])
        plt.xlabel('y')
        plt.ylabel('z')

        inside_x_start = inside_tensor[:,OctreeTensorMapping.BBOX_X0]
        inside_x_end = inside_tensor[:,OctreeTensorMapping.BBOX_X1]
        inside_cuts_x_val = (inside_x_start<=x_val) & (inside_x_end>x_val)
        inside_tensor_cuts_x_val = inside_tensor[inside_cuts_x_val]
        for inside_cell in inside_tensor_cuts_x_val:
            ax.add_patch(Rectangle((inside_cell[OctreeTensorMapping.BBOX_Y0], inside_cell[OctreeTensorMapping.BBOX_Z0]),    #start_point
                                    inside_cell[OctreeTensorMapping.BBOX_Y1]-inside_cell[OctreeTensorMapping.BBOX_Y0],      #size_y
                                    inside_cell[OctreeTensorMapping.BBOX_Z1]-inside_cell[OctreeTensorMapping.BBOX_Z0],      #size_z
                                    edgecolor = 'pink',
                                    facecolor = 'blue',
                                    fill=True,
                                    lw=1,
                                    alpha=inside_cell[OctreeTensorMapping.BETA]))
            # plt.savefig(f"axial_x_{x_val}.png")

            
        boundary_x_start = boundary_tensor[:,OctreeTensorMapping.BBOX_X0]
        boundary_x_end = boundary_tensor[:,OctreeTensorMapping.BBOX_X1]
        boundary_cuts_x_val = (boundary_x_start<=x_val) & (boundary_x_end>x_val)
        boundary_tensor_cuts_x_val = boundary_tensor[boundary_cuts_x_val]
        for boundary_cell in boundary_tensor_cuts_x_val:
            ax.add_patch(Rectangle((boundary_cell[OctreeTensorMapping.BBOX_Y0], boundary_cell[OctreeTensorMapping.BBOX_Z0]),    #start_point
                                    boundary_cell[OctreeTensorMapping.BBOX_Y1]-boundary_cell[OctreeTensorMapping.BBOX_Y0],      #size_y
                                    boundary_cell[OctreeTensorMapping.BBOX_Z1]-boundary_cell[OctreeTensorMapping.BBOX_Z0],      #size_z
                                    edgecolor = 'black',
                                    facecolor = 'red',
                                    fill=True,
                                    lw=1))
            # plt.savefig(f"axial_x_{x_val}.png")

        plt.savefig(f"axial_x_{x_val}.png")

    def plot_2D_y(self, tree_tensor, y_val):
        inside_tensor = OctreeTensorHandler.get_interior(tree_tensor)
        boundary_tensor = OctreeTensorHandler.get_boundary(tree_tensor)

        min_x = min(inside_tensor[:,OctreeTensorMapping.BBOX_X0].min(), boundary_tensor[:,OctreeTensorMapping.BBOX_X0].min())
        max_x = max(inside_tensor[:,OctreeTensorMapping.BBOX_X1].max(), boundary_tensor[:,OctreeTensorMapping.BBOX_X1].max())
        min_z = min(inside_tensor[:,OctreeTensorMapping.BBOX_Z0].min(), boundary_tensor[:,OctreeTensorMapping.BBOX_Z0].min())
        max_z = max(inside_tensor[:,OctreeTensorMapping.BBOX_Z1].max(), boundary_tensor[:,OctreeTensorMapping.BBOX_Z1].max())

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.xlim([min_x, max_x])
        plt.ylim([min_z, max_z])
        plt.xlabel('x')
        plt.ylabel('z')

        inside_y_start = inside_tensor[:,OctreeTensorMapping.BBOX_Y0]
        inside_y_end = inside_tensor[:,OctreeTensorMapping.BBOX_Y1]
        inside_cuts_y_val = (inside_y_start<=y_val) & (inside_y_end>y_val)
        inside_tensor_cuts_y_val = inside_tensor[inside_cuts_y_val]
        for inside_cell in inside_tensor_cuts_y_val:
            ax.add_patch(Rectangle((inside_cell[OctreeTensorMapping.BBOX_X0], inside_cell[OctreeTensorMapping.BBOX_Z0]),    #start_point
                                    inside_cell[OctreeTensorMapping.BBOX_X1]-inside_cell[OctreeTensorMapping.BBOX_X0],      #size_x
                                    inside_cell[OctreeTensorMapping.BBOX_Z1]-inside_cell[OctreeTensorMapping.BBOX_Z0],      #size_z
                                    edgecolor = 'pink',
                                    facecolor = 'blue',
                                    fill=True,
                                    lw=1,
                                    alpha=inside_cell[OctreeTensorMapping.BETA]))
            # plt.savefig(f"axial_y_{y_val}.png")

            
        boundary_y_start = boundary_tensor[:,OctreeTensorMapping.BBOX_Y0]
        boundary_y_end = boundary_tensor[:,OctreeTensorMapping.BBOX_Y1]
        boundary_cuts_y_val = (boundary_y_start<=y_val) & (boundary_y_end>y_val)
        boundary_tensor_cuts_y_val = boundary_tensor[boundary_cuts_y_val]
        for boundary_cell in boundary_tensor_cuts_y_val:
            ax.add_patch(Rectangle((boundary_cell[OctreeTensorMapping.BBOX_X0], boundary_cell[OctreeTensorMapping.BBOX_Z0]),    #start_point
                                    boundary_cell[OctreeTensorMapping.BBOX_X1]-boundary_cell[OctreeTensorMapping.BBOX_X0],      #size_x
                                    boundary_cell[OctreeTensorMapping.BBOX_Z1]-boundary_cell[OctreeTensorMapping.BBOX_Z0],      #size_z
                                    edgecolor = 'black',
                                    facecolor = 'red',
                                    fill=True,
                                    lw=1))
            # plt.savefig(f"axial_y_{y_val}.png")

        plt.savefig(f"axial_y_{y_val}.png")

    def plot_2D_z(self, tree_tensor, z_val):
        inside_tensor = OctreeTensorHandler.get_interior(tree_tensor)
        boundary_tensor = OctreeTensorHandler.get_boundary(tree_tensor)

        min_y = min(inside_tensor[:,OctreeTensorMapping.BBOX_Y0].min(), boundary_tensor[:,OctreeTensorMapping.BBOX_Y0].min())
        max_y = max(inside_tensor[:,OctreeTensorMapping.BBOX_Y1].max(), boundary_tensor[:,OctreeTensorMapping.BBOX_Y1].max())
        min_x = min(inside_tensor[:,OctreeTensorMapping.BBOX_X0].min(), boundary_tensor[:,OctreeTensorMapping.BBOX_X0].min())
        max_x = max(inside_tensor[:,OctreeTensorMapping.BBOX_X1].max(), boundary_tensor[:,OctreeTensorMapping.BBOX_X1].max())

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])
        plt.xlabel('x')
        plt.ylabel('y')

        inside_z_start = inside_tensor[:,OctreeTensorMapping.BBOX_Z0]
        inside_z_end = inside_tensor[:,OctreeTensorMapping.BBOX_Z1]
        inside_cuts_z_val = (inside_z_start<=z_val) & (inside_z_end>z_val)
        inside_tensor_cuts_z_val = inside_tensor[inside_cuts_z_val]
        for inside_cell in inside_tensor_cuts_z_val:
            ax.add_patch(Rectangle((inside_cell[OctreeTensorMapping.BBOX_X0], inside_cell[OctreeTensorMapping.BBOX_Y0]),    #start_point
                                    inside_cell[OctreeTensorMapping.BBOX_X1]-inside_cell[OctreeTensorMapping.BBOX_X0],      #size_x
                                    inside_cell[OctreeTensorMapping.BBOX_Y1]-inside_cell[OctreeTensorMapping.BBOX_Y0],      #size_y
                                    edgecolor = 'pink',
                                    facecolor = 'blue',
                                    fill=True,
                                    lw=1,
                                    alpha=inside_cell[OctreeTensorMapping.BETA]))
            # plt.savefig(f"axial_z_{z_val}.png")

            
        boundary_z_start = boundary_tensor[:,OctreeTensorMapping.BBOX_Z0]
        boundary_z_end = boundary_tensor[:,OctreeTensorMapping.BBOX_Z1]
        boundary_cuts_z_val = (boundary_z_start<=z_val) & (boundary_z_end>z_val)
        boundary_tensor_cuts_z_val = boundary_tensor[boundary_cuts_z_val]
        for boundary_cell in boundary_tensor_cuts_z_val:
            ax.add_patch(Rectangle((boundary_cell[OctreeTensorMapping.BBOX_X0], boundary_cell[OctreeTensorMapping.BBOX_Y0]),    #start_point
                                    boundary_cell[OctreeTensorMapping.BBOX_X1]-boundary_cell[OctreeTensorMapping.BBOX_X0],      #size_x
                                    boundary_cell[OctreeTensorMapping.BBOX_Y1]-boundary_cell[OctreeTensorMapping.BBOX_Y0],      #size_y
                                    edgecolor = 'black',
                                    facecolor = 'red',
                                    fill=True,
                                    lw=1))
            # plt.savefig(f"axial_z_{z_val}.png")

        plt.savefig(f"axial_z_{z_val}.png")



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
        return self.create_leaves_tensor(self._init_res, vertices, 
                                         OctreeTensorHandler.stack_base_data(lvl, obj_start, obj_end, loc))
    
    @staticmethod
    def create_leaves_tensor(split_size: int, vertices: torch.Tensor | None, tree_tensor: torch.Tensor) -> torch.Tensor:
        cell_size = OctreeTensorHandler.get_bbox_size(tree_tensor) / split_size
        
        split_range = torch.arange(split_size)
        idxs = torch.cartesian_prod(*([split_range] * 3))
        lvl = (OctreeTensorHandler.get_lvl(tree_tensor) + 1).repeat(idxs.shape[0], 1)
        bbox_start = OctreeTensorHandler.get_bbox_start(tree_tensor).repeat(idxs.shape[0], 1)
        cell_start = bbox_start + (cell_size[None] * idxs.unsqueeze(1)).reshape(-1, 3)
        cell_end = bbox_start + (cell_size[None] * (idxs + 1).unsqueeze(1)).reshape(-1 ,3)
        
        if vertices is not None:
            is_in_bbox = is_vertex_in_bbox(vertices, cell_start, cell_end)
            loc = torch.where(is_in_bbox, torch.tensor([Location.BOUNDARY]), torch.tensor([Location.UNKNOWN])).unsqueeze(1)
        else:
            loc = torch.ones_like(cell_start[:, :1]) * Location.INSIDE
        return OctreeTensorHandler.stack_base_data(lvl, cell_start, cell_end, loc)
    
    def build_from_mesh(self, mesh_obj: MeshObj):
        vertices = torch.tensor(mesh_obj.vertices)
        tree_tensor = self._build_init_res(vertices)
        for _ in range(1, self._max_level):
            is_bound = OctreeTensorHandler.get_loc(tree_tensor) == Location.BOUNDARY
            tree_tensor = torch.vstack((tree_tensor[~is_bound], 
                                        self.create_leaves_tensor(2, vertices, tree_tensor[is_bound])))
        tree_tensor = OctreeTensorHandler.calc_inner_outter_location(mesh_obj, tree_tensor)
        # self._plot(tree_tensor)
        # self._plot_2D_x(tree_tensor, 0)
        # self._plot_2D_y(tree_tensor, 0)
        # self._plot_2D_z(tree_tensor, 0)
        # voxel_grid = OctreeTensorHandler.create_voxel_grid(tree_tensor)
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
