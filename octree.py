from trimesh import Trimesh
import numpy as np
import itertools
import pandas as pd
from utils import is_vertex_in_bbox, Location #, is_bbox_inside_mesh
from igl import fast_winding_number_for_meshes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d

class Octree:
    def __init__(self, init_resolution: int, max_resolution_levels: int, roh: float) -> None:
        self._init_res = init_resolution
        self._max_level = max_resolution_levels
        self._tree_df = None
        self._roh = roh
    
    @staticmethod
    def _create_df(lvl, cell_start, cell_end, loc=Location.UNKNOWN):
        return pd.DataFrame.from_dict([{"level": lvl, 
                                        "bbox_x0": cell_start[0], "bbox_y0": cell_start[1], "bbox_z0": cell_start[2], 
                                        "bbox_x1": cell_end[0], "bbox_y1": cell_end[1], "bbox_z1": cell_end[2], 
                                        "loc": loc}])
    
    @staticmethod
    def _concat_df(lvl, cell_start, cell_end):
        if "bbox_x0" in cell_end.keys():
            cell_end.rename(columns={"bbox_x0": "bbox_x1", "bbox_y0": "bbox_y1", "bbox_z0": "bbox_z1"}, inplace=True)
        return pd.concat([lvl, cell_start, cell_end], axis=1)
    
    def _build_init_res(self, vertices: np.array):
        obj_start = vertices.min(axis=0)
        obj_end = vertices.max(axis=0)
        lvl = -1
        father_idx = None
        df = self._create_df(lvl, obj_start, obj_end, father_idx)
        return self._create_leaves_df(self._init_res, vertices, df)
    
    def _create_leaves_df(self, split_size: int, vertices: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
        cell_size = self._get_bbox_size(df) / split_size
        leaves_list = []
        for idxs in itertools.product(np.arange(split_size), repeat=3):
            idxs = np.array(idxs)
            
            lvl = df["level"] + 1
            lvl.reset_index(drop=True, inplace=True)
            
            bbox_start = self._get_bbox_start(df)
            bbox_start.reset_index(drop=True, inplace=True)
            cell_start = bbox_start + cell_size * idxs[None]
            cell_end = bbox_start + cell_size * (idxs[None] + 1)
            
            curr_df = self._concat_df(lvl, cell_start, cell_end)
            leaves_list.append(curr_df)
            
        leaves_df = pd.concat(leaves_list, ignore_index=True)
        is_in_bbox = is_vertex_in_bbox(vertices, self._get_bbox_start(leaves_df).to_numpy(), 
                                       self._get_bbox_end(leaves_df).to_numpy())
        loc = pd.DataFrame(np.where(is_in_bbox, Location.BOUNDARY, Location.UNKNOWN), columns=["loc"])
        return pd.concat([leaves_df, loc], axis=1)
    
    @staticmethod
    def _get_bbox_start(df: pd.DataFrame) -> pd.DataFrame:
        return df[["bbox_x0", "bbox_y0", "bbox_z0"]]
    
    @staticmethod
    def _get_bbox_end(df: pd.DataFrame) -> pd.DataFrame:
        return df[["bbox_x1", "bbox_y1", "bbox_z1"]]
    
    @staticmethod
    def _get_bbox_center(df: pd.DataFrame) -> np.array:
        return (Octree._get_bbox_end(df).to_numpy() + Octree._get_bbox_start(df).to_numpy()) / 2
    
    @staticmethod
    def _get_bbox_size(df: pd.DataFrame) -> np.ndarray:
        return Octree._get_bbox_end(df).to_numpy() - Octree._get_bbox_start(df).to_numpy()
    
    def get_boundary(self):
        return self._tree_df.loc[(self._tree_df['loc'] == Location.BOUNDARY)]
    
    def get_interior(self):
        return self._tree_df.loc[(self._tree_df['loc'] == Location.INSIDE)]
    
    def get_internal_beta(self):
        return self._tree_df.loc[(self._tree_df['loc'] == Location.INSIDE), 'beta']

    def set_internal_beta(self, internal_beta):
        self._tree_df.loc[(self._tree_df['loc'] == Location.INSIDE), 'beta'] = internal_beta

    def get_internal_s_vector(self):
        return self._tree_df.loc[(self._tree_df['loc'] == Location.INSIDE), ["s_1", "s_x", "s_y", "s_z", "s_xy", "s_xz", "s_yz", "s_xx", "s_yy", "s_zz"]]

    def get_boundary_s_vector(self):
        return self._tree_df.loc[(self._tree_df['loc'] == Location.BOUNDARY), ["s_1", "s_x", "s_y", "s_z", "s_xy", "s_xz", "s_yz", "s_xx", "s_yy", "s_zz"]]

    def _set_inner_outter_location(self, mesh_obj: Trimesh):
        is_unknown = self._tree_df["loc"] == Location.UNKNOWN
        centers = self._get_bbox_center(self._tree_df.loc[is_unknown])
        is_inner = fast_winding_number_for_meshes(np.array(mesh_obj.vertices, order='F'), 
                                                  np.array(mesh_obj.faces, order='F'), 
                                                  centers) > 0.5
        self._tree_df.loc[is_unknown, "loc"] = np.where(is_inner, Location.INSIDE, Location.OUTSIDE)
            
    

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



    def build_from_mesh(self, mesh_obj: Trimesh):
        vertices = np.array(mesh_obj.vertices)
        
        self._tree_df = self._build_init_res(vertices)
        for _ in range(1, self._max_level):
            is_bound = self._tree_df['loc'] == Location.BOUNDARY
            self._tree_df = pd.concat([self._tree_df.loc[~is_bound], 
                                       self._create_leaves_df(2, vertices, self._tree_df.loc[is_bound])], 
                                      ignore_index=True)
        self._set_inner_outter_location(mesh_obj)
        self._set_beta()

        self._plot()
        
    def _set_beta(self):
        is_outside = (self._tree_df['loc'] == Location.OUTSIDE)
        beta_vals = np.where(is_outside, 0., 1.)
        beta_df = pd.DataFrame(beta_vals, columns=["beta"])
        self._tree_df = pd.concat([self._tree_df, beta_df], axis=1)

    def set_s_vector(self):
        p0 = self._get_bbox_start(self._tree_df).to_numpy()
        p1 = self._get_bbox_end(self._tree_df).to_numpy()
        size = self._get_bbox_size(self._tree_df)
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

        if 's_1' in self._tree_df.columns:
            self._tree_df[['s_1', 's_x', 's_y', 's_z', 's_xy', 's_xz', 's_yz', 's_xx', 's_yy', 's_zz']] = np.stack((s_1, s_x, s_y, s_z, s_xy, s_xz, s_yz, s_xx, s_yy, s_zz), axis=-1)
        else:
            s_vector = pd.DataFrame(np.stack((s_1, s_x, s_y, s_z, s_xy, s_xz, s_yz, s_xx, s_yy, s_zz), 
                                    axis=-1), 
                                    columns=['s_1', 's_x', 's_y', 's_z', 's_xy', 's_xz', 's_yz', 's_xx', 's_yy', 's_zz'])
            self._tree_df = pd.concat([self._tree_df, s_vector], axis=1)
        