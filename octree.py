from trimesh import Trimesh
import numpy as np
import itertools
import pandas as pd
from utils import is_vertex_in_bbox, Location #, is_bbox_inside_mesh
from igl import fast_winding_number_for_meshes


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
    
    def _set_inner_outter_location(self, mesh_obj: Trimesh):
        is_unknown = self._tree_df["loc"] == Location.UNKNOWN
        centers = self._get_bbox_center(self._tree_df.loc[is_unknown])
        is_inner = fast_winding_number_for_meshes(np.array(mesh_obj.vertices, order='F'), 
                                                  np.array(mesh_obj.faces, order='F'), 
                                                  centers) > 0.5
        self._tree_df.loc[is_unknown, "loc"] = np.where(is_inner, Location.INSIDE, Location.OUTSIDE)
            
    def build_from_mesh(self, mesh_obj: Trimesh):
        vertices = np.array(mesh_obj.vertices)
        
        self._tree_df = self._build_init_res(vertices)
        for _ in range(1, self._max_level):
            is_bound = self._tree_df['loc'] == Location.BOUNDARY
            self._tree_df = pd.concat([self._tree_df.loc[~is_bound], 
                                       self._create_leaves_df(2, vertices, self._tree_df.loc[is_bound])], 
                                      ignore_index=True)
        self._set_inner_outter_location(mesh_obj)
        
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
        s_vector = pd.DataFrame(s_1, s_x, s_y, s_z, s_xy, s_xz, 
                                s_yz, s_xx, s_yy, s_zz, 
                                columns=["s_1", "s_x", "s_y", "s_z", "s_xy", "s_xz",
                                         "s_yz", "s_xx", "s_yy", "s_zz"])
        self._tree_df = pd.concat([self._tree_df, s_vector], axis=1)
        