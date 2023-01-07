from trimesh import Trimesh
import numpy as np
import itertools
import pandas as pd
from utils import is_vertex_in_bbox, Location #, is_bbox_inside_mesh
from igl import fast_winding_number_for_meshes


class Octree:
    def __init__(self, init_res: int, max_level: int) -> None:
        self._init_res = init_res
        self._max_level = max_level
        self._tree_df = None
    
    @staticmethod
    def _create_df(lvl, cell_start, cell_end, father_idx=None, loc=Location.UNKNOWN):
        return pd.DataFrame.from_dict([{"level": lvl, "M": 0, 
                                       "bbox_x0": cell_start[0], "bbox_y0": cell_start[1], "bbox_z0": cell_start[2], 
                                       "bbox_x1": cell_end[0], "bbox_y1": cell_end[1], "bbox_z1": cell_end[2], 
                                       "loc": loc}])
    
    @staticmethod
    def _concat_df(lvl, cell_start, cell_end, M=None):
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
        centers = self._get_bbox_center(self._tree_df.loc[self._tree_df["loc"] == Location.UNKNOWN])
        is_inner = fast_winding_number_for_meshes(np.array(mesh_obj.vertices), 
                                                  np.array(mesh_obj.faces), 
                                                  centers) > 0.5
        
            
    def build_from_mesh(self, mesh_obj: Trimesh):
        vertices = np.array(mesh_obj.vertices)
        
        self._tree_df = self._build_init_res(vertices)
        for _ in range(1, self._max_level):
            is_bound = self._tree_df['loc'] == Location.BOUNDARY
            self._tree_df = pd.concat([self._tree_df.loc[~is_bound], 
                                       self._create_leaves_df(2, vertices, self._tree_df.loc[is_bound])], 
                                      ignore_index=True)
        self._set_inner_outter_location(mesh_obj)
        