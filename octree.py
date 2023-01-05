import operator
from trimesh import Trimesh
import numpy as np
import itertools
import pandas as pd
from utils import is_vertex_in_bbox, Location #, is_bbox_inside_mesh


class Octree:
    def __init__(self, init_res: int, max_level: int) -> None:
        self._init_res = init_res
        self._max_level = max_level
        self._tree_df = None
    
    @staticmethod
    def _create_df(lvl, cell_start, cell_end, father_idx=None, loc=None):
        return pd.DataFrame.from_dict([{"level": lvl, "M": 0, 
                                       "bbox_x0": cell_start[0], "bbox_y0": cell_start[1], "bbox_z0": cell_start[2], 
                                       "bbox_x1": cell_end[0], "bbox_y1": cell_end[1], "bbox_z1": cell_end[2], 
                                       "loc": loc, "father_idx": father_idx}])
    
    @staticmethod
    def _concat_df(lvl, cell_start, cell_end, father_idx, M=None):
        if "bbox_x0" in cell_end.keys():
            cell_end.rename(columns={"bbox_x0": "bbox_x1", "bbox_y0": "bbox_y1", "bbox_z0": "bbox_z1"}, inplace=True)
        return pd.concat([lvl, cell_start, cell_end, father_idx], axis=1)
    
    def _build_init_res(self, vertices: np.array):
        obj_start = vertices.min(axis=0)
        obj_end = vertices.max(axis=0)
        lvl = -1
        father_idx = None
        df = self._create_df(lvl, obj_start, obj_end, father_idx)
        return self._create_leaves_df(self._init_res, vertices, df)
    
    def _create_leaves_df(self, split_size: int, vertices: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
        cell_size = self.get_bbox_size(df) / split_size
        leaves_list = []
        for idxs in itertools.product(np.arange(split_size), repeat=3):
            idxs = np.array(idxs)
            if (df["level"] == -1).all():
                father_idx = pd.DataFrame(np.array([None]), columns=["father_idx"])
            else:
                father_idx = pd.DataFrame(df.index.values, columns=["father_idx"])
            
            lvl = df["level"] + 1
            lvl.reset_index(drop=True, inplace=True)
            
            bbox_start = self.get_bbox_start(df)
            bbox_start.reset_index(drop=True, inplace=True)
            cell_start = bbox_start + cell_size * idxs[None]
            cell_end = bbox_start + cell_size * (idxs[None] + 1)
            
            curr_df = self._concat_df(lvl, cell_start, cell_end, father_idx)
            leaves_list.append(curr_df)
            
        leaves_df = pd.concat(leaves_list, ignore_index=True)
        is_in_bbox = is_vertex_in_bbox(vertices, self.get_bbox_start(leaves_df).to_numpy(), 
                                       self.get_bbox_end(leaves_df).to_numpy())
        loc = pd.DataFrame(np.where(is_in_bbox, Location.BOUNDARY, None), columns=["loc"])
        return pd.concat([leaves_df, loc], axis=1)
    
    @staticmethod
    def get_bbox_start(df: pd.DataFrame) -> pd.DataFrame:
        return df[["bbox_x0", "bbox_y0", "bbox_z0"]]
    
    @staticmethod
    def get_bbox_end(df: pd.DataFrame) -> pd.DataFrame:
        return df[["bbox_x1", "bbox_y1", "bbox_z1"]]
    
    @staticmethod
    def get_bbox_size(df: pd.DataFrame) -> np.ndarray:
        return Octree.get_bbox_end(df).to_numpy() - Octree.get_bbox_start(df).to_numpy()
    
    # def get_leaves(self):
    #     return self._tree_df.loc[(self._tree_df['level'])]
            
    def build_from_mesh(self, mesh_obj: Trimesh):
        vertices = np.array(mesh_obj.vertices)
        self._tree_df = self._build_init_res(vertices)
        for lvl in range(1, self._max_level):
            curr_df = self._tree_df.loc[(self._tree_df['level'] == lvl - 1) & (self._tree_df['loc'] == Location.BOUNDARY)]
            self._tree_df = pd.concat([self._tree_df, self._create_leaves_df(2, vertices, curr_df)], ignore_index=True)
        