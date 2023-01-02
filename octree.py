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
    
    def _calc_loc(self, vertices, point_start, point_end):
        if is_vertex_in_bbox(vertices, point_start, point_end):
            return Location.BOUNDARY
        if is_bbox_inside_mesh(): # TODO: fix
            return Location.INSIDE
        return Location.OUTSIDE
    
    def _build_init_res(self, mesh_obj: Trimesh):
        vertices = np.array(mesh_obj.vertices)
        cell_x, cell_y, cell_z = (vertices.max(axis=0) - vertices.min(axis=0)) / self._init_res
        
        leaf_list = []
        for i, j, k in itertools.product(np.arange(self._init_res), repeat=3):
            point_start = (i * cell_x, j * cell_y, k * cell_z)
            point_end = ((i + 1) * cell_x, (j + 1) * cell_y, (k + 1) * cell_z)
            loc = self._calc_loc(vertices, point_start, point_end)
            leaf = {"level": 0, "M": 0,
                    "bbox_start": point_start,
                    "bbox_end": point_end,
                    "loc": loc, "father_idx": None}
            leaf.update({f"child{idx}": None for idx in range(8)})
            leaf_list.append(leaf)
        return pd.DataFrame.from_records(leaf_list)
    
    def _split_leaves(self, lvl, leaves_df: pd.DataFrame, mesh_obj: Trimesh):
        vertices = np.array(mesh_obj.vertices)
        for leaf in leaves_df:
            cell_x = (leaf["x1"] - leaf["x0"]) / 2
            cell_y = (leaf["y1"] - leaf["y0"]) / 2
            cell_z = (leaf["z1"] - leaf["z0"]) / 2
            
            new_leaves = []
            for i, j, k in itertools.product([0, 1], repeat=3):
                point_start = (leaf["x0"] + i * cell_x, leaf["y0"] + j * cell_y, leaf["z0"] + k * cell_z)
                point_end = (leaf["x0"] + (i + 1) * cell_x, leaf["y0"] + (j + 1) * cell_y, leaf["z0"] + (k + 1) * cell_z)
                loc = self._calc_loc(vertices, point_start, point_end)
                new_leaf = {"level": lvl, "M": 0,
                            "bbox_start": point_start,
                            "bbox_end": point_end,
                            "loc": loc, "father_idx": new_leaf.index}
                new_leaf.update({f"child{idx}": None for idx in range(8)})
                new_leaves.append(new_leaf)
            # update father's children
            # update df
            # pd.DataFrame.from_records(new_leaves)
            
    
    def build_from_mesh(self, mesh_obj: Trimesh):
        self._tree_df = self._build_init_res(mesh_obj)
        for lvl in range(1, self._max_level):
            curr_df = self._tree_df.loc[(self._tree_df['lvl'] == lvl - 1) & (self._tree_df['loc'] == Location.BOUNDARY)]
            self._split_leaves(curr_df, mesh_obj)
        