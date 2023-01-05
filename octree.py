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
    
    def _build_init_res(self, mesh_obj: Trimesh):
        vertices = np.array(mesh_obj.vertices)
        obj_start = vertices.min(axis=0)
        obj_end = vertices.max(axis=0)
        cell_size = (obj_end - obj_start) / self._init_res
        
        lvl = 0
        father_idx = None
        leaves_list = self._create_leaves_list(self._init_res, obj_start, cell_size, vertices, lvl, father_idx)
        return pd.DataFrame.from_records(leaves_list)
    
    def _create_leaves_list(self, split_size: int, position_start: np.ndarray, cell_size: tuple[float, float, float],
                            vertices: np.ndarray, lvl: int, father_idx: int | None
                            ) -> list[dict[str, int | tuple[float,float,float]]]:
        cell_size = np.array(cell_size)
        leaves_list = []
        for idxs in itertools.product(np.arange(split_size), repeat=3):
            idxs = np.array(idxs)
            cell_start = position_start + idxs * cell_size
            cell_end = position_start + (idxs + 1) * cell_size
            
            loc = Location.BOUNDARY if is_vertex_in_bbox(vertices, cell_start, cell_end) else None
            leaf = {"level": lvl, "M": 0, "bbox_start": cell_start, "bbox_end": cell_end, "loc": loc, 
                    "father_idx": father_idx}
            leaf.update({f"child{idx}": None for idx in range(8)})
            leaves_list.append(leaf)
        return leaves_list
    
    def _split_bound_cells(self, lvl: int, leaves_df: pd.DataFrame, mesh_obj: Trimesh):
        vertices = np.array(mesh_obj.vertices)
        for leaf in leaves_df:
            cell_size = (leaf["bbox_end"] - leaf["bbox_start"]) / 2
            self._create_leaves_list(2, leaf["bbox_start"], cell_size, vertices, lvl, leaf.index)
            # update father's children
            # update df
            # pd.DataFrame.from_records(new_leaves)
            
    
    def build_from_mesh(self, mesh_obj: Trimesh):
        self._tree_df = self._build_init_res(mesh_obj)
        for lvl in range(1, self._max_level):
            curr_df = self._tree_df.loc[(self._tree_df['level'] == lvl - 1) & (self._tree_df['loc'] == Location.BOUNDARY)]
            self._split_bound_cells(lvl, curr_df, mesh_obj)
        