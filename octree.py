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
    
    def _calc_loc(self, vertices, point_start, point_end):
        if is_vertex_in_bbox(vertices, point_start, point_end):
            return Location.BOUNDARY
        if is_bbox_inside_mesh(): # TODO: fix
            return Location.INSIDE
        return Location.OUTSIDE
    
    def _build_init_res(self, mesh_obj: Trimesh):
        vertices = np.array(mesh_obj.vertices)
        cell_size = (vertices.max(axis=0) - vertices.min(axis=0)) / self._init_res
        
        leaves_list = self._create_leaves_list(itertools.product(np.arange(self._init_res), repeat=3), (0, 0, 0), 
                                               cell_size, vertices, 0, None)
        return pd.DataFrame.from_records(leaves_list)
    
    def _create_leaves_list(self, iterator: itertools.product, init_point: tuple[float, float, float], 
                            cell_size:tuple[float, float, float], vertices: np.ndarray, lvl: int, father_idx: int | None
                            ) -> list[dict[str, int | tuple[float,float,float]]]:
        leaves_list = []
        for idxs in iterator:
            point_start = tuple(map(operator.mul, idxs, cell_size))
            point_start = tuple(map(operator.add, point_start, init_point))
            
            point_end = tuple(map(operator.add, point_start, (1,) * 3))
            point_end = tuple(map(operator.mul, idxs, cell_size))
            point_end = tuple(map(operator.add, point_start, init_point))
            
            loc = self._calc_loc(vertices, point_start, point_end)
            leaf = {"level": lvl, "M": 0, "bbox_start": point_start, "bbox_end": point_end, "loc": loc, 
                    "father_idx": father_idx}
            leaf.update({f"child{idx}": None for idx in range(8)})
            leaves_list.append(leaf)
        return leaves_list
    
    def _split_leaves(self, lvl, leaves_df: pd.DataFrame, mesh_obj: Trimesh):
        vertices = np.array(mesh_obj.vertices)
        for leaf in leaves_df:
            cell_size = (leaf["bbox_end"] - leaf["bbox_start"]) / 2
            self._create_leaves_list(itertools.product([0, 1], repeat=3), leaf["bbox_start"], cell_size, vertices, lvl, 
                                     leaf.index)
            # update father's children
            # update df
            # pd.DataFrame.from_records(new_leaves)
            
    
    def build_from_mesh(self, mesh_obj: Trimesh):
        self._tree_df = self._build_init_res(mesh_obj)
        for lvl in range(1, self._max_level):
            curr_df = self._tree_df.loc[(self._tree_df['lvl'] == lvl - 1) & (self._tree_df['loc'] == Location.BOUNDARY)]
            self._split_leaves(curr_df, mesh_obj)
        
