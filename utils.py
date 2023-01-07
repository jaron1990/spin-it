import numpy as np
from enum import Enum
# import trimesh


# def get_inner_outer_points(voxel_grid):
#     voxel_grid.fill()
#     all_points_in_voxel_grid_filled = [
#         [p[0], p[1], p[2]] for p in voxel_grid.points]
#     voxel_grid.hollow()
#     outer_points = [[p[0], p[1], p[2]] for p in voxel_grid.points]
#     interior_points = [
#         point for point in all_points_in_voxel_grid_filled if point not in outer_points]
#     return interior_points, outer_points

class Location(Enum):
    UNKNOWN = 0
    INSIDE = 1
    OUTSIDE = 2
    BOUNDARY = 3


def is_vertex_in_bbox(vertices: np.ndarray, cell_start: np.ndarray, cell_end: np.ndarray):
    vertices = np.expand_dims(vertices,axis=1)
    log_and = np.logical_and(vertices > cell_start[None], vertices < cell_end[None])
    return np.logical_and(np.logical_and(log_and[..., 0], log_and[..., 1]), log_and[..., 2]).any(axis=0)
