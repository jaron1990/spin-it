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
    INSIDE = 1
    OUTSIDE = 2
    BOUNDARY = 3


def is_vertex_in_bbox(vertices, point_start, point_end):
    is_in_x = np.logical_and(vertices[:,0] > point_start[0], vertices[:,0] < point_end[0])
    is_in_y = np.logical_and(vertices[:,1] > point_start[1], vertices[:,1] < point_end[1])
    is_in_z = np.logical_and(vertices[:,2] > point_start[2], vertices[:,2] < point_end[2])
    return np.logical_and(np.logical_and(is_in_x, is_in_y), is_in_z).any()
