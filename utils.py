import trimesh


def get_shape(stl_path):
    return trimesh.load(stl_path)


def get_inner_outer_points(voxel_grid):
    voxel_grid.fill()
    all_points_in_voxel_grid_filled = [
        [p[0], p[1], p[2]] for p in voxel_grid.points]
    voxel_grid.hollow()
    outer_points = [[p[0], p[1], p[2]] for p in voxel_grid.points]
    interior_points = [
        point for point in all_points_in_voxel_grid_filled if point not in outer_points]
    return interior_points, outer_points