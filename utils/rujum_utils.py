import math
import os
from copy import deepcopy
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from utils.params import *

import utils.trimesh_utlis as trimesh_utlis

# =========================================
# ========= main alg utils ==========
# =========================================
def get_safe_zone(cur_shape, next_shape, threshold):
    # next shape is below cur shape

    # calc the upper plane of the next shape
    # calc the intersection of the plane and cur_shape
    # calc safe zone inside it with regards to center of mass

    # returns a convex hull of the safe surface area

    if next_shape is None:
        vertices_2d, hull_vertices = trimesh_utlis.find_support_polygon_convex_hull(cur_shape, z_tolerance=threshold)
    else:
        vertices_2d, hull_vertices = trimesh_utlis.find_support_convex_hull_two_meshes(
            base_mesh=next_shape, target_mesh=cur_shape, threshold=threshold)
    support_polygon_mesh = trimesh_utlis.create_fully_connected_mesh(hull_vertices)

    # target_center_mass, safety_hull_2d, safety_hull_3d = find_target_center_mass_safety_region(support_polygon_mesh,
    #                                                                                            cur_shape.center_mass,
    #                                                                                            vertices_2d,
    #                                                                                            2.)
    # print("safety_hull_3d", safety_hull_3d)
    # support_polygon_mesh = trimesh_utlis.create_fully_connected_mesh(safety_hull_3d)

    return support_polygon_mesh


def update_center_of_mass(cur_center_of_mass, carved_mesh):
    total_mass = cur_center_of_mass['mass'] + carved_mesh['mass']
    loc = (cur_center_of_mass['mass'] * cur_center_of_mass['location'] + carved_mesh['mass'] * carved_mesh[
        'location']) / total_mass
    return {'location': loc, 'mass': total_mass}


def get_shapes(stl_dir):
    # assuming stl file for each stone
    stl_path_list = [os.path.join(stl_dir, f) for f in os.listdir(stl_dir) if f.endswith('.stl')]
    shapes = [trimesh.load(stl_path) for stl_path in stl_path_list]
    shapes.sort(key=lambda mesh: mesh.center_mass[2])  # sort according to height
    return shapes


def publish_results(carved_shapes, results_dir):
    print('=' * print_start_repeat)
    print('Saving result STLs in [{}]'.format(results_dir))
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    for i, shape in enumerate(carved_shapes):
        if shape['full_half_mesh'] is not None:
            for part in ['full', 'carved']:
                get_mesh_to_publish(shape[f'{part}_half_mesh']).export(
                    os.path.join(results_dir, 'shape{:02d}_{}.stl'.format(i, part)))
        else:
            splited_mesh = split_carved_mesh(shape['carved_half_mesh'])
            for k in splited_mesh.keys():
                get_mesh_to_publish(splited_mesh[k]).export(
                    os.path.join(results_dir, 'shape{:02d}_carved_{}.stl'.format(i, k)))


def split_carved_mesh(mesh):
    middle_z = np.mean([mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()])
    return {
        'top': trimesh.intersections.slice_mesh_plane(mesh, (0., 0., 1.), (0., 0., middle_z), cap=False),
        'bottom': trimesh.intersections.slice_mesh_plane(mesh, (0., 0., -1.), (0., 0., middle_z), cap=False),
    }


def get_mesh_to_publish(mesh):
    publish_mesh = deepcopy(mesh)
    publish_mesh.apply_scale(published_mesh_scale)
    mesh_bottom = publish_mesh.vertices[:, 2].min()
    mesh_x_center = np.mean([publish_mesh.vertices[:, 0].min(), publish_mesh.vertices[:, 0].max()])
    mesh_y_center = np.mean([publish_mesh.vertices[:, 1].min(), publish_mesh.vertices[:, 1].max()])
    publish_mesh.apply_translation((-mesh_x_center, -mesh_y_center, -mesh_bottom))
    return publish_mesh


# =========================================
# ========= safety region utils ===========
# =========================================
def get_inner_outer_points(voxel_grid):
    voxel_grid.fill()
    all_points_in_voxel_grid_filled = [
        [p[0], p[1], p[2]] for p in voxel_grid.points]
    voxel_grid.hollow()
    outer_points = [[p[0], p[1], p[2]] for p in voxel_grid.points]
    interior_points = [
        point for point in all_points_in_voxel_grid_filled if point not in outer_points]
    return interior_points, outer_points


def find_closet_point_to_convex_hull(convex_hull_points, point):
    convex_hull_points_polygon = [(x[0], x[1]) for x in convex_hull_points]
    poly = Polygon(convex_hull_points_polygon)
    point_to_search = Point(point[0], point[1])
    # The points are returned in the same order as the input geometries:
    p1, p2 = nearest_points(poly, point_to_search)
    return np.array([p1.x, p1.y])


def calc_safety_region_convex_hull(full_convex_polygon_mesh, support_polygon_vertices_2d, h, gamma_angle=5):
    approximate_center_of_convex = get_mid_point_of_convex_hull_2d(full_convex_polygon_mesh)
    saftey_region_convex_hull_points_raw = []
    size_to_move_inside_polygon = math.tan(math.radians(gamma_angle)) * h
    approximate_center_of_convex_numpy = np.array(
        [approximate_center_of_convex[0], approximate_center_of_convex[1]])
    for point in support_polygon_vertices_2d:
        direction_vector = approximate_center_of_convex_numpy - point
        normalized_direction = direction_vector / np.linalg.norm(direction_vector)
        vector_to_add = normalized_direction * size_to_move_inside_polygon
        safety_border = point + vector_to_add
        saftey_region_convex_hull_points_raw.append([safety_border[0], safety_border[1]])
    return find_convex_hull_2d(vertices=saftey_region_convex_hull_points_raw)


def get_mid_point_of_convex_hull_2d(support_polygon_mesh):
    bounds = support_polygon_mesh.bounds
    mins = bounds[0]
    maxs = bounds[1]
    midx = (mins[0] + maxs[0]) / 2
    midy = (mins[1] + maxs[1]) / 2
    return [midx, midy]


def find_convex_hull_2d(vertices):
    points = [[point[0], point[1]] for point in vertices]  ## remove Z
    points = np.asarray(points)
    hull = ConvexHull(points)
    nd_vertices = points[hull.vertices]

    simple_array_hull_points = []
    for point in nd_vertices:
        simple_array_hull_points.append(list(point))

    extended_to_3d = [[point[0], point[1], 0] for point in simple_array_hull_points]
    return simple_array_hull_points, extended_to_3d


def find_target_center_mass_safety_region(support_polygon_mesh, origin_center_mass,
                                          support_polygon_vertices_2d, angle=5):
    h = origin_center_mass[2]
    origin_center_mass_2d = [origin_center_mass[0], origin_center_mass[1]]
    safety_convex_hull_points_2d, safety_convex_hull_3d = calc_safety_region_convex_hull(
        full_convex_polygon_mesh=support_polygon_mesh,
        support_polygon_vertices_2d=support_polygon_vertices_2d,
        h=h,
        gamma_angle=angle)
    closet_point_on_safety_convex = find_closet_point_to_convex_hull(safety_convex_hull_points_2d,
                                                                     origin_center_mass_2d)
    in_safety_region = equal_points(closet_point_on_safety_convex, origin_center_mass_2d)
    if in_safety_region:
        return [origin_center_mass[0], origin_center_mass[1], 5], safety_convex_hull_points_2d, safety_convex_hull_3d
    else:
        return [closet_point_on_safety_convex[0], closet_point_on_safety_convex[1],
                5], safety_convex_hull_points_2d, safety_convex_hull_3d
    

def equal_points(p1, p2):
    return p1[0] == p2[0] and p1[1] == p2[1]

