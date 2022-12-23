import math
import os

import trimesh
from trimesh.primitives import Sphere
from trimesh import creation, Trimesh
from trimesh import Scene
from trimesh import load
from trimesh import intersections
from trimesh import util
from trimesh import visual
from trimesh import PointCloud
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt

marker_color = [124, 252, 0, 255]

def apply_rotation_90_over_z(mesh):
    rotate_matrix = trimesh.transformations.rotation_matrix(np.radians(90.0), [0, 0, 1])
    mesh.apply_transform(rotate_matrix)


def apply_rotation_90_over_y(mesh):
    rotate_matrix = trimesh.transformations.rotation_matrix(np.radians(90.0), [0, 1, 0])
    mesh.apply_transform(rotate_matrix)


def apply_rotation_over_x(mesh, degree=90.0):
    rotate_matrix = trimesh.transformations.rotation_matrix(np.radians(degree), [1, 0, 0])
    mesh.apply_transform(rotate_matrix)


def apply_rotation(mesh, degree=90.0, axis=[1, 0, 0]):
    rotate_matrix = trimesh.transformations.rotation_matrix(np.radians(degree), axis)
    mesh.apply_transform(rotate_matrix)


def make_voxel_grid_transparent(voxel_grid):
    return voxel_grid.as_boxes([100, 100, 100, 100])


def apply_transform_to_origin(model, epsilon=15):
    min_bound = model.bounds[0]
    max_bound = model.bounds[1]
    model.apply_translation([-min_bound[0], -min_bound[1], -min_bound[2]])
    model.apply_translation([epsilon, epsilon, 0])


def make_mesh_transparent(model):
    model.visual.face_colors = [100, 100, 100, 100]


def set_sphere_color(sphere, color):
    sphere.visual.face_colors = color


def slice_mesh(mesh, output_file, above_z_to_cut=10):
    cutting_plane_normal = [0, 0, 1]

    lowest_point = [-1, -1, math.inf]
    for i, vertex in enumerate(mesh.vertices):
        if vertex[2] < lowest_point[2]:
            lowest_point = vertex

    point_in_cutting_plane = [lowest_point[0], lowest_point[1], lowest_point[2] + above_z_to_cut]
    sliced_mesh = intersections.slice_mesh_plane(mesh, cutting_plane_normal, point_in_cutting_plane)
    sliced_mesh.export(output_file)


def set_mesh_color(mesh, color):
    mesh.visual.face_colors = color


def find_support_polygon_convex_hull(mesh, z_tolerance=3.0):
    ground_vertices = []
    for index, vertex in enumerate(mesh.vertices):
        if 0 <= vertex[2] <= z_tolerance:
            mesh.visual.vertex_colors[index] = [124, 252, 0, 255]
            ground_vertices.append(vertex)

    return find_convex_hull_2d(ground_vertices)


def find_support_convex_hull_two_meshes(base_mesh, target_mesh, threshold=3.0):
    _, distance, _ = trimesh.proximity.closest_point(base_mesh, target_mesh.vertices)
    filter = distance < threshold
    close_vertices = target_mesh.vertices[filter]
    target_mesh.visual.vertex_colors[filter] = marker_color
    return find_convex_hull_2d(close_vertices)


def find_convex_hull_2d(vertices, should_plot=False):
    points = [[point[0], point[1]] for point in vertices]  ## remove Z
    points = np.asarray(points)
    hull = ConvexHull(points)
    if should_plot:
        plt.plot(points[:, 0], points[:, 1], 'o')
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        plt.show()
    nd_vertices = points[hull.vertices]

    simple_array_hull_points = []
    for point in nd_vertices:
        simple_array_hull_points.append(list(point))

    extended_to_3d = [[point[0], point[1], 0] for point in simple_array_hull_points]
    return simple_array_hull_points, extended_to_3d


def create_fully_connected_mesh(vertices, color=None):
    if color is None:
        color = [128, 128, 128, 255]
    count_vertices = len(vertices)
    faces = []
    for i in range(count_vertices):
        for j in range(count_vertices):
            for k in range(count_vertices):
                faces.append([i, j, k])

    connected_mesh = Trimesh(vertices=vertices, faces=faces, face_colors=color)
    return connected_mesh

# model = load('../../resources/spheres.obj')
# model.apply_scale(20.0)
# slice_mesh(model, above_z_to_cut=10,
#            output_file=os.path.join('resources', 'sliced_spheres_raw.stl'))
