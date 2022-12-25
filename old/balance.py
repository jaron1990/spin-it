import math
import time
import trimesh
import numpy as np
from utils.params import *
from copy import deepcopy
import utils.trimesh_utlis as trimesh_utlis
import utils.rujum_utils as rujum_utils


# =========================================
# ============ center of mass =============
# =========================================
def get_center_of_mass(center_of_mass_list):
    sum_masses = sum([center_of_mass["mass"]
                     for center_of_mass in center_of_mass_list])
    center_of_mass_new = sum(
        [center_of_mass["location"] * center_of_mass["mass"] for center_of_mass in center_of_mass_list])
    return center_of_mass_new / sum_masses


def center_mass_over_time(baseline_summation, carved, filled_point_count):
    sums = [0, 0, 0]
    sums[0] = baseline_summation[0] - carved[0]
    sums[1] = baseline_summation[1] - carved[1]
    sums[2] = baseline_summation[2] - carved[2]
    return [sums[0] / filled_point_count, sums[1] / filled_point_count, sums[2] / filled_point_count]


def grid_center_mass(voxel_grid, points=None):
    if points == None:
        points = voxel_grid.points
    sums = [0, 0, 0]
    for point in points:
        sums[0] += point[0]
        sums[1] += point[1]
        sums[2] += point[2]
    return [sums[0] / len(points), sums[1] / len(points), sums[2] / len(points)], sums


def find_target_center_mass(support_polygon_mesh, origin_center_mass):
    h = origin_center_mass[2]
    origin_center_mass_2d = origin_center_mass[:2]
    safety_convex_hull_3d = support_polygon_mesh.vertices
    safety_convex_hull_points_2d = safety_convex_hull_3d[:, :2]
    closet_point_on_safety_convex = rujum_utils.find_closet_point_to_convex_hull(safety_convex_hull_points_2d,
                                                                     origin_center_mass_2d)
    # print("closet_point_on_safety_convex", closet_point_on_safety_convex)
    in_safety_region = rujum_utils.equal_points(
        closet_point_on_safety_convex, origin_center_mass_2d)
    if in_safety_region:
        return np.array(origin_center_mass[:2]), safety_convex_hull_points_2d, safety_convex_hull_3d
    else:
        return np.array(closet_point_on_safety_convex[:2]), safety_convex_hull_points_2d, safety_convex_hull_3d


def get_marker(loc, color, radius=marker_radius):
    marker = trimesh.creation.uv_sphere(radius=radius)
    marker.apply_translation(loc)
    marker.visual.face_colors = color
    return marker


# =========================================
# =============== balance =================
# =========================================
def balance(shape, safe_zone, cumulative_center_of_mass, should_carve=True, scene=None):
    shape.visual.face_colors = grey_transp
    scene_dict = {
        'shape': shape
    }

    safe_zone_height = np.min(shape.vertices[:, 2])
    scene_dict['safe_zone_to_vis'] = deepcopy(safe_zone)
    scene_dict['safe_zone_to_vis'].vertices[:, 2] = safe_zone_height

    current_shape_center_mass = {
        "location": shape.center_mass, "mass": shape.mass}
    total_center_mass = get_center_of_mass(
        [current_shape_center_mass, cumulative_center_of_mass])

    scene_dict['total_center_mass'] = get_marker(
        total_center_mass, magenta_solid)
    scene_dict['current_shape_center_mass'] = get_marker(
        current_shape_center_mass['location'], gray_solid)

    # hull_vertices = safe_zone.vertices
    # vertices_2d = [[vert[0], vert[1]] for vert in hull_vertices]
    target_center_mass, safety_hull_2d, safety_hull_3d = find_target_center_mass(
        safe_zone, total_center_mass)
    scene_dict['safety_hull_3d'] = trimesh_utlis.create_fully_connected_mesh(
        safety_hull_3d)
    scene_dict['safety_hull_3d'].visual.face_colors = green_solid
    scene_dict['safety_hull_3d'].vertices[:, 2] = safe_zone_height + .1

    if (target_center_mass == total_center_mass[:2]).all():
        print("[RESULT] Model already balanced, no carving needed")
        should_carve = False

    target_center_mass = list(target_center_mass) + [
        total_center_mass[2]]  # add arbitrary z value to present it, and later for plane calculation
    scene_dict['target_center_mass_sphere'] = get_marker(target_center_mass,
                                                         green_solid)  # FIXME - locations looks wrong

    # z is from origin to be on the same "level"
    cutting_plane_normal = list(
        total_center_mass[:2] - target_center_mass[:2]) + [0.]

    min_length = np.min(np.max(shape.vertices, axis=0) -
                        np.min(shape.vertices, axis=0))
    voxel_size = min_length / \
        20.  # A heuristic that enshures at least 20 voxel for the shortest dim, hopefully it's enough (it makes about 200K voxels)
    voxel_grid = shape.voxelized(pitch=voxel_size)
    interior_points, outer_points = rujum_utils.get_inner_outer_points(voxel_grid)
    # scene_dict['interior_points'] = get_voxel_mesh(interior_points, voxel_size, gray_solid)
    voxel_grid.fill()
    mass_per_voxel = shape.mass / voxel_grid.filled_count

    # FIXME - set should_carve according to the hollowed mesh

    res = {
        "location": np.array(current_shape_center_mass["location"]),
        "orig_mass": shape.mass,
        "mass": len(outer_points) * mass_per_voxel,
        # FIXME - is this a good approx for curved mesh? it heavily depends on voxel size!
        "full_half_mesh": None,
        "carved_half_mesh": shape
    }

    if should_carve:
        distances = trimesh.points.point_plane_distance(
            interior_points, cutting_plane_normal, target_center_mass)
        arr = distances > 0
        trues = [i for i, x in enumerate(arr) if x]
        # print("num of voxels we can carve {} / {} ".format(len(trues), len(distances)))

        distances = distances.astype(int)
        distances_argsort = np.argsort(distances)

        approximated_center_mass, baseline_sums = grid_center_mass(voxel_grid)
        # TODO - add cumulative
        filled_count = voxel_grid.filled_count

        print("Approximated center of mass is : %s, %s, %s " % (approximated_center_mass[0],
                                                                approximated_center_mass[1],
                                                                approximated_center_mass[2]))
        print("Center of mass by library : " + str(total_center_mass))
        print('*' * print_start_repeat)
        index = len(interior_points) - 1
        current_distance_carving = distances[distances_argsort[index]]
        distance_to_arbitrary_point_dict = dict()
        distance_to_arbitrary_point_dict[current_distance_carving] = interior_points[distances_argsort[index]]
        points_to_carve = [interior_points[distances_argsort[index]]]
        # scene_dict['points_to_carve'] = get_voxel_mesh(points_to_carve, voxel_size, gray_solid)
        total_center_mass_dict = {}
        local_center_mass_dict = {}

        # Backward to be descending order
        carved_so_far = np.zeros(3)
        while index >= 0:
            index -= 1
            if distances[distances_argsort[index]] == current_distance_carving:
                # adding points to carve with same distance
                points_to_carve.append(
                    interior_points[distances_argsort[index]])

            else:
                # Carve all points in current distance level
                filled_count -= len(points_to_carve)
                curr_carved_summation = np.sum(points_to_carve, axis=0)
                carved_so_far += curr_carved_summation
                new_center_mass = {
                    'location': np.array(center_mass_over_time(baseline_sums, carved_so_far, filled_count)),
                    'mass': filled_count * mass_per_voxel
                }
                new_center_mass_total = get_center_of_mass(
                    [new_center_mass, cumulative_center_of_mass])
                total_center_mass_dict[current_distance_carving] = new_center_mass_total
                local_center_mass_dict[current_distance_carving] = new_center_mass['location']
                # Define new current carving distance depth
                current_distance_carving = distances[distances_argsort[index]]
                # new points to explore
                points_to_carve = [interior_points[distances_argsort[index]]]
                distance_to_arbitrary_point_dict[current_distance_carving] = interior_points[distances_argsort[index]]
            if distances[distances_argsort[index]] < 0:
                # negative distance, no need to carve
                break
        clock_2 = time.clock()
        # print("Carving process algorithm took {} seconds".format(round(clock_2 - clock_1, 2)))
        # print('*' * print_start_repeat)
        best_dist = math.inf
        best_key = None
        for carved_distance, carved_center_mass in total_center_mass_dict.items():
            distance_from_target_center = np.linalg.norm(
                carved_center_mass[:2] - np.array(target_center_mass[:2]))
            # if distance_from_target_center < best_dist:
            if distance_from_target_center < best_dist:
                best_dist = distance_from_target_center
                best_key = carved_distance

        final_cumulative_center_mass = total_center_mass_dict[best_key]
        final_center_mass = local_center_mass_dict[best_key]

        print("Target center of mass is {}".format(target_center_mass))
        print("Center of mass achieved by carving process is {} ".format(
            final_center_mass))
        print('*' * print_start_repeat)
        scene_dict['final_center_mass'] = get_marker(
            final_center_mass, red_solid)
        scene_dict['final_cumulative_center_mass'] = get_marker(
            final_cumulative_center_mass, blue_solid)

        # Draw the way to carve
        scene_dict['caved_traj'] = []
        for carved_distance, carved_center_mass in total_center_mass_dict.items():
            scene_dict['caved_traj'].append(get_marker(
                carved_center_mass, yellow_solid, radius=traj_radius))

        closet_point_on_safety_convex = rujum_utils.find_closet_point_to_convex_hull(safety_hull_2d,
                                                                         final_cumulative_center_mass)
        in_safety_region = rujum_utils.equal_points(
            closet_point_on_safety_convex, final_cumulative_center_mass)
        if not in_safety_region:
            print(
                "[RESULT] Model is not stable yet, consider to deform the model and run again")
        else:
            print("[RESULT] Model is stable after carving process!")

        arbitrary_point_in_min_enregy = distance_to_arbitrary_point_dict[best_key]
        carved_half_mesh = trimesh.intersections.slice_mesh_plane(shape, cutting_plane_normal,
                                                                  arbitrary_point_in_min_enregy, cap=False)
        carved_half_mesh.visual.face_colors = transparent_deep_pink

        invert_normal = [-1 * cutting_plane_normal[0], -1 *
                         cutting_plane_normal[1], cutting_plane_normal[2]]
        full_half_mesh = trimesh.intersections.slice_mesh_plane(shape, invert_normal, arbitrary_point_in_min_enregy,
                                                                cap=True)
        full_half_mesh.visual.face_colors = transparent_lemonchiffon

        scene_dict['carved_half_mesh'] = carved_half_mesh
        # scene_dict['carved_half_mesh'] = trimesh.boolean.difference([carved_half_mesh])
        scene_dict['full_half_mesh'] = full_half_mesh

        res = {
            "location": final_center_mass,
            "orig_mass": shape.mass,
            "mass": filled_count * mass_per_voxel,
            "full_half_mesh": full_half_mesh,
            "carved_half_mesh": carved_half_mesh
        }

    # add objects to scene
    if scene is not None:
        for v in scene_dict.values():
            if type(v) == list:
                for it in v:
                    scene.add_geometry(it)
            else:
                scene.add_geometry(v)

    return res
