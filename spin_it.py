import argparse
import yaml
from octree import Octree
from io_utils import load_stl #, save_stl
from optimizer import QPOptimizer


def parse_args():
    parser = argparse.ArgumentParser(prog='Spin-it')
    parser.add_argument('-c', '--config', required=True, type=str, dest='config', help='Configurations file path')
    return parser.parse_args()


class SpinIt:
    def __init__(self, init_resolution, max_resolution_levels, gamma_i, gamma_c, gamma_l, calc_type) -> None:
        self._octree_obj = Octree(init_resolution, max_resolution_levels)
        self._optimizer = QPOptimizer(gamma_i, gamma_c, gamma_l, calc_type)
    
    def run(self, mesh_obj):
        self._octree_obj.build_from_mesh(mesh_obj)
        boundary_df = self._octree_obj.get_boundary()
        opt_int_df = self._optimizer(self._octree_obj.get_interior())
        # TODO: concat boundary_df, opt_int_df


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as file:
        configs = yaml.safe_load(file)
        mesh_path = configs.pop("mesh_path")
        spin_it = SpinIt(**configs)
    
    mesh_obj = load_stl(mesh_path)
    new_obj = spin_it.run(mesh_obj)
    
    output_path = args.file_path # TODO: change to new path
    # save_obj(new_obj, output_path)
    
# import numpy as np
# # from utils.rujum_utils import get_shapes, get_safe_zone, update_center_of_mass, publish_results
# from trimesh import Scene, Trimesh
# import os
# import utils
# import balance
# import itertools
# import time
# import datetime
# from multiprocessing import Pool

# from enum import Enum
# class Location(Enum):
#     INSIDE = 1
#     OUTSIDE = 2
#     BOUNDARY = 3
#     UNKNOWN = 0

# class octree_node:
#     def __init__(self, point_start, size, shape: Trimesh, current_level = 0, max_level=7):
#         start = datetime.datetime.now()
        
#         self.shape=shape
#         self.current_level  = current_level
#         self.max_level      = max_level
#         self.location = Location.UNKNOWN

#         self.point_start = point_start
#         self.size = size
#         self.point_end = self.point_start+self.size
#         self.center = self.point_start+self.size/2

#         self.combinations = list(itertools.product([0, 1], repeat=3))
#         self.corners = self.point_start + self.size*self.combinations

#         if current_level>=max_level:
#             self.children=None
#             if self.contains_vertices():
#                 self.location=Location.BOUNDARY
#             else:
#                 if self.is_inside:
#                     self.location=Location.INSIDE
#                 else:
#                     self.location=Location.OUTSIDE
#         else:
#             to_split = self.contains_vertices()

#             if to_split:
#                 self.location = Location.BOUNDARY
#                 children_starts = self.point_start + self.size/2*self.combinations
#                 self.children = []
#                 for i in range(8):
#                     self.children.append(octree_node(point_start=children_starts[i], size=self.size/2, shape=self.shape, current_level=self.current_level+1, max_level=self.max_level))

#             else:
#                 self.children=None
#                 if self.is_inside():
#                     self.location=Location.INSIDE
#                 else:
#                     self.location=Location.OUTSIDE

#         end = datetime.datetime.now()

#         if(self.current_level <=1):
#             locations = self.count_locations()
#             start_checks = datetime.datetime.now()
#             leafs=self.num_of_leafs()
#             inside=locations[0]
#             outside=locations[1]
#             boundary=locations[2]
#             print(f"finished building tree of level {self.current_level}")
#             print(f"num_of_leafs = {leafs}")
#             print(f"num_inside = {inside}")
#             print(f"num_outside = {outside}")
#             print(f"num_boundary = {boundary}")
#             print(f"level took {(end-start).total_seconds()} seconds")

#             print()

#     def is_inside(point):
#         check_if_inside = True
#         if check_if_inside:
#             return fast_winding_number_for_meshes(np.array(shape.vertices), np.array(shape.faces), np.array([self.center])) > 0.5
    
#     def contains_vertices(self):
#         points_gt_x_min = (self.shape.vertices[:,0] > self.point_start[0])
#         points_lt_x_max = (self.shape.vertices[:,0] < self.point_end[0])
#         points_gt_y_min = (self.shape.vertices[:,1] > self.point_start[1])
#         points_lt_y_max = (self.shape.vertices[:,1] < self.point_end[1])
#         points_gt_z_min = (self.shape.vertices[:,2] > self.point_start[2])
#         points_lt_z_max = (self.shape.vertices[:,2] < self.point_end[2])

#         points_in_x = np.logical_and(points_lt_x_max, points_gt_x_min)
#         points_in_y = np.logical_and(points_lt_y_max, points_gt_y_min)
#         points_in_z = np.logical_and(points_lt_z_max, points_gt_z_min)

#         return np.logical_and(np.logical_and(points_in_x, points_in_y), points_in_z).any()

#     def count_locations(self):
#         if(self.is_leaf()):
#             if self.location==Location.BOUNDARY:
#                 return np.array([0,0,1])
#             elif self.location==Location.INSIDE:
#                 return np.array([1,0,0])
#             elif self.location==Location.OUTSIDE:
#                 return np.array([0,1,0])
#             else:
#                 throw("got to leaf with no location. BUG")
            
#         else:
#             res = np.array([0,0,0])
#             for child in self.children:
#                 res += child.count_locations()
#             return res
            
#     def is_leaf(self):
#         return self.children==None

#     def num_of_leafs(self):
#         if(self.is_leaf()):
#             return 1
#         else:
#             res = 0
#             for child in self.children:
#                 res += child.num_of_leafs()
#             return res

# def rujum_balance(src_dir, results_dir):
#     """
#     Balance a given rujum (stack of stones) based on the "make it stand" paper.
#     :param src_dir: path to directory with stl files containing the stones.
#     :param results_dir: path to output directory to save the balanced objects.
#     :return: plot the balanced rujum, and save the stl files to print.
#     """
#     scene = Scene()
#     shape = utils.get_shape(src_dir)  # list of shapes sorted from bottom to top

#     # cell = octree_cell(np.array([1,-20,17]), np.array([0.1,0.1,0.1]), 0.2, 0)
#     # cell_inside_mesh(cell, shape)
#     # select spin axis - point
  
#     max_indices = np.array(np.max(shape.vertices, axis=0))
#     min_indices = np.array(np.min(shape.vertices, axis=0))
#     size = max_indices-min_indices
#     volume = size.prod()
#     print(f"volume is {volume}")

#     graph = octree_node(point_start=min_indices, size=max_indices-min_indices, shape=shape, max_level=5)

    #################################################
    # balance.balance(shape) #, spin_axis)
    
    # cur_center_of_mass = {'location': np.array([0., 0., 0.]), 'mass': 0.}
    # carved_shapes = []
    # for shape_i, cur_shape, next_shape in zip(reversed(range(len(shapes))), reversed(shapes), reversed(next_shapes)):
    #     print('='*50)
    #     print('Processing shape [{}] (working top to bottom)'.format(shape_i))
    #     cur_surface_area = get_safe_zone(cur_shape, next_shape, 0.5)
    #     balanced_mesh = balance.balance(cur_shape, cur_surface_area, cur_center_of_mass, scene=scene)
    #     cur_center_of_mass = update_center_of_mass(cur_center_of_mass, balanced_mesh)
    #     carved_shapes.append(balanced_mesh)
    # publish_results(carved_shapes, results_dir)
    # test_res(scene)
    # scene.show(scene)

# def test_res(scene):
#     import io
#     from PIL import Image
#     import matplotlib.pyplot as plt
#     data = scene.save_image(resolution=(1080,1080))
#     image = np.array(Image.open(io.BytesIO(data)))
#     plt.imshow(image)
#     plt.show()


# if __name__ == '__main__':
#     stl_path = "resources/balloon.stl"
#     # src_dir = os.path.join('resources', rujum_name)
#     results_dir = os.path.join('results', "orang")

#     rujum_balance(stl_path, results_dir)
