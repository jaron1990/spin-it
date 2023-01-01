import numpy as np
# from utils.rujum_utils import get_shapes, get_safe_zone, update_center_of_mass, publish_results
from trimesh import Scene, Trimesh
import os
import utils
import balance
import itertools
import time
import datetime
from multiprocessing import Pool


class octree_node:
    def __init__(self, point_start, size, shape: Trimesh, current_level = 0, max_level=7):
        start = datetime.datetime.now()
        
        self.shape=shape
        self.current_level  = current_level
        self.max_level      = max_level
        self.deepest_level = current_level

        self.point_start = point_start
        self.size = size
        self.point_end = self.point_start+self.size

        self.combinations = list(itertools.product([0, 1], repeat=3))
        self.corners = self.point_start + self.size*self.combinations

        is_inside = self.is_inside_mesh(shape)

        if is_inside==1:
            self.beta_val=1
            self.children=None
        elif is_inside==0:
            self.beta_val=0
            self.children=None
        else:
            self.beta_val=-1
            if current_level>=max_level:
                self.children=None
            else:
                self.children_starts = self.point_start + self.size/2*self.combinations

                # # for parallel (not working because of some issue in trimesh):
                # with Pool(8) as p:    
                #     self.children = p.map(self.build_octree_node, range(8))
                #     for child in self.children:
                #         self.deepest_level = max(self.deepest_level, child.deepest_level)

                # for series:
                self.children = []
                for i in range(8):
                    self.children.append(octree_node(point_start=self.children_starts[i], size=self.size/2, shape=self.shape, current_level=self.current_level+1, max_level=self.max_level))
                    self.deepest_level = max(self.deepest_level, self.children[-1].deepest_level)

        end = datetime.datetime.now()

        if(self.current_level <=2):
            print(f"finished building tree of level {self.current_level}. deepest level was {self.deepest_level}")
            print(f"level took {(end-start).total_seconds()} seconds")
            
    def build_octree_node(self, i):
        child = octree_node(point_start=self.children_starts[i], size=self.size/2, shape=self.shape, current_level=self.current_level+1, max_level=self.max_level)
        print(self.children_starts[i])
        print(self.size/2)
        print(self.current_level+1, self.max_level)
        print(f"is child inside mesh? {child.is_inside_mesh(self.shape)}")
        return child

    def split_if_needed(self, cell, shape, octree_cells_list):
        if(cell.level>=self.max_level):
            cell.beta_val=-1
            octree_cells_list.append(cell)
            return

        cell_in_mesh = self.is_inside_mesh(shape)
        if cell_in_mesh==-1:
            self.remaining_cells_to_split += cell.split_cell()
        else:
            cell.beta_val=cell_in_mesh
            octree_cells_list.append(cell)

    ##TODO - check if there's a different way... this MIGHT be buggy
    #returns 1 if inside, 0 if outside, -1 if partial
    def is_inside_mesh(self, shape: Trimesh):
        inside = shape.contains(list(self.corners) + [(self.point_start+self.size/2)])
        if (inside==True).all():    return 1
        if (inside==False).all():   return 0
        else:                       return -1

def rujum_balance(src_dir, results_dir):
    """
    Balance a given rujum (stack of stones) based on the "make it stand" paper.
    :param src_dir: path to directory with stl files containing the stones.
    :param results_dir: path to output directory to save the balanced objects.
    :return: plot the balanced rujum, and save the stl files to print.
    """
    scene = Scene()
    shape = utils.get_shape(src_dir)  # list of shapes sorted from bottom to top

    # cell = octree_cell(np.array([1,-20,17]), np.array([0.1,0.1,0.1]), 0.2, 0)
    # cell_inside_mesh(cell, shape)
    # select spin axis - point
  
    max_indices = np.array(np.max(shape.vertices, axis=0))
    min_indices = np.array(np.min(shape.vertices, axis=0))
    
    graph = octree_node(point_start=min_indices, size=max_indices-min_indices, shape=shape, max_level=4)

    balance.balance(shape) #, spin_axis)
    
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

def test_res(scene):
    import io
    from PIL import Image
    import matplotlib.pyplot as plt
    data = scene.save_image(resolution=(1080,1080))
    image = np.array(Image.open(io.BytesIO(data)))
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    stl_path = "resources/GiantToad.stl"
    # src_dir = os.path.join('resources', rujum_name)
    results_dir = os.path.join('results', "orang")

    rujum_balance(stl_path, results_dir)

