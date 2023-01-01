import numpy as np
# from utils.rujum_utils import get_shapes, get_safe_zone, update_center_of_mass, publish_results
from trimesh import Scene, Trimesh
import os
import utils
import balance
import itertools
import time
import datetime
import multiprocessing as mp


class octree_cell:
    def __init__(self, point_start, size, beta_val, level):
        assert(len(point_start)==3 and len(size)==3 and ((beta_val>=0 and beta_val<=1) or beta_val==-1))
        self.point_start=point_start
        self.size = size
        self.point_end=point_start + size
        self.beta_val=beta_val
        self.level = level
        combinations = list(itertools.product([0, 1], repeat=3))
        self.corners = []
        for comb in combinations:
            corner = point_start + size*comb
            self.corners.append(corner)

    def split_cell(self):
        new_cells = []
        combinations = list(itertools.product([0, 1], repeat=3))
        for comb in combinations:
            start_step = comb*self.size/2
            new_cell = octree_cell(self.point_start + start_step, self.size/2, self.beta_val, self.level+1)
            new_cells.append(new_cell)
        return new_cells

##TODO - check if there's a different way... this MIGHT be buggy
#returns 1 if inside, 0 if outside, -1 if partial
def cell_inside_mesh(cell: octree_cell, shape: Trimesh):
    inside = shape.contains(cell.corners + [(cell.point_start+cell.size/2)])
    if (inside==True).all():    return 1
    if (inside==False).all():   return 0
    else:                       return -1

class octree_graph:
    def __init__(self, shape: Trimesh, max_level=7):
        self.max_level=max_level
        max_indices = np.array(np.max(shape.vertices, axis=0))
        min_indices = np.array(np.min(shape.vertices, axis=0))
        bbox = np.array([min_indices, max_indices])

        octree_cells = []
        octree_root_cell = octree_cell(point_start=min_indices, size=max_indices-min_indices, beta_val=-1, level=0)
        self.remaining_cells_to_split = [octree_root_cell]
        current_level=0
       
        start = datetime.datetime.now()
        level_start = datetime.datetime.now()

        while len(self.remaining_cells_to_split) > 0:
            current_cell = self.remaining_cells_to_split.pop(0)
            if current_cell.level>current_level:
                end = datetime.datetime.now()
                print(f"finished level {current_level}, {len(self.remaining_cells_to_split)} remained to split, {len(octree_cells)} already in octree")
                current_level = current_cell.level
                print(f"level took {(end-level_start).total_seconds()} seconds")
                level_start = datetime.datetime.now()


            splitted = self.split_if_needed(current_cell, shape, octree_cells)

            
        end = datetime.datetime.now()

        print(f" \
                finished level {current_level}, \
                {len(self.remaining_cells_to_split)} remained to split, \
                {len(octree_cells)} already in octree")
        print(f"level took {(end-level_start).total_seconds()} seconds")

        print(f"finished building octree. took {(end-start).total_seconds()} seconds")
            
    def split_if_needed(self, cell, shape, octree_cells_list):
        if(cell.level>=self.max_level):
            cell.beta_val=-1
            octree_cells_list.append(cell)
            return

        cell_in_mesh = cell_inside_mesh(cell, shape)
        if cell_in_mesh==-1:
            self.remaining_cells_to_split += cell.split_cell()
        else:
            cell.beta_val=cell_in_mesh
            octree_cells_list.append(cell)


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
  
    graph = octree_graph(shape)

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

