import numpy as np
from utils.rujum_utils import get_shapes, get_safe_zone, update_center_of_mass, publish_results
from trimesh import Scene
import os
import balance


def rujum_balance(src_dir, results_dir):
    """
    Balance a given rujum (stack of stones) based on the "make it stand" paper.
    :param src_dir: path to directory with stl files containing the stones.
    :param results_dir: path to output directory to save the balanced objects.
    :return: plot the balanced rujum, and save the stl files to print.
    """
    scene = Scene()
    shapes = get_shapes(src_dir)  # list of shapes sorted from bottom to top
    next_shapes = [None] + shapes[:-1]  # points to the shape underneath current shape
    cur_center_of_mass = {'location': np.array([0., 0., 0.]), 'mass': 0.}
    carved_shapes = []
    for shape_i, cur_shape, next_shape in zip(reversed(range(len(shapes))), reversed(shapes), reversed(next_shapes)):
        print('='*50)
        print('Processing shape [{}] (working top to bottom)'.format(shape_i))
        cur_surface_area = get_safe_zone(cur_shape, next_shape, 0.5)
        balanced_mesh = balance.balance(cur_shape, cur_surface_area, cur_center_of_mass, scene=scene)
        cur_center_of_mass = update_center_of_mass(cur_center_of_mass, balanced_mesh)
        carved_shapes.append(balanced_mesh)
    publish_results(carved_shapes, results_dir)
    test_res(scene)
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
    rujum_name = 'RUJUM_4'
    src_dir = os.path.join('resources', rujum_name)
    results_dir = os.path.join('results', rujum_name)
    rujum_balance(src_dir, results_dir)

