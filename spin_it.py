import argparse
import yaml
import torch
from functools import partial
from torch.optim import Adam
from mesh_obj import MeshObj
from octree import Octree, OctreeTensorHandler
from loss import SpinItLoss
from optimizer import QPOptimizer
from model import SpinItModel


def parse_args():
    parser = argparse.ArgumentParser(prog='Spin-it')
    parser.add_argument('-c', '--config', required=True, type=str, dest='config', help='Configurations file path')
    return parser.parse_args()


class SpinIt:
    def __init__(self, octree_configs, optimizer_configs, loss_configs, epsilon) -> None:
        self._octree_obj = Octree(**octree_configs)
        self._loss = SpinItLoss(**loss_configs)
        self._optimizer, self._opt_name = self._init_optimizer(optimizer_configs, loss_configs)
        self._epsilon = epsilon
    
    def _init_optimizer(self, optimizer_configs, loss_configs):
        name = optimizer_configs["name"].lower()
        args = optimizer_configs["args"]
        if name == "adam":
            opt = partial(Adam, **args)
        elif name == "nlopt":
            opt = QPOptimizer(**args, **loss_configs)
        return opt, name
    
    def _run_model(self, beta, tree_tensor):
        model = SpinItModel(beta.shape[0]).float()
        opt = self._optimizer(model.parameters())
        iterations = 10 # TODO: change
        model.train()
        for _ in range(iterations):
            opt.zero_grad()
            output = model(beta, tree_tensor) #.cuda()
            loss = self._loss(output, tree_tensor)
            loss.backward()
            opt.step()
    
    def run(self, mesh_obj: MeshObj):
        tree_tensor = self._octree_obj.build_from_mesh(mesh_obj)
        tree_tensor = OctreeTensorHandler.set_beta(tree_tensor)
        
        for i in range(10):
            print(f'split_iter: {i}. num_of_cells = {tree_tensor.shape[0]}')
            tree_tensor = OctreeTensorHandler.calc_s_vector(tree_tensor, mesh_obj.rho)
            internal_beta = OctreeTensorHandler.get_internal_beta(tree_tensor).float()
            
            if self._opt_name == "adam":
                self._run_model(internal_beta, tree_tensor)
            elif self._opt_name == "nlopt":
                optimal_beta = self._optimizer(internal_beta, tree_tensor)
            else:
                raise NotImplementedError()
            
            optimal_beta[optimal_beta > 1 - self._epsilon] = 1.
            optimal_beta[optimal_beta < self._epsilon] = 0.
            tree_tensor = OctreeTensorHandler.set_internal_beta(tree_tensor, optimal_beta)

            #split cells with beta inside (eps, 1-eps)
            octree_external = OctreeTensorHandler.get_exterior(tree_tensor)
            octree_boundary = OctreeTensorHandler.get_boundary(tree_tensor)

            to_split = ~((optimal_beta==0.) | (optimal_beta==1.))
            octree_internal_no_split = OctreeTensorHandler.get_interior(tree_tensor)[~to_split]
            octree_to_split = OctreeTensorHandler.get_interior(tree_tensor)[to_split]
            splitted_tree = Octree.create_leaves_tensor(2, None, octree_to_split)
            
            tree_tensor = torch.vstack((octree_external, octree_boundary, octree_internal_no_split, splitted_tree))

            print(f'finished iter {i}. num_of_cells = {tree_tensor.shape[0]}')


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as file:
        configs = yaml.safe_load(file)
    
    mesh_obj = MeshObj(**configs.pop("object"))
    spin_it = SpinIt(**configs) #["octree"], configs["optimizer"], configs["loss"], configs["epsilon"], confi)
    new_obj = spin_it.run(mesh_obj)
    
    output_path = args.file_path # TODO: change to new path
    # save_obj(new_obj, output_path)
