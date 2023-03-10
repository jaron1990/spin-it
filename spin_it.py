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
from utils import OctreeTensorMapping, Location



def parse_args():
    parser = argparse.ArgumentParser(prog='Spin-it')
    parser.add_argument('-c', '--config', required=True, type=str, dest='config', help='Configurations file path')
    return parser.parse_args()


class SpinIt:
    def __init__(self, octree_configs, optimizer_configs, loss_configs, epsilon, num_of_iters) -> None:
        self._octree_obj = Octree(**octree_configs)
        self._loss = SpinItLoss(**loss_configs)
        self._optimizer, self._opt_name = self._init_optimizer(optimizer_configs, loss_configs)
        self._epsilon = epsilon
        self._num_of_iters = num_of_iters
    
    def _init_optimizer(self, optimizer_configs, loss_configs):
        name = optimizer_configs["name"].lower()
        args = optimizer_configs["args"]
        if name == "adam":
            opt = partial(Adam, **args)
        elif name == "nlopt":
            loss_configs.pop("constraints_weights")
            opt = QPOptimizer(**args, **loss_configs)
        return opt, name
    
    def _run_model(self, stable_beta_mask, unstable_beta_mask, tree_tensor):
        beta = OctreeTensorHandler.get_beta(tree_tensor)[unstable_beta_mask].double()
        model = SpinItModel(beta)
        opt = self._optimizer(model.parameters())
        iterations = 1000 # TODO: change
        model.train()
        for i in range(iterations):
            opt.zero_grad()
            s_unstable = tree_tensor[unstable_beta_mask,OctreeTensorMapping.S_1:OctreeTensorMapping.S_ZZ+1]
            cells_boundary = tree_tensor[tree_tensor[:,OctreeTensorMapping.LOC]==Location.BOUNDARY]
            s_boundary = cells_boundary[:,OctreeTensorMapping.S_1:OctreeTensorMapping.S_ZZ+1]
            s_stable = tree_tensor[stable_beta_mask,OctreeTensorMapping.S_1:OctreeTensorMapping.S_ZZ+1]
            beta_stable = tree_tensor[stable_beta_mask,OctreeTensorMapping.BETA]
            s_stable_total = (s_stable.T)*beta_stable
            if s_stable_total.shape[1]==0:
                s_stable_total = torch.zeros((1,10))
            s_rest = s_stable_total.sum(axis=1) + s_boundary.sum(axis=0)
            output = model(s_unstable, s_rest) #.cuda()
            loss = self._loss(output, model._beta)
            if (model._beta>1.).sum() + (model._beta<0.).sum() > 0:
                break
            loss.backward()
            opt.step()
            print(f"iter {i}: loss={loss.item()}, \nbeta[:10]={model._beta[:10]}")
            # print(f'finished iteration {i}/{iterations}')
        return model._beta.detach()
    
    def run(self, mesh_obj: MeshObj):
        tree_tensor = self._octree_obj.build_from_mesh(mesh_obj)
        tree_tensor = OctreeTensorHandler.set_beta(tree_tensor)
        OctreeTensorHandler.plot_slices(tree_tensor, iteration=-1)
        
        for i in range(self._num_of_iters):
            print(f'split_iter: {i}. num_of_cells = {tree_tensor.shape[0]}')
            tree_tensor = OctreeTensorHandler.calc_s_vector(tree_tensor, mesh_obj.rho)
            # internal_beta = OctreeTensorHandler.get_internal_beta(tree_tensor).float()
            stable_beta_mask, unstable_beta_mask = OctreeTensorHandler.get_interior_beta_mask(
                tree_tensor, self._epsilon)
            
            if unstable_beta_mask.sum()==0:
                print('all betas are stable')
                break
            if self._opt_name == "adam":
                optimal_beta = self._run_model(stable_beta_mask, unstable_beta_mask, tree_tensor)
            elif self._opt_name == "nlopt":

                optimal_beta = self._optimizer(unstable_beta_mask, tree_tensor) # TODO: FIX
            else:
                raise NotImplementedError()
            
            optimal_beta[optimal_beta > 1 - self._epsilon] = 1.
            optimal_beta[optimal_beta < self._epsilon] = 0.
            tree_tensor = OctreeTensorHandler.set_beta(tree_tensor, unstable_beta_mask, optimal_beta)

            OctreeTensorHandler.plot_slices(tree_tensor, iteration=i)


            #split cells with beta inside (eps, 1-eps)
            octree_external = OctreeTensorHandler.get_exterior(tree_tensor)
            octree_boundary = OctreeTensorHandler.get_boundary(tree_tensor)

            # to_split = ~((optimal_beta==0.) | (optimal_beta==1.))
            stable_beta_mask, unstable_beta_mask = OctreeTensorHandler.get_interior_beta_mask(tree_tensor, self._epsilon)
            splitted_tree = Octree.create_leaves_tensor(2, None, tree_tensor[unstable_beta_mask])
            
            tree_tensor = torch.vstack((octree_external, octree_boundary, tree_tensor[stable_beta_mask], splitted_tree))

            print(f'finished iter {i}. num_of_cells = {tree_tensor.shape[0]}')

        mesh = OctreeTensorHandler.create_mesh(tree_tensor)
        return mesh

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as file:
        configs = yaml.safe_load(file)
    
    mesh_obj = MeshObj(**configs.pop("object"))
    spin_it = SpinIt(**configs) #["octree"], configs["optimizer"], configs["loss"], configs["epsilon"], confi)
    new_obj_mesh = spin_it.run(mesh_obj)

    # new_obj_mesh.apply_scale(mesh_obj.scale)
    new_obj_mesh.export('output.stl')
