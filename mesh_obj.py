import numpy as np
import trimesh
from trimesh import Trimesh


class MeshObj:
    def __init__(self, mesh_path: str, roh:float) -> None:
        mesh = trimesh.load(mesh_path)
        self._faces = np.array(mesh.faces)
        vertices = np.array(mesh.vertices)
        self._scale = abs(vertices).max()
        self._vertices = vertices / self._scale
        self._roh = roh
    
    @property
    def rho(self) -> float:
        return self._roh
    
    @property
    def vertices(self) -> np.ndarray:
        return self._vertices
    
    @property
    def faces(self) -> np.ndarray:
        return self._faces
    
    @property
    def scale(self) -> float:
        return self._scale