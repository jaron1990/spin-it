import trimesh
from trimesh import Trimesh


class MeshObj:
    def __init__(self, mesh_path: str, roh:float) -> None:
        self._mesh = trimesh.load(mesh_path)
        self._roh = roh
    
    @property
    def roh(self) -> float:
        return self._roh
    
    @property
    def mesh(self) -> Trimesh:
        return self._mesh