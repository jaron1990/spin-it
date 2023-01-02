import trimesh


def load_stl(stl_path):
    return trimesh.load(stl_path)


# def save_stl(obj, stl_path):
#     trimesh.