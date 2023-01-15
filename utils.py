import torch


class Location:
    UNKNOWN = 0
    INSIDE = 1
    OUTSIDE = 2
    BOUNDARY = 3


class Constraints:
    X = 0
    Y = 1
    XZ = 2
    YZ = 3
    EQ = 4


class OctreeTensorMapping:
    LVL = 0
    BBOX_X0 = 1
    BBOX_Y0 = 2
    BBOX_Z0 = 3
    BBOX_X1 = 4
    BBOX_Y1 = 5
    BBOX_Z1 = 6
    LOC = 7
    BETA = 8
    S_1 = 9
    S_X = 10
    S_Y = 11
    S_Z = 12
    S_XY = 13
    S_XZ = 14
    S_YZ = 15
    S_XX = 16
    S_YY = 17
    S_ZZ = 18


class SVector:
    ONE = OctreeTensorMapping.S_1 - OctreeTensorMapping.S_1
    X = OctreeTensorMapping.S_X - OctreeTensorMapping.S_1
    Y = OctreeTensorMapping.S_Y - OctreeTensorMapping.S_1
    Z = OctreeTensorMapping.S_Z - OctreeTensorMapping.S_1
    XY = OctreeTensorMapping.S_XY - OctreeTensorMapping.S_1
    XZ = OctreeTensorMapping.S_XZ - OctreeTensorMapping.S_1
    YZ = OctreeTensorMapping.S_YZ - OctreeTensorMapping.S_1
    XX = OctreeTensorMapping.S_XX - OctreeTensorMapping.S_1
    YY = OctreeTensorMapping.S_YY - OctreeTensorMapping.S_1
    ZZ = OctreeTensorMapping.S_ZZ - OctreeTensorMapping.S_1


def is_vertex_in_bbox(vertices: torch.Tensor, cell_start: torch.Tensor, cell_end: torch.Tensor):
    vertices = vertices.unsqueeze(1)
    log_and = torch.bitwise_and(vertices > cell_start[None], vertices < cell_end[None])
    return torch.bitwise_and(torch.bitwise_and(log_and[..., 0], log_and[..., 1]), log_and[..., 2]).any(axis=0)
