import torch


class Location:
    UNKNOWN = 0
    INSIDE = 1
    OUTSIDE = 2
    BOUNDARY = 3


def is_vertex_in_bbox(vertices: torch.Tensor, cell_start: torch.Tensor, cell_end: torch.Tensor):
    vertices = vertices.unsqueeze(1)
    log_and = torch.bitwise_and(vertices > cell_start[None], vertices < cell_end[None])
    return torch.bitwise_and(torch.bitwise_and(log_and[..., 0], log_and[..., 1]), log_and[..., 2]).any(axis=0)
