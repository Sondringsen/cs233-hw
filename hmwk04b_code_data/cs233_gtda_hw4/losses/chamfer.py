import torch

def chamfer_loss(pc_a, pc_b):
    """ Compute the chamfer loss for batched pointclouds.
    :param pc_a: torch.Tensor B x Na-points per point-cloud x 3
    :param pc_b: torch.Tensor B x Nb-points per point-cloud x 3
    :return: B floats, indicating the chamfer distances
    """
    dist = torch.cdist(pc_a, pc_b, p=2) ** 2  # B x Na x Nb
    dist_a = torch.min(dist, dim=2)[0]  # B x Na
    dist_b = torch.min(dist, dim=1)[0]  # B x Nb
    dist = torch.mean(dist_a, dim=1) + torch.mean(dist_b, dim=1)  # B
    return dist