import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##############################################
# NOTICE: This file is not official version.
# There are some bugs here.
# I should read the paper more carefully.
##############################################

def EM_routing(a: torch.Tensor, V: torch.Tensor, layer: int, betaa: float, betau: float, lamb: float, iterations: int = 5):
    """
    V^{h}_{ij} is the h dimension of vote from 
    capsule i with activation a_i in Layer L to
    capsule j in layer L+1

    - a (I, 1): activations
    - V (H, I, J): votes
    """
    H, I, J = V.shape
    R = torch.ones((I, J), dtype=V.dtype) / (layer+1)
    twopi = np.pi * 2
    for t in range(iterations):
        # M_Step
        R = R * a
        R1 = torch.reshape(R, (1, I, J))
        sumRi = torch.sum(R, 0, keepdim=True)
        mu = torch.sum(R1 * V, 1, keepdim=True) / sumRi.reshape(1, 1, J)
        sigma = torch.sum(R1*(V - mu)**2, 1) / sumRi
        cost = (betau + 0.5*torch.log(sigma)) * sumRi
        aj = logistic(lamb*(betaa - torch.sum(cost, 0)))
        # E_Step
        gaupj = (V-mu)**2 / (2*sigma.reshape(H, 1, J))
        pj = torch.exp(-torch.sum(gaupj, 0))
        pj /= torch.sqrt(twopi*torch.prod(sigma))
        R = a * pj
        R = R / torch.sum(R)

    return a


def logistic(x):
    return 1 / (1 + torch.exp(-1*x))
