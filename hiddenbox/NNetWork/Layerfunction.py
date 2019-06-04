import numpy as np
import NNetWork.Variables as uvar

def oneHot(vector, depth):
    """
    Input:
    - vector: a vector of indices
    - depth: number of ...
    Output:
    - one hot matrix
    """
    vd = vector.data
    assert np.max(vd) < depth, "the maximum element must smaller than depth"
    out = _oneHot(vd, depth)
    return uvar.BaseVar(out)

def _oneHot(vector,depth):
    r = vector.shape[0]
    c = depth
    out = np.zeros((r,c))
    for i in range(r):
        out[i,vector[i]] = 1
    return out
