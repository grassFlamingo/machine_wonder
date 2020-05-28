import numpy as np


def tensor_ring(G):
    """Compute the tensor ring of G

    Requires:
    - G (list[np.ndarray]): the list of core tensors
    """
    tG = G[0]
    for g in G[1::]:
        tG = np.tensordot(tG, g, axes=1)
    n = tG.shape[0]
    return np.sum(tG[range(n), Ellipsis, range(n)], axis=0)

def tensor_ring_notn(G, n, foldn=-1):
    """Compute the tensor ring of G without core n

    Requires:
    - G (list[np.ndarray]): the list of core tensor
    - n (int):
    - foldn (int): -1 means do not fold the return Tensor; 
        other wise fold at `fold` before return.
    Return: 
    - Tensor ring product without core n
    """
    leng = len(G)
    t = n+1;
    tG = G[t]
    while True:
        t = (t + 1) % leng
        if t == n:
            break
        tG = np.tensordot(tG, G[t], axes=1)

    if foldn == -1:
        return tG
    else:
        return fold(tG, foldn)


def fold(X, n):
    """fold Tensor X at n"""
    X = np.swapaxes(X, n, 0)
    n0 = X.shape[0]
    return np.reshape(X, (n0, -1))


def unfold(X, n, shape):
    """unfold Tensor X at n.

    Requires:
    - X: a matrix with shape 
    - n: 
    - shape (tuple| list): please make sure shape[n] == X.shape[0]
    Return:
    - Tensor: unfold Tensor with shape `shape`
    """
    assert shape[n] == X.shape[0]
    shape = list(shape)

    shape[0], shape[n] = shape[n], shape[0]
    X = np.reshape(X, shape)
    return np.swapaxes(X, 0, n)

def singular_value_thresholding(X, mu):
    """Compute singular value thresholding
    U max(S - mu, 0) V.T
    """
    U, S, VH = np.linalg.svd(X, full_matrices=False)
    S = S - mu
    S[S < 0] = 0.0
    return np.matmul(U * S, VH)
