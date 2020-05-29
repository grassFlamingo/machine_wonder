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


def tensor_ring_notn(G, n, foldn=None):
    """Compute the tensor ring of G without core n

    Requires:
    - G (list[np.ndarray]): the list of core tensor
    - n (int):
    - foldn (int): `None` means do not fold the return Tensor; 
        other wise fold Result as ...
    Return: 
    - Tensor ring product without core n
    """
    leng = len(G)
    t = (n+1) % leng
    tG = G[t]
    while True:
        t = (t + 1) % leng
        if t == n:
            break
        tG = np.tensordot(tG, G[t], axes=1)

    if foldn is None:
        return tG
    else:
        tG = np.reshape(tG, (tG.shape[0], -1, tG.shape[-1]))
        tG = np.swapaxes(tG, 0, 1)
        tG = np.reshape(tG, (tG.shape[0], -1))
        return tG


def fold(X, n):
    """fold Tensor X at n

    Return:
    X in R(I_n, I_0 I_1 ... I_{n-1} I_{n+1} ... I_{N})
    """
    newaxes = [n]
    for i in range(len(X.shape)):
        if i == n:
            continue
        newaxes.append(i)

    X = np.transpose(X, newaxes)
    n0 = X.shape[0]
    return np.reshape(X, (n0, -1))


def fold_modn(X, n):
    """fold Tensor X at n

    Return:
    X in R(I_{n}, I_{n+1} I_{n+2} ... I_{N} I_{0} ... I_{n-1})
    """
    newaxes = [i for i in range(n, len(X.shape))]
    for i in range(0, n):
        newaxes.append(i)

    X = np.transpose(X, newaxes)
    n0 = X.shape[0]
    return np.reshape(X, (n0, -1))


def unfold(X, n, shape):
    """unfold Tensor X at n.

    X comes form `fold()`

    X in R(I_n, I_0 I_1 ... I_{n-1} I_{n+1} ... I_{N})
    -> 
    Y in R(I_0, I_1, ..., I_N)

    Requires:
    - X: a matrix with shape 
    - n: 
    - shape (tuple| list):
        I_0, I_1, ..., I_N
         please make sure shape[n] == X.shape[0]
    Return:
    - Tensor: unfold Tensor with shape `shape`
    """
    assert shape[n] == X.shape[0]
    newaxes = []
    oldshape = [shape[n]]
    for i, s in enumerate(shape):
        if i == n:
            newaxes.append(0)
            continue
        elif i > n:
            newaxes.append(i)
        else:
            newaxes.append(i+1)
        oldshape.append(s)

    X = np.reshape(X, oldshape)
    return np.transpose(X, newaxes)

def unfold_moden(X, n, shape):
    """unfold Tensor X at n.

    X come from `fold_moden()`

    X in R(I_{n}, I_{n+1} I_{n+2} ... I_{N} I_{0} ... I_{n-1})
    -> 
    Y in R(I_0, I_1, ..., I_N)

    Requires:
    - X: a matrix with shape 
    - n: 
    - shape (tuple| list): please make sure shape[n] == X.shape[0]
    """
    assert shape[n] == X.shape[0]

    newaxes = []
    oldshape = []
    for i in range(n, len(shape)):
        oldshape.append(shape[i])
        newaxes.append(i)

    for i in range(0, len(shape)-n):
        oldshape.append(shape[i])
        newaxes.append(i)

    X = np.reshape(X, oldshape)
    return np.transpose(X, newaxes)






def singular_value_thresholding(X, mu):
    """Compute singular value thresholding
    U max(S - mu, 0) V.T
    """
    U, S, V = np.linalg.svd(X, full_matrices=False)
    S = S - mu
    S[S < 0] = 0.0
    return np.matmul(U * S, V)
