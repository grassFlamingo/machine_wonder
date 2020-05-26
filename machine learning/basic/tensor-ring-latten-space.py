r"""

Implementation of:

Tensor Ring Decomposition with Rank Minimization on Latent Space: An Efficient Approach for Tensor Completion

(AAAI-19)


$$
\min_{[G], X} \sum_{n=1}^N \sum_{i=1}^3 \| G^n_i \|_* + \lambda/2 \| X - \Phi([G])\|^2_F
\text{s.t. } P_{\Omega}(X) = P_{\Omega}(T).
$$
"""

import numpy as np
import matplotlib.pyplot as plt

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
    

a, b = np.random.randn(2,2,3,2)
print(a)
print(b)
ab = np.tensordot(a,b, axes=1)
print(np.einsum("i...i", ab))

print(tensor_ring([a,b]))



