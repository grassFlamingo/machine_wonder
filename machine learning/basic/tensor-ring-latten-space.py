r"""

Implementation of:

Tensor Ring Decomposition with Rank Minimization on Latent Space: An Efficient Approach for Tensor Completion

(AAAI-19)


$$
\min_{[G], X} \sum_{n=1}^N \sum_{i=1}^3 \| G^n_i \|_* + \lambda/2 \| X - \Phi([G])\|^2_F
\text{s.t. } P_{\Omega}(X) = P_{\Omega}(T).
$$
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image


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


# %%
ImageX = Image.open("./alex-tai-FKf4ixzVz_8-unsplash-square.jpg")
# print(ImageX.size)
ImageX = ImageX.resize((2048, 2048))
ImageX = np.asarray(ImageX)

plt.imshow(ImageX)
plt.show()
# %%

TensorX = np.reshape(ImageX, (8, 8, 32, 32, 8, 8))


def tensor_ring_low_rank_factors(T, mask, Rs):
    """Tensor ring low-rank factors(TRLRF)

    Requires:
    - T (np.ndarray): Corrupted tensor
    - mask (np.ndarray): mask in {0, 1}
    - Rs (list[np.ndarray]): list of Tensor ring rank

    Returns:
    - X (np.ndarray): computed tensor
    - G (list[np.ndarray]): TR factors
    """
    numG = len(Rs)

    Gs = [np.random.randn(Rs[i], T.shape[i], Rs[i+1 % numG])
          for i in range(numG)]
    lamb = 5
    mu0 = 1
    mumax = 1e2
    rho = 1.01
    tol = 1e-6
    k = 0
    kmax = 300

    for k in range(kmax):
        pass

