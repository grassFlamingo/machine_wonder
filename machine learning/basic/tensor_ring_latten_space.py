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

import itertools

from tensor_ring_latten_space_lib import *

# %%
# ImageX = Image.open("./alex-tai-FKf4ixzVz_8-unsplash-square.jpg")
ImageX = Image.open("machine learning/basic/alex-tai-FKf4ixzVz_8-unsplash-square.jpg")
# print(ImageX.size)
# ImageX = ImageX.resize((2048, 2048))
ImageX = ImageX.resize((256, 256))
ImageX = np.asarray(ImageX)

# plt.imshow(ImageX)
# plt.show()
# %%

TensorX = np.reshape(ImageX, (16, 16, 16, 16, 3)) / 255.0
mask = np.random.choice([True, False], size=(16, 16, 16, 16, 1), p=[0.5, 0.5])

# plt.imshow(mask.reshape(16*16,16*16).astype(np.int8))
# plt.show()

# %%


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

    Gs = [np.random.randn(Rs[i], T.shape[i], Rs[(i+1) % numG])
          for i in range(numG)]
    Ms = [np.zeros((3, *g.shape), dtype=g.dtype) for g in Gs]
    Ys = [np.zeros((3, *g.shape), dtype=g.dtype) for g in Gs]

    lamb = 5
    mu = 1
    mumax = 1e2
    rho = 1.01
    tol = 1e-6
    k = 0
    kmax = 300

    X = T.copy()
    for k in range(kmax):
        Xlast = X.copy()

        # update Gs
        for n, g in enumerate(Gs):
            tmy = np.sum(mu * Ms[n] + Ys[n], axis=0)
            tmy = fold(tmy, 1)
            trng = tensor_ring_notn(Gs, n, 1)
            txg = lamb * np.matmul(fold_modn(X, n), trng)
            tggi = lamb * np.matmul(trng.T, trng) + 3 * \
                mu * np.eye(trng.shape[1])
            Gs[n] = unfold(
                np.matmul(tmy + txg, np.linalg.inv(tggi)), 1, g.shape)

        # update M
        for n, i in itertools.product(range(numG), range(3)):
            tm = singular_value_thresholding(fold(Gs[n] - 1/mu * Ys[n][i], i), 1/mu)
            Ms[n][i] = unfold(tm, i, Ms[n][i].shape)

        # update X
        X = T * mask + tensor_ring(Gs) * (~mask)

        # update Y
        for n, i in itertools.product(range(numG), range(3)):
            Ys[n][i] += mu * (Ms[n][i] - Gs[n])
            

        mu = max(rho * mu, mumax)
        err = np.linalg.norm(X - Xlast) / np.linalg.norm(X)
        print("epho", k, "error", err)
        if err < tol:
            break

    return X, Gs

X, Gs = tensor_ring_low_rank_factors(TensorX, mask, [2]*len(TensorX.shape))

print("Got X")
X = np.reshape(X, (16*16, 16*16, 3))
plt.imshow(X)
plt.show()
