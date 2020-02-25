# %%
import numpy as np

from gradient_tools import *

# %%


def row_sum_one(V: np.ndarray, grad=False):
    """
    $$
    C = \sum_{ij} \left(\dfrac{v_{ij}}{\sum_k v_{ik}}\right)^2
    $$
    $$
    \dfrac{\partial C}{\partial v_{ij}} 
    = 2 \dfrac{v_{ij}}{(\sum_k v_{ik})^2} 
    - 2 \dfrac{\sum_k v_{ik} v_{ik}}{(\sum_j v_{ij})^3}
    $$
    """
    rsum = np.sum(V, 1, keepdims=True)
    vnorm = V / rsum

    loss = np.square(vnorm).sum()
    if not grad:
        return loss

    dV = V*rsum - np.sum(V*V, 1, keepdims=True)
    dV /= rsum**3
    return loss, 2*dV,  vnorm


# %%
ranV = np.random.rand(3, 3)
rsum = np.sum(ranV, 1, keepdims=True)

los, dV, vnorm = row_sum_one(ranV, True)

ndV = eval_numerical_gradient(row_sum_one, ranV)

print("E dV : ", np.linalg.norm(ndV - dV))

print(dV)
print(ndV)


# %%
