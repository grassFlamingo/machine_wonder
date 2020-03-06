"""
[1] Daniel D. Lee, H. Sebastian Seung. Algorithms for Non-negative Matrix Factorization

X ~ W H

X_{ij} >= 0
W_{ij} >= 0
H_{ij} >= 0
"""

import numpy as np


X = np.random.rand(8, 8)
W = np.random.rand(8, 7)
H = np.random.rand(7, 8)

# Euclidean distance
for i in range(300):
    e = np.square(X - np.matmul(W, H)).sum()
    if e < 1e-2:
        print(f"{i :03d} loss {e :.8f}")
        print("converged")
        break
    if i % 10 == 0:
        print(f"{i :03d} loss {e :.8f}")

    H = H * np.matmul(W.T, X) / (W.T @ W @ H)
    W = W * np.matmul(X, H.T) / (W @ H @ H.T)

print("EC", np.linalg.norm(X - np.matmul(W, H)))


X = np.random.rand(8, 8)
W = np.random.rand(8, 8)
H = np.random.rand(8, 8)

# KL divergence
for i in range(300):
    B = np.matmul(W, H)
    e = np.sum(X * (np.log(X) - np.log(B)) - X + B)
    if e < 1e-2:
        print(f"{i :03d} loss {e :.8f}")
        print("converged")
        break
    if i % 10 == 0:
        print(f"{i :03d} loss {e :.8f}")
    # avoid div 0
    H = H * np.matmul(W.T, X / (B + 1e-8)) / (np.sum(W, 0) + 1e-8)
    B = np.matmul(W, H)
    W = W * np.matmul(X / (B + 1e-8), H.T) / (np.sum(H, 1) + 1e-8)

print("KL", np.linalg.norm(X - np.matmul(W, H)))
