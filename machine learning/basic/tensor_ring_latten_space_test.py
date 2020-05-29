import unittest
from tensor_ring_latten_space_lib import *


class TestTRLS(unittest.TestCase):
    def test_fold(self):
        X = np.random.randn(3, 5, 6, 4)
        foldX = fold(X, 1)
        self.assertTupleEqual(foldX.shape, (5, 3*6*4))
        nfoldx = X.transpose(1, 0, 2, 3).reshape(5, -1)
        self.assertLessEqual(np.linalg.norm(foldX - nfoldx), 1e-8)

        unfX = unfold(foldX, 1, (3, 5, 6, 4))
        self.assertLessEqual(np.linalg.norm(X - unfX), 1e-8)

        foldX = fold(X, 2)
        self.assertTupleEqual(foldX.shape, (6, 3*5*4))
        unfX = unfold(foldX, 2, (3, 5, 6, 4))
        self.assertLessEqual(np.linalg.norm(X - unfX), 1e-8)

        foldX = fold_modn(X, 2)
        self.assertTupleEqual(foldX.shape, (6, 4*3*5))
        unfX = unfold_moden(foldX, 2, (3, 5, 6, 4))
        self.assertLessEqual(np.linalg.norm(X - unfX), 1e-8)


    def test_tensor_ring(self):
        G = [
            np.random.randn(2, 3, 4),
            np.random.randn(4, 5, 6),
            np.random.randn(6, 5, 2)
        ]

        trg = tensor_ring(G)

        self.assertTupleEqual(trg.shape, (3, 5, 5))

        tG = np.matmul(G[0].reshape(-1, 4),
                       G[1].reshape(4, -1)).reshape(2, 3, 5, 6)
        tG = np.matmul(tG.reshape(-1, 6),
                       G[2].reshape(6, -1)).reshape(2, 3, 5, 5, 2)
        nrg = 0
        for i in range(tG.shape[0]):
            nrg += tG[i, Ellipsis, i]
        self.assertLessEqual(np.linalg.norm(nrg - trg), 1e-8)

        trgn = tensor_ring_notn(G, 1)
        self.assertTupleEqual(trgn.shape, (6, 5, 3, 4))
        nrgn = np.matmul(G[2].reshape(-1, 2),
                         G[0].reshape(2, -1)).reshape(6, 5, 3, 4)
        self.assertLessEqual(np.linalg.norm(nrgn - trgn), 1e-8)

        trgn = tensor_ring_notn(G, 1, 2)
        self.assertTupleEqual(trgn.shape, (3, 6*5*4))
        nrgn = fold(nrgn, 2)
        self.assertLessEqual(np.linalg.norm(nrgn - trgn), 1e-8)
