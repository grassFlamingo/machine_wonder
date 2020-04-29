import unittest

import torch

import capsule.capsuleEM as cap

class TestCapsuleEM(unittest.TestCase):
    def test_em(self):
        a = torch.rand(4, 1)
        V = torch.rand(2, 4, 3)

        cap.EM_routing(a, V, 3, 0.2, 0.2, 1.0, 2)


        