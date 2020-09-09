import os
import sys
sys.path.append(os.path.join(os.getcwd(), "hiddenbox"))

print("system paths")
print("\n".join(sys.path))

##############################################################
import SNet.Optim as optim
import SNet.Layers as layers
from SNet.Variables import SimpleVariable
from SNet.toolkit import *
import unittest
import numpy as np



class TestSVMTool(unittest.TestCase):

    def test_simple_linear(self):
        X = SimpleVariable(np.random.rand(8, 5))
        df = SimpleVariable(None, np.random.rand(8, 2))

        lin = layers.Linear(5, 2)

        slw, _rslw = invoke_simple_layer_member(lin, 'W', X)
        slb, _rslb = invoke_simple_layer_member(lin, 'b', X)
        slx = invoke_simple_layer_output(lin)

        out = lin.forward(X)

        dx = lin.backward(df).grad
        dw = lin.W.grad
        db = lin.b.grad

        ndx = eval_numerical_gradient_array(slx, X.data.copy(), df.grad)

        ndw = eval_numerical_gradient_array(slw, lin.W.data.copy(), df.grad)
        _rslw()
        ndb = eval_numerical_gradient_array(slb, lin.b.data.copy(), df.grad)
        _rslb()

        self.assertLessEqual(np.linalg.norm(dx - ndx), 1e-8)
        self.assertLessEqual(np.linalg.norm(dw - ndw), 1e-8)
        self.assertLessEqual(np.linalg.norm(db - ndb), 1e-8)

    def test_simple_layer_backwards(self):
        X = SimpleVariable(np.random.rand(10))
        df = SimpleVariable(None, np.random.randn(10))

        xlayers = [
            layers.Sigmoid(),
            layers.Tanh(),
            layers.LeakyReLU(0.2, True),
            layers.LeakyReLU(0.2, False),
            layers.ReLU6(True),
            layers.ReLU6(False),
            layers.Sin(),
            layers.Cos(),
        ]

        for xl in xlayers:
            X.grad = 0
            xl.forward(X)
            dx = xl.backward(df).grad
            ndx = eval_numerical_gradient_array(
                invoke_simple_layer_output(xl),
                X.data,
                df.grad
            )
            self.assertLessEqual(np.linalg.norm(dx - ndx), 1e-8)

    def test_simple_layer_forwards(self):
        xlayers = [
            [layers.ReLU(True), [-2, -1, 0, 1, 2], [0, 0, 0, 1, 2]],
            [layers.ReLU6(True), [-2, -1, 0, 1, 2,
                                  7, 8], [0, 0, 0, 1, 2, 0, 0]],
        ]

        for layer, inp, oup in xlayers:
            x = SimpleVariable(np.asarray(inp))
            o = np.asarray(oup)

            lo = layer(x).data

            self.assertLessEqual(np.linalg.norm(lo - o), 1e-9)

    def test_l2_regulizer(self):
        params = [
            SimpleVariable.randn(4, 4),
            SimpleVariable.randn(5, 3),
        ]

        l2reg = layers.L2Regulizer(params, 1e-1)

        l2reg.forward()
        l2reg.backward()

        fpx = [invoke_simple_layer_member(
            l2reg, "parm%03d" % i) for i in range(len(params))]

        for p, (fl, fr) in zip(params, fpx):
            ndw = eval_numerical_gradient(fl, p.data.copy())
            fr()
            self.assertLessEqual(np.linalg.norm(ndw - p.grad), 1e-8)

    def test_simple_sequences(self):
        X = SimpleVariable(np.random.rand(10, 3))
        df = SimpleVariable(None, np.random.randn(10, 3))

        layer = layers.Sequences(
            layers.Linear(3, 3),
            layers.Sigmoid(),
            layers.Linear(3, 3),
            layers.ReLU(),
            layers.Sequences(
                layers.Linear(3, 3),
                layers.ReLU(),
            )
        )

        layer(X)

        dx = layer.backward(df).grad

        cout = invoke_simple_layer_output(layer)

        ndx = eval_numerical_gradient_array(cout, X.data.copy(), df.grad)

        self.assertLessEqual(np.linalg.norm(dx - ndx), 1e-8)

    def test_simple_loss(self):
        X = SimpleVariable(np.random.rand(10, 3))
        y = SimpleVariable(np.random.randn(10, 3))

        print(X)

        sloss = layers.SimpleLoss()
        slos2 = layers.LossL2()

        dsl = invoke_simple_loss_output(sloss, y)
        dsl2 = invoke_simple_loss_output(slos2, y)

        opt = optim.SimpleOptimizer([X, y])

        opt.zero_grad()
        sloss(X, y)
        ds = sloss.backward().grad

        opt.zero_grad()
        slos2(X, y)
        ds2 = slos2.backward().grad

        nds = eval_numerical_gradient(dsl, X.data)
        nds2 = eval_numerical_gradient(dsl2, X.data)

        self.assertLessEqual(np.linalg.norm(ds - nds), 1e-8)
        self.assertLessEqual(np.linalg.norm(ds2 - nds2), 1e-8)

    def test_KL_loss(self):
        P = SimpleVariable.rand(10, 2)
        Q = SimpleVariable.rand(10, 2)

        kllos = layers.LossKL()
        loss = kllos(Q, P)
        opt = optim.SimpleOptimizer([P, Q])
        self.assertGreaterEqual(loss.data, 0)
        opt.zero_grad()
        dQ = kllos.backward().grad

        ndQ = eval_numerical_gradient(
            invoke_simple_loss_output(kllos, P),
            Q.data
        )

        self.assertLessEqual(np.linalg.norm(dQ - ndQ), 2e-7)

    def test_simple_momentum(self):
        X = SimpleVariable(np.random.rand(10, 3))
        y = SimpleVariable(np.random.randn(10, 3))

        layer = layers.Sequences(
            layers.Linear(3, 3),
            layers.Sigmoid(),
            layers.Linear(3, 3),
            layers.ReLU(),
            layers.Sequences(
                layers.Linear(3, 3),
                layers.ReLU(),
            )
        )

        params = layer.parameters()

        self.assertIn(layer.sequence[0].W, params)
        self.assertIn(layer.sequence[4].sequence[0].W, params)

        opt = optim.SimpleOptimizer(params, learning_rate=1e-3)

        sloss = layers.SimpleLoss()
        realoss = sloss(layer(X), y).data

        opt.zero_grad()
        dsloss = sloss.backward()
        layer.backward(dsloss)

        opt.step()

        self.assertLess(sloss(layer(X), y).data, realoss)
