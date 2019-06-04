import unittest

import matplotlib.pyplot as plt
import numpy as np

import NNetWork.Layerfunction as F
import NNetWork.Layers as nn
import NNetWork.Variables as uvar  # piline : ignore bulid in function
from NNetWork.LossFunc import svmLoss
from NNetWork.Util import *

class TestLayers(unittest.TestCase):
		
	def averagePool(self):
		x = uvar.RandnVar((1,1,4,4), 10)
		mp = nn.AveragePool((2,2),stride=2)
		k = mp.forward(x)
		print(x)
		print(k)

	def tinyFC(self):
		x = np.random.randn(3,4)
		tfc = nn.TinyAfineLayer()
		y = tfc.forward(x)
		print(y)

	def Serial(self):
		x = np.random.randn(3,3)
		model = nn.Serial(
				nn.Linear(3,5),
				nn.ReLU(),
				nn.Linear(5,2)
			)

		y = np.random.randint(0,2,3)

		out = model.forward(uvar.BaseVar(data=x))
		loss = svmLoss(out, uvar.BaseVar(data=y))
		print("loss is: ", loss)
		model.backward(out)

	def reshape(self):
		x = uvar.RandnVar((3,4,5))
		tfc = nn.ReshapeLayer((3,-1))
		y = tfc.forward(x)
		print(y.data.shape)
		y.grad = y.data
		bac = tfc.backward(y)
		print(bac.grad.shape)
		
	def Conv2d(self):
		x_shape = (2, 3, 4, 4)
		w_shape = (3, 3, 4, 4)
		x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
		w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
		b = np.linspace(-0.1, 0.2, num=3)

		tcv = nn.Conv2d(3,3,(4,4),2,1)
		tcv.w = uvar.BaseVar(w)
		tcv.b = uvar.BaseVar(b)
		out = tcv.forward(uvar.BaseVar(x))
		correct_out = np.array([[[[-0.08759809, -0.10987781],
								[-0.18387192, -0.2109216 ]],
								[[ 0.21027089,  0.21661097],
								[ 0.22847626,  0.23004637]],
								[[ 0.50813986,  0.54309974],
								[ 0.64082444,  0.67101435]]],
								[[[-0.98053589, -1.03143541],
								[-1.19128892, -1.24695841]],
								[[ 0.69108355,  0.66880383],
								[ 0.59480972,  0.56776003]],
								[[ 2.36270298,  2.36904306],
								[ 2.38090835,  2.38247847]]]])

		# Compare your output to ours; difference should be around 2e-8
		print('Testing conv_forward_naive')
		print('difference: ', rel_error(out.data, correct_out))

		print('Testing conv backward')
		fx = lambda x: tcv.forward(uvar.BaseVar(x)).data
		def fw(w):
			tcv.w.data = w
			return tcv.forward(uvar.BaseVar(x)).data

		def fb(b):
			tcv.b.data = b
			return tcv.forward(uvar.BaseVar(x)).data

		tout = np.ones_like(out.data)
		gdx = eval_numerical_gradient_array(fx, x, tout)

		gdw = eval_numerical_gradient_array(fw, w, tout)
		tcv.w.data = w
		gdb = eval_numerical_gradient_array(fb, b, tout)
		tcv.b.data = b

		dx = tcv.backward(uvar.BaseVar(None, tout)).grad
		dw = tcv.w.grad
		db = tcv.b.grad
		
		print("diff x", eval_diff(dx, gdx))
		print("diff w", eval_diff(dw, gdw))
		print("diff b", eval_diff(db, gdb))

	def Conv2d_Backword(self):
		x = np.random.randn(2,3,9,9)
		tcv = nn.Conv2d(3,3,(3,3))
		w = tcv.w.data.copy()
		b = tcv.b.data.copy()
		print("Computing forward")
		X = uvar.BaseVar(x)
		out = tcv.forward(X)
		print("conv out", out.shape)
		print('Testing conv backward')
		fx = lambda x: tcv.forward(uvar.BaseVar(x)).data
		def fw(w):
			tcv.w.data = w
			return tcv.forward(X).data

		def fb(b):
			tcv.b.data = b
			return tcv.forward(X).data

		tout = np.ones_like(out.data)
		gdx = eval_numerical_gradient_array(fx, x, tout)

		gdw = eval_numerical_gradient_array(fw, w, tout)
		tcv.w.data = w
		gdb = eval_numerical_gradient_array(fb, b, tout)
		tcv.b.data = b

		dx = tcv.backward(uvar.BaseVar(None, tout)).grad
		dw = tcv.w.grad
		db = tcv.b.grad
		
		print("diff x", eval_diff(dx, gdx))
		print("diff w", eval_diff(dw, gdw))
		print("diff b", eval_diff(db, gdb))

	def SaveAndLoad(self):
		model = nn.Serial(
			nn.Linear(20,20),
			nn.ReLU(),
			nn.Linear(20,2)
		)
		model.save("test.model")
		k = model.nns
		model = nn.Serial(
			nn.Linear(20,20),
			nn.ReLU(),
			nn.Linear(20,2)
		)
		model.load("test.model")
		for o,n in zip(model.nns, k):
			if None in [o,n, o.parameters(), n.parameters()]:
				continue
			for ot, nt in zip(o.parameters(), n.parameters()):
				if None in [ot,nt]: continue
				print(rel_error(ot.data, nt.data))

	def LeakyReLU(self):
		x = np.linspace(-10,10,200)
		model = nn.LeakyReLU(0.03)
		out = model.forward(uvar.BaseVar(x))
		# plt.plot(x,out.data)
		# plt.show()
		tout = np.ones_like(out.data)
		fx = lambda x: model.forward(uvar.BaseVar(x)).data
		dout = eval_numerical_gradient_array(fx,x, tout)
		dbac = model.backward(uvar.BaseVar(None, tout))
		print("diff of grad", eval_diff(dout, dbac.grad))

	def Sigmoid(self):
		x = np.linspace(-20, 20, 4000).reshape(20, 20, 10)
		model = nn.Sigmoid()
		out = model.forward(uvar.BaseVar(x))
		# plt.plot(x,out.data)
		# plt.show()
		tout = np.ones_like(out.data)
		fx = lambda x: model.forward(uvar.BaseVar(x)).data
		dout = eval_numerical_gradient_array(fx,x, tout)
		dbac = model.backward(uvar.BaseVar(None, tout))
		print("diff of grad", eval_diff(dout, dbac.grad))

	def tahh(self):
		x = np.linspace(-20, 20, 4000).reshape(20, 20, 10)
		model = nn.Tanh()
		out = model.forward(uvar.BaseVar(x))
		# print(out)
		# plt.plot(x,out.data)
		# plt.show()
		tout = np.ones_like(out.data)
		fx = lambda x: model.forward(uvar.BaseVar(x)).data
		dout = eval_numerical_gradient_array(fx,x, tout)
		dbac = model.backward(uvar.BaseVar(None, tout))
		print("diff of grad", eval_diff(dout, dbac.grad))

	def Tanhshrink(self):
		x = np.linspace(-20, 20, 4000).reshape(20, 20, 10)
		model = nn.Tanhshrink()
		out = model.forward(uvar.BaseVar(x))
		# print(out)
		# plt.plot(x,out.data)
		# plt.show()
		tout = np.ones_like(out.data)
		fx = lambda x: model.forward(uvar.BaseVar(x)).data
		dout = eval_numerical_gradient_array(fx,x, tout)
		dbac = model.backward(uvar.BaseVar(None, tout))
		print("diff of grad", eval_diff(dout, dbac.grad))

	def RNNCell(self):
		N, T, D, H = 2, 3, 4, 5
		x = np.linspace(-0.1, 0.3, num=N*T*D).reshape(N, T, D)
		h0 = np.linspace(-0.3, 0.1, num=N*H).reshape(N, H)
		Wx = np.linspace(-0.2, 0.4, num=D*H).reshape(D, H)
		Wh = np.linspace(-0.4, 0.1, num=H*H).reshape(H, H)
		b = np.linspace(-0.7, 0.1, num=H).reshape(1,H)

		rnn = nn.RNNCell(D, H, H, activation=nn.Tanh())
		assert rnn.wh.data.shape == Wh.shape, "rnn wh shape is {0} while wh.shape is {1}".format(rnn.wh.data.shape, Wh.shape)
		assert rnn.wx.data.shape == Wx.shape, "rnn wx shape is {0} while wx.shape is {1}".format(rnn.wx.data.shape, Wx.shape)
		assert rnn.bh.data.shape == b.shape, "rnn b shape is {0} while b.shape is {1}".format(rnn.bh.data.shape, b.shape)

		rnn.wh = uvar.BaseVar(Wh)
		rnn.wx = uvar.BaseVar(Wx)
		rnn.bh = uvar.BaseVar(b)

		h = []
		ht = uvar.BaseVar(h0)
		for xt in x.transpose(1,0,2):
			ht = rnn(xt, ht)
			h.append(ht.data)
		h = np.array(h).transpose(1,0,2)
		expected_h = np.asarray([
		[
			[-0.42070749, -0.27279261, -0.11074945,  0.05740409,  0.22236251],
			[-0.39525808, -0.22554661, -0.0409454,   0.14649412,  0.32397316],
			[-0.42305111, -0.24223728, -0.04287027,  0.15997045,  0.35014525],
		],
		[
			[-0.55857474, -0.39065825, -0.19198182,  0.02378408,  0.23735671],
			[-0.27150199, -0.07088804,  0.13562939,  0.33099728,  0.50158768],
			[-0.51014825, -0.30524429, -0.06755202,  0.17806392,  0.40333043]]])
		print('h error: ', rel_error(expected_h, h))

	def RNNCell_Backward(self):
		N, T, D, H = 2, 3, 4, 5
		rnn = nn.RNNCell(D, H, H, activation=nn.Tanh())
		x = np.random.randn(N, D)
		X = uvar.BaseVar(x)
		h = np.random.randn(N, H)
		H = uvar.BaseVar(h)
		def fattr(attr, value):
			getattr(rnn,attr).data = value
			return rnn(X,H)
		fdxx = lambda x: rnn(uvar.BaseVar(x), H).data
		fdwx = lambda wx: fattr('wx', wx).data
		fdhh = lambda h: rnn(X, uvar.BaseVar(h)).data
		fdwh = lambda wh: fattr('wh', wh).data
		fdbh = lambda bh: fattr('bh', bh).data
		dout = np.random.randn(*rnn(X,H).shape)
		wx, wh, bh = rnn.wx.data, rnn.wh.data, rnn.bh.data
		gfdxx = eval_numerical_gradient_array(fdxx, x, dout)
		gfdwx = eval_numerical_gradient_array(fdwx, wx, dout)
		gfdhh = eval_numerical_gradient_array(fdhh, h, dout)
		gfdwh = eval_numerical_gradient_array(fdwh, wh, dout)
		gfdbh = eval_numerical_gradient_array(fdbh, bh, dout)

		rnn.backward(uvar.BaseVar(None,dout))

		nfdxx = rnn.x.grad
		nfdwx = rnn.wx.grad
		nfdhh = rnn.h.grad
		nfdwh = rnn.wh.grad
		nfdbh = rnn.bh.grad

		print("nfdxx and gfdxx", eval_diff(nfdxx, gfdxx))
		print("nfdwx and gfdwx", eval_diff(nfdwx, gfdwx))
		print("nfdhh and gfdhh", eval_diff(nfdhh, gfdhh))
		print("nfdwh and gfdwh", eval_diff(nfdwh, gfdwh))
		print("nfdbh and gfdbh", eval_diff(nfdbh, gfdbh))
		

	def linear(self):
		X = np.linspace(-10,10,100).reshape(10,10)
		W = np.linspace(-5,5, 50).reshape(10,5)
		B = np.linspace(-2,2, 5).reshape(1,5)
		tout = np.array([[ 52.1125,  34.5597,  17.0068,  -0.5461, -18.0990],
			[ 43.8668,  30.4368,  17.0068,   3.5768,  -9.8532],
			[ 35.6211,  26.3140,  17.0068,   7.6996,  -1.6075],
			[ 27.3754,  22.1911,  17.0068,  11.8225,   6.6382],
			[ 19.1297,  18.0682,  17.0068,  15.9454,  14.8839],
			[ 10.8839,  13.9454,  17.0068,  20.0682,  23.1297],
			[  2.6382,   9.8225,  17.0068,  24.1911,  31.3754],
			[ -5.6075,   5.6996,  17.0068,  28.3140,  39.6211],
			[-13.8532,   1.5768,  17.0068,  32.4368,  47.8668],
			[-22.0989,  -2.5461,  17.0068,  36.5597,  56.1126]])
		lin = nn.Linear(10,5)
		lin.w.data = W
		lin.b.data = B
		out = lin(X)

		self.assertLessEqual(rel_error(out.data, tout), 12e-5)

		dout = np.random.randn(*out.data.shape)
		fx = lambda x: lin(uvar.BaseVar(x)).data
		fw = lambda w: _magic_attr(lin, 'w', X, w)
		fb = lambda b: _magic_attr(lin, 'b', X, b)

		ndx = eval_numerical_gradient_array(fx, X, dout)
		ndw = eval_numerical_gradient_array(fw, W, dout)
		ndb = eval_numerical_gradient_array(fb, B, dout)
		
		lin.w.data = W
		lin.b.data = B
		lin.backward(uvar.BaseVar(None,dout))

		self.assertLessEqual(rel_error(ndx, lin.x.grad), 1e-6)
		self.assertLessEqual(rel_error(ndw, lin.w.grad), 1e-6)
		self.assertLessEqual(rel_error(ndb, lin.b.grad), 1e-6)


	def one_hot(self):
		N = 5
		D = np.random.randint(0,10,N)
		out = F.oneHot(D,10).data
		P = True
		for i in range(N):
			if out[i,D[i]] != 1:
				P = False
				break
		self.assertTrue(P)
		self.assertRaises(AssertionError, F.oneHot, D, 2)


def _magic_attr(nnlayer, attrib, npx, npdata):
	getattr(nnlayer, attrib).data = npdata
	return nnlayer(uvar.BaseVar(npx)).data
