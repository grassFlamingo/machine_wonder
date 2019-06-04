import numpy as np
import NNetWork.Variables as uvar
import pickle

try:
    from NNetWork.CnnLayers import conv2dForward, conv2dBackward
except Exception as e:
    print(e)
    print('run the following from the current directory and try again:')
    print('python setup.py build_ext --inplace')
    print('You may also need to restart your iPython kernel')


class Empty():
    def forward(self, x):
        return x

    def backward(self, dout):
        return dout

    def parameters(self):
        return [None]

    def parameterDic(self):
        return None

    def loadParameterDic(self, p):
        pass

    def __str__(self):
        return "Layer: {0}".format(self.__class__)

    def __call__(self, *x):
        return self.forward(*x)

    def save(self, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(self.parameterDic(), f)

    def load(self, fileName):
        """
        Load the params from file
        Input:
        - fileName: the file to be load
        Output: None
        """
        try:
            with open(fileName, 'rb') as f:
                params = pickle.load(f)
                self.loadParameterDic(params)
        except FileNotFoundError:
            print("File {0} Not Found".format(fileName))


class Serial(Empty):
    def __init__(self, *nns):
        self.nns = nns

    def forward(self, x):
        for nn in self.nns:
            # print("forward: ", nn)
            x = nn.forward(x)
            # print("x: ", x)
        return x

    def backward(self, dout):
        for nn in reversed(self.nns):
            # print("backward: ", nn)
            dout = nn.backward(dout)
            # print("dout: ", dout)
        return dout

    def parameters(self):
        param = []
        for nn in self.nns:
            p = nn.parameters()
            if p is None: continue
            param += p
        return param

    def parameterDic(self):
        pdic = {i: nn.parameterDic() for i, nn in enumerate(self.nns)}
        return pdic

    def loadParameterDic(self, params):
        for p, n in zip(params, self.nns):
            n.loadParameterDic(params[p])


class Linear(Empty):
    def __init__(self, W, O, alpha=1, beta=0):
        """
        - W: input width
        - O: output size
        - alpha, beta: in w initialization alpha*randn(M,O)+beta
        - self: 
            - w: (M,O) 
            - b: (O,)
        """
        self.w = uvar.RandnVar((W, O), alpha=alpha, beta=beta)
        self.b = uvar.RandnVar((1, O), alpha=alpha, beta=beta)

    def forward(self, x):
        """
        -x : a BaseVar
        """
        self.x = x
        # print("self fc", type(x))
        return uvar.BaseVar(np.dot(x.data, self.w.data) + self.b.data)

    def backward(self, dout):
        x, w, b = self.x, self.w, self.b
        dout = dout.grad
        dx = np.dot(dout, w.data.T)
        dw = np.dot(x.data.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)
        x.grad, w.grad, b.grad = dx, dw, np.reshape(db, b.data.shape)
        return x

    def parameters(self):
        return [self.w, self.b]

    def parameterDic(self):
        return {"w": self.w, "b": self.b}

    def loadParameterDic(self, p):
        w, b = p["w"], p["b"]
        assert w.shape == self.w.shape, "The Shape of w will be {0}, but now is {1}".format(
            self.w.shape, w.shape)
        assert b.shape == self.b.shape, "The Shape of b will be {0}, but now is {1}".format(
            self.b.shape, b.shape)
        self.w, self.b = w, b

    def __str__(self):
        return "Linear(%d,%d)" % self.w.data.shape


class TinyAfine(Linear):
    def __init__(self, w=None, b=None):
        """
        all w and b is a BaseVar
        """
        w = w if w != None else uvar.RandnVar((1, ), 5)
        b = b if b != None else uvar.RandnVar((1, ), 5)
        self.w = w
        self.b = b

    def forward(self, x):
        self.x = x
        out = x.data * self.w.data + self.b.data
        return uvar.BaseVar(out)

    def backward(self, dout):
        dout = dout.grad
        self.x.grad = self.w.data * dout
        self.w.grad = np.mean(self.x.data)
        self.b.grad = np.mean(dout)
        return self.x


class ReLU(Empty):
    def forward(self, x):
        """
        - x: a BaseVar
        return a BaseVar
        """
        # print("self ReLU", type(x))
        self.x = x
        return uvar.BaseVar(np.where(x.data > 0, x.data, 0))

    def backward(self, dout):
        x = self.x
        dx = np.where(x.data > 0, dout.grad, 0)
        x.grad = dx
        return x


class LeakyReLU(Empty):
    """
    $y = \alpha x, x < 0 | x, x \ge 0$
    """

    def __init__(self, alpha):
        self.alpfa = alpha

    def forward(self, x):
        self.x = x
        data = x.data
        return uvar.BaseVar(np.where(data > 0, data, self.alpfa * data))

    def backward(self, dout):
        x = self.x.data
        grad = dout.grad
        dx = np.where(x > 0, grad, self.alpfa * grad)
        self.x.grad = dx
        return self.x


class Sigmoid(Empty):
    """
    Comput $\sigma = \dfrac{1}{1 + e^{-x}}$
    """

    def forward(self, x):
        self.x = x
        out = 1 / (1 + np.exp(-x.data))
        self.out = out
        return uvar.BaseVar(out)

    def backward(self, dout):
        grad = dout.grad
        dx = self.out * (1 - self.out)
        self.x.grad = grad * dx
        return self.x


class Tanh(Empty):
    """
    tanh
    $y = dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$
    """

    def forward(self, x):
        self.x = x
        out = np.tanh(x.data)
        self.out = out
        return uvar.BaseVar(out)

    def backward(self, dout):
        dx = dout.grad * (1 - np.square(self.out))
        self.x.grad = dx
        return self.x


class Tanhshrink(Empty):
    def forward(self, x):
        """
        out = x - tanh(x)
        """
        self.x = x
        x = x.data
        bthah = np.tanh(x)
        self.btanh = bthah
        return uvar.BaseVar(x - bthah)

    def backward(self, dout):
        dx = dout.grad * np.square(self.btanh)
        self.x.grad = dx
        return self.x


class MaxPool(Empty):
    def __init__(self, kernel_size, stride=1):
        """
        - kernel_size: a tupe of (M,N)
        """
        self.HH, self.WW = kernel_size
        self.stride = stride

    def forward(self, x):
        """
        - x.data: shape of (N,C,H,W) -> (Number of imges, channels, height, width)
        """
        self.x = x
        x = x.data
        N, C, H, W = x.shape
        stride = self.stride
        # Hpai = ( H - self.HH ) // stride + 1
        # Wpai = ( W - self.WW ) // stride + 1
        xreshaped = x.reshape(N, C, H // stride, stride, W // stride, stride)
        out = xreshaped.max(axis=3).max(axis=4)
        self.xreshaped = xreshaped
        self.out = out
        return uvar.BaseVar(data=out)

    def backward(self, dout):
        """
        - dout: shape of (N, C, H', W')
        """
        x = self.x.data
        dout = dout.grad
        # N, C, H, W = x.shape
        xreshaped = self.xreshaped
        # dout (N, C, H', W')
        # Hpai, Wpai = dout.shape[2], dout.shape[3]
        dRx = np.zeros_like(xreshaped)
        mask = (xreshaped == self.out[:, :, :, None, :, None])
        doutReshape = dout[:, :, :, None, :, None]
        doutReshape, _ = np.broadcast_arrays(doutReshape, dRx)
        dRx[mask] = doutReshape[mask]
        dx = dRx.reshape(x.shape)
        self.x.grad = dx
        return self.x


class AveragePool(Empty):
    def __init__(self, kernel_size, stride=1):
        """
        - kernel_size: a tupe of (M,N)
        """
        self.HH, self.WW = kernel_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        x = x.data
        N, C, H, W = x.shape
        stride = self.stride
        xreshaped = x.reshape(N, C, H // stride, stride, W // stride, stride)
        out = xreshaped.mean(axis=3).mean(axis=4)
        self.xreshaped = xreshaped
        self.out = out
        return uvar.BaseVar(out)

    def backward(self, dout):
        # TODO: backward impement
        return dout


class Dropout(Empty):
    mask = None

    def __init__(self, p):
        """
        - p: Dropout parameter. We drop each neuron output with probability p.
        """
        np.random.seed(100)
        self.p = p

    def forward(self, x):
        x = x.data
        mask = (np.random.rand(*x.shape) < self.p) / self.p
        out = x * mask
        out = out.astype(x.dtype, copy=False)
        self.mask = mask
        return uvar.BaseVar(data=out)

    def backward(self, dout):
        dx = self.mask * dout.data
        return uvar.BaseVar(grad=dx)


class Reshape(Empty):
    def __init__(self, *outShape):
        self.outShape = outShape

    def forward(self, x):
        dat = x.data
        self.xShape = dat.shape
        x.data = np.reshape(dat, self.outShape)
        return x

    def backward(self, dout):
        dout.grad = np.reshape(dout.grad, self.xShape)
        return dout


class BatchNormalization(Empty):
    # TODO: Implement this
    def forward(self, x):
        pass

    def backward(self, dout):
        pass


class Conv2d(Linear):
    def __init__(self, C, F, filter_size, stride=1, padding=0):
        """
        Input:
        - C: input channel
        - F: output channel
        - filter_size: size of filter in shape (HH,WW)
        - stride: stride default 0
        - padding: padding default 0
        """
        self.C = C
        self.F = F
        self.HH, self.WW = filter_size
        self.S = stride
        self.P = padding
        self.w = uvar.RandnVar((F, C, self.HH, self.WW))
        self.b = uvar.RandnVar((F, ))

    def forward(self, x):
        # - out: Output data, of shape (N, F, H', W') where H' and W' are given by
        #   H' = 1 + (H - HH) / stride
        #   W' = 1 + (W - WW) / stride
        # - x: Input data of shape (N, C, H, W) a BaseVar
        # - w: Filter weights of shape (F, C, HH, WW)
        # - b: Biases, of shape (F,)
        self.x = x
        x = x.data
        N, C, H, W = np.shape(x)
        assert C == self.C, "The input Channel of X is {:d} we need {:d}".format(
            C, self.C)
        pad = self.P
        H += pad + pad
        W += pad + pad
        Hpai = 1 + (H - self.HH) // self.S
        Wpai = 1 + (W - self.WW) // self.S
        out = np.zeros((N, self.F, Hpai, Wpai))
        padX = np.pad(
            x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            'constant',
            constant_values=0)
        self.padX = padX
        conv2dForward(out, padX, self.w.data, self.b.data, self.S)
        return uvar.BaseVar(data=out)

    def backward(self, dout):
        # x = self.x.data
        # N, C, H, W = np.shape(x)
        # assert C == self.C, "The input Channel of X is {:d} we need {:d}".format(C, self.C)

        dx, dw, db = conv2dBackward(self.padX, self.w.data, self.b.data,
                                    dout.grad, self.P, self.S)

        self.w.grad = dw
        self.b.grad = db
        self.x.grad = dx
        return self.x


## Sequence Model
class RNNCell(Empty):
    def __init__(self, D, H, alpha=0.01, beta=0, activation=Tanh()):
        """
        Input:
        - D: dimention of input X
        - H: dimention of hidden H
        - alpha, beta: in w initialization alpha*randn(M,O)+beta
        - activation: activation function, Tanh by default
            all supported activation functions are in this module
        - self: 
            - wx: 
        """
        self.wx = uvar.RandnVar((D, H), alpha=alpha, beta=beta)
        self.wh = uvar.RandnVar((H, H), alpha=alpha, beta=beta)
        self.bh = uvar.RandnVar((1, H), alpha=alpha, beta=beta)
        if isinstance(activation, Empty):
            self.activ = activation
        else:
            raise ValueError("%s is not support yet" % activation)

    def forward(self, x, h):
        """
        h_{t+1} = g(x Wx + h_{t} Wh + b)
        Input
        - x: input (N, D)
        - h: states (N, H)
        Return
        - ns: next state (N, H)
        """
        self.x, self.h = x, h
        w1 = np.dot(x.data, self.wx.data)
        w2 = np.dot(h.data, self.wh.data)
        w3 = w1 + w2 + self.bh.data
        ns = self.activ(uvar.BaseVar(w3))
        return ns

    def backward(self, dout):
        dout = self.activ.backward(dout).grad
        x, wx, h, wh, bh = self.x, self.wx, self.h, self.wh, self.bh
        x.grad = np.dot(dout, wx.data.T)
        wx.grad = np.dot(x.data.T, dout)
        h.grad = np.dot(dout, wh.data.T)
        wh.grad = np.dot(h.data.T, dout)
        bh.grad = np.sum(dout, axis=0, keepdims=True)
        return x

    def parameters(self):
        return [self.wx, self.wh, self.bh]

    def parameterDic(self):
        return {"wx": self.wx, "wh": self.wh, "bh": self.bh}

    def loadParameterDic(self, p):
        self.wx = p['wx']
        self.wh = p['wh']
        self.bh = p['bh']


class BidirectionalRNNCell(Empty):
    """
    TODO: </>
    Bidirectional RNN

    """

    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, dout):
        return dout


class GRUCell(Empty):
    """
    TODO: </>
    GRU
    [Cho et al., 2014. On the properties of neural translation: Encoder-decoder appeoaches]
    [Chung et al., 2014. Empircal Evaluation of Gated Recurrent Neural Networks on Sequence Modeling]
    """

    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, dout):
        return dout


class LSTMCell(Empty):
    """
    TODO: </>
    LSTM Long short term memory
    """

    def __init__(self):
        pass

    def forward(self, x, h):
        return x

    def backward(self, dout):
        return dout


class Attention(Empty):
    """
    TODO: </>
    Attention Model
    [2014, Neural machine translation by jointly learning to align and translate]
    [2018, Attention is all you need]
    """

    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, dout):
        return dout

