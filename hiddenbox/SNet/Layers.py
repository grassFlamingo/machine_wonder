import numpy as np
from SNet.Variables import SimpleVariable


################################
# Layers
################################


class SimpleLayer:
    def forward(self, x: SimpleVariable):
        self.cache_in = x
        return x

    def backward(self, df: SimpleVariable):
        self.cache_in.grad += df
        return self.cache_in

    def __call__(self, *x):
        return self.forward(*x)

    def parameters(self):
        params = []
        for value in self.__dict__.values():
            if isinstance(value, SimpleLayer):
                params += value.parameters()
            elif isinstance(value, SimpleVariable):
                if value.auto_grad:
                    params.append(value)
            else:
                continue
        return params


class Linear(SimpleLayer):
    def __init__(self, in_channel, out_channel):
        self.W = SimpleVariable.randn(
            in_channel, out_channel, auto_grad=True)
        self.b = SimpleVariable.randn(out_channel, auto_grad=True)

    def forward(self, x: SimpleVariable):
        out = np.matmul(x.data, self.W.data) + self.b.data
        out = SimpleVariable(out)
        self.cache_in = x
        self.cache_out = out
        return out

    def backward(self, df: SimpleVariable):
        dx = np.matmul(df.grad, self.W.data.T)
        dw = np.matmul(self.cache_in.data.T, df.grad)
        db = np.sum(df.grad, axis=0)

        self.W.grad = dw
        self.b.grad = db
        self.cache_in.grad += dx
        del self.cache_out
        return self.cache_in


class Sigmoid(SimpleLayer):

    def forward(self, x: SimpleVariable):
        ex = np.exp(-x.data)

        out = 1 / (1 + ex)

        out = SimpleVariable(out)

        self.cache_in = x
        self.cache_out = out
        self.cache_ex = ex

        return out

    def backward(self, df: SimpleVariable):
        out = self.cache_out.data

        dx = self.cache_ex * out * out

        self.cache_in.grad = df.grad * dx

        del self.cache_out, self.cache_ex
        return self.cache_in


class ReLU(SimpleLayer):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def forward(self, x: SimpleVariable):
        mask = x.data < 0

        self.cache_in = x
        if self.inplace:
            x.data[mask] = 0
        else:
            y = x.data.copy()
            y[mask] = 0
            x = SimpleVariable(y)
        self.cache_mask = mask
        return x

    def backward(self, df: SimpleVariable):
        dx = df.grad.copy()
        dx[self.cache_mask] = 0
        self.cache_in.grad += dx
        del self.cache_mask
        return self.cache_in


class ReLU6(ReLU):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def forward(self, x: SimpleVariable):
        mask = (x.data < 0) + (x.data > 6)

        self.cache_in = x
        if self.inplace:
            x.data[mask] = 0
        else:
            y = x.data.copy()
            y[mask] = 0
            x = SimpleVariable(y)
        self.cache_mask = mask
        return x


class LeakyReLU(SimpleLayer):
    def __init__(self, negative_slope=0.02, inplace=False):
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: SimpleVariable):
        mask = x.data < 0

        self.cache_in = x
        if self.inplace:
            x.data[mask] *= self.negative_slope
        else:
            y = x.data.copy()
            y[mask] *= self.negative_slope
            x = SimpleVariable(y)
        self.cache_mask = mask
        return x

    def backward(self, df: SimpleVariable):
        dx = df.grad.copy()
        dx[self.cache_mask] *= self.negative_slope
        self.cache_in.grad += dx
        del self.cache_mask
        return self.cache_in


class Tanh(SimpleLayer):
    def forward(self, x: SimpleVariable):
        out = np.tanh(x.data)
        out = SimpleVariable(out)
        self.cache_in = x
        self.cache_out = out
        return out

    def backward(self, df: SimpleVariable):
        dx = df.grad * (1 - np.square(self.cache_out.data))
        self.cache_in.grad += dx
        del self.cache_out
        return self.cache_in


class Sin(SimpleLayer):
    def forward(self, x: SimpleVariable):
        out = np.sin(x.data)

        self.cache_in = x
        return SimpleVariable(out)

    def backward(self, df: SimpleVariable):
        dx = np.cos(self.cache_in.data) * df.grad

        self.cache_in.grad += dx

        return self.cache_in


class Cos(SimpleLayer):
    def forward(self, x: SimpleVariable):
        out = np.cos(x.data)
        self.cache_in = x
        return SimpleVariable(out)

    def backward(self, df: SimpleVariable):
        dx = -np.sin(self.cache_in.data) * df.grad

        self.cache_in.grad += dx
        return self.cache_in


class Sequences(SimpleLayer):
    def __init__(self, *sequence):
        self.sequence = sequence
        for i, seq in enumerate(sequence):
            self.__dict__["sublayer%03d" % i] = seq

    def forward(self, x: SimpleVariable):
        tmp = x
        for layer in self.sequence:
            tmp = layer(tmp)
        return tmp

    def backward(self, df: Linear):
        tmp = df
        for layer in reversed(self.sequence):
            tmp = layer.backward(tmp)
        return tmp


################################
# Losses
################################


class SimpleLoss(SimpleLayer):
    """
    This is Simplly loss
    """

    def forward(self, predict: SimpleVariable, target: SimpleVariable) -> SimpleVariable:
        diff = predict.data - target.data
        adiff = np.abs(diff)
        out = SimpleVariable(np.sum(adiff))

        dx = np.ones(diff.shape)
        dx *= np.sign(diff)

        self.cache_dx = dx
        self.cache_in = predict
        return out

    def backward(self):
        self.cache_in.grad += self.cache_dx
        del self.cache_dx
        return self.cache_in


class LossL2(SimpleLoss):
    def forward(self, predict: SimpleVariable, target: SimpleVariable) -> SimpleVariable:
        diff = predict.data - target.data
        out = np.square(diff).sum()
        out = SimpleVariable(out)

        self.cache_in = predict
        self.cache_dx = 2 * diff
        return out


class L2Regulizer(SimpleLoss):
    def __init__(self, parameters, factor):
        super(L2Regulizer, self).__init__()

        self.parameters = []
        self.factor = factor

        for i, p in enumerate(parameters):
            if not isinstance(p, SimpleVariable):
                continue
            self.parameters.append(p)
            self.__dict__["parm%03d" % i] = p

    def forward(self):
        loss = 0
        for p in self.parameters:
            loss += np.sum(p.data * p.data)
        return SimpleVariable(self.factor * loss)

    def backward(self):
        factor = 2*self.factor
        for p in self.parameters:
            p.grad += factor * p.data


class LossKL(SimpleLoss):
    """Kullback-Leibler loss.

    ```
    kl(p || q) = Sum p log(p/q)
    ```

    Require:
    - predict (SimpleVariable): (q) predict probabilities, must in [0,1]
    - target (SimpleVariable): (p) target probabilities, must in [0,1]

    Return:
    - loss
    """

    def forward(self, predict: SimpleVariable, target: SimpleVariable) -> SimpleVariable:
        self.cache_in = predict

        logtarget = np.log(target.data)
        logpredict = np.log(predict.data)

        loss = target.data * (logtarget - logpredict)

        dx = - target.data / predict.data
        self.cache_dx = dx
        return SimpleVariable(np.sum(loss))


class CrossEntropyWithLogits(SimpleLoss):
    """
    $L = - z \log(\sigma(x))) - (1-z)\log(1 - \sigma(x))
       = x - zx + \log(1 + e^{-x})$
    - dx
    $dx = \dfrac{1}{1 + e^{-x}} - z$
    """

    def forward(self, predict: SimpleVariable, target: SimpleVariable) -> SimpleVariable:
        x = predict.data
        z = target.data
        sigma = 1 / (1 + np.exp(-x))
        loss = x - z*x - np.log(sigma)
        self.cache_in = predict
        self.cache_dx = sigma - z
        return SimpleVariable(loss)


class SoftMaxCrossEntropyWithLogits(SimpleLoss):
    def forward(self, predict, target) -> SimpleVariable:
        x = target.data
        y = target.data
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = x.shape[0]
        loss = -1 * np.sum(log_probs[np.arange(N), y]) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        self.cache_dx = dx
        self.cache_in = predict
        return SimpleVariable(loss)


class SVMLoss(SimpleLoss):
    def forward(self, predict, target):
        self.cache_in = predict
        x = predict.data
        y = target.data
        N = x.shape[0]
        correct_class_scores = x[np.arange(N), y]
        margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
        margins[np.arange(N), y] = 0
        loss = np.sum(margins) / N
        num_pos = np.sum(margins > 0, axis=1)
        dx = np.zeros_like(x)
        dx[margins > 0] = 1
        dx[np.arange(N), y] -= num_pos
        dx /= N
        self.cache_dx = dx
        return SimpleVariable(loss)


#####################################################
