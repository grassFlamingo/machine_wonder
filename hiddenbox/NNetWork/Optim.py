import numpy as np
import NNetWork.Variables as uvar


class Optim():
    def __init__(self, params, lr=1e-3):
        self.params = [p for p in params if p is not None]
        self.lr = lr

    def step(self):
        lr = self.lr
        for p in self.params:
            p.data -= lr * p.grad

class Adam(Optim):
    def __init__(self, params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-08):
        super(Adam,self).__init__(params, lr=lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.mvt = [[np.zeros(p.shape), np.zeros(p.shape), 1] 
                    for p in self.params]

    def step(self):
        lr = self.lr
        for p, mvt in zip(self.params, self.mvt):
            p.data -= lr * self._step(p.grad, mvt)

    def _step(self, dx, mvt):
        beta1, beta2 = self.beta1, self.beta2
        m,v,t = mvt
        t = t + 1
        # Momentum
        m = beta1 * m + (1 - beta1) * dx

        # AdaGrad
        v = beta2 * v + (1 - beta2) * dx * dx

        mt = m / (1 - beta1**t)  # for batch normalization
        vt = v / (1 - beta2**t)  # for batch normalization

        mvt[0], mvt[1], mvt[2] = m, v, t

        # AdaGrid/RMSProp
        return mt / (np.sqrt(vt) + self.eps)


class Momentum(Optim):
    def __init__(self, params,  beta=0.9, lr=1e-3):
        super(Momentum,self).__init__(params, lr=lr)
        self.beta1 = beta
        self.beta2 = 1 - beta
        self.vdx = [np.zeros_like(p) for p in self.params]

    def step(self):
        beta1, beta2 = self.beta1, self.beta2
        vdx = self.vdx
        lr = self.lr
        for i, p in enumerate(self.params):
            v = beta1 * vdx[i] + beta2 * p.grad
            vdx[i] = v
            p.data -= lr * v

class RMSProp(Optim):
    def __init__(self, params, beta=0.99, eps=1e-8, lr=1e-3):
        super(RMSProp, self).__init__(params, lr=lr)
        self.beta1 = beta
        self.beta2 = 1 - beta
        self.eps = eps
        self.sdx = [np.zeros_like(p) for p in self.params]

    def step(self):
        beta1, beta2 = self.beta1, self.beta2
        sdx = self.sdx
        lr = self.lr
        for i, p in enumerate(self.params):
            dx = p.grad
            s = beta1 * sdx[i] + beta2 * np.square(dx)
            sdx[i] = s
            p.data -= lr * dx / (np.sqrt(s) + self.eps)

