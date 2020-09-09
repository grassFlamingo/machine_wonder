import numpy as np
import SNet.Variables as SVar


################################
# The Optimizer
################################


class SimpleOptimizer:
    def __init__(self, params, learning_rate=1e-3):
        self.params = params
        self.learning_rate = learning_rate

    def step(self):
        lr = self.learning_rate

        for p in self.params:
            p.data -= lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                del p.grad
            p.grad = 0


class Adam(SimpleOptimizer):
    def __init__(self, params, learning_rate=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, learning_rate=learning_rate)

        self.beta1, self.beta2 = betas
        self.eps = eps
        self.mvt = [[np.zeros(p.shape), np.zeros(p.shape), 1]
                    for p in self.params]

    def step(self):
        lr = self.learning_rate
        for p, mvt in zip(self.params, self.mvt):
            p.data -= lr * self._step(p.grad, mvt)

    def __step(self, dx, mvt):
        beta1, beta2 = self.beta1, self.beta2
        m, v, t = mvt
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


class Momentum(SimpleOptimizer):
    def __init__(self, params, learning_rate=1e-3, beta=0.7):
        super(Momentum, self).__init__(params, learning_rate)

        self.hisparm = [np.zeros_like(p.data) for p in params]
        self.beta = beta
        self.ateb = 1 - beta

    def step(self):
        beta, ateb, lr = self.beta, self.ateb, self.learning_rate

        for i, p in enumerate(self.params):
            hp = beta * self.hisparm[i] + ateb * p.grad
            p.data -= lr * hp
            self.hisparm[i] = hp


class RMSProp(SimpleOptimizer):
    def __init__(self, params, learning_rate=1e-3, beta=0.99, eps=1e-8):
        super(RMSProp, self).__init__(params, learning_rate)
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
