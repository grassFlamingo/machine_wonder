import numpy as np
import numpy.random

################################
# Variables
################################


class SimpleVariable:
    def __init__(self, data, grad=None, auto_grad=False):
        self.data = np.asarray(data)
        self.grad = 0 if grad is None else np.asarray(grad)
        self.grad_fn = []
        self.auto_grad = auto_grad

    def add_grad_fn(self, gfn):
        self.grad_fn.append(gfn)

    def __str__(self):
        return f"SimpleVariable{{shape: {self.shape}, dtype: {self.dtype}}}\n{self.data}"

    def __getattribute__(self, name):
        if name == 'shape':
            return self.data.shape
        elif name == 'dtype':
            return self.data.dtype
        else:
            return super().__getattribute__(name)


    @staticmethod
    def randn(*shape, grad=None, auto_grad=False):
        sv = SimpleVariable(np.random.randn(*shape),
                            grad=grad, auto_grad=auto_grad)
        return sv

    @staticmethod
    def rand(*shape, grad=None, auto_grad=False):
        sv = SimpleVariable(np.random.rand(*shape),
                            grad=grad, auto_grad=auto_grad)
        return sv

