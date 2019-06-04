import numpy as np
import numpy.random

class BaseVar:
    data = None
    grad = None
    shape = None
    def __init__(self, data=None, grad=None):
        self.data = data
        self.grad = grad
        self.shape = None if data is None else data.shape
    
    def __str__(self):
        return "BaseVar\nData {0}\nGrid {1}\n".format(self.data, self.grad)

    def __add__(self, other):
        diff = None
        if isinstance(other, BaseVar):
            diff = self.data + other.data
        else:
            diff = self.data + other
        return BaseVar(data=diff)

    def __sub__(self, other):
        diff = None
        if isinstance(other, BaseVar):
            diff = self.data - other.data
        else:
            diff = self.data - other
        return BaseVar(data=diff)

    def __mul__(self, other):
        diff = None
        if isinstance(other, BaseVar):
            diff = self.data * other.data
        else:
            diff = self.data * other
        return BaseVar(data=diff)

    def __truediv__(self, other):
        diff = None
        print("calling div", other)
        if isinstance(other, BaseVar):
            diff = self.data / other.data
        else:
            diff = self.data / other
        return BaseVar(data=diff)

    def __getitem__(self, key):
        if self.data is None:
            raise ValueError("data is None")
        return BaseVar(data=self.data[key])

    def argmax(self, axis=None):
        """
        Input:
        - axis: the axis to get max argument
        Return:
        A numpy the max grgument index of data
        """
        return BaseVar(np.argmax(self.data, axis=axis))

    def transpose(self, *args):
        return BaseVar(self.data.transpose(*args))
        
class ZeroVar(BaseVar):
   
    def __init__(self, shape):
        """
        - shape: tupe the shape of data type
        """
        data = np.zeros(shape)
        super(ZeroVar,self).__init__(data)

class RandnVar(BaseVar):

    def __init__(self, shape, alpha=1, beta=0):
        """
        - shape: tupe the shape of data type
        """
        data = alpha * np.random.randn(*shape) + beta
        super(RandnVar,self).__init__(data)

    def __str__(self):
        return "RandnVar\nData {0}\nGrad {1}".format(self.data, self.grad)