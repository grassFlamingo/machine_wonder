import numpy as np
from SNet.Variables import SimpleVariable


def eval_numerical_gradient(f, x, verbose=False, h=0.00001):
    """ 
    a naive implementation of numerical gradient of f at x 
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad

# code copied from cs231n assignment1 2017


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def random_split_index(start, end, count):
    left = start
    skip = 2*(end - start) // count
    for _ in range(count-1):
        right = left + np.random.randint(1, skip)
        yield left, right
        left = right
    yield left, end

#####################################################
# invoke

def invoke_simple_layer_member(obj, name, X: SimpleVariable = None):

    w = obj.__dict__[name]
    old = w.data.copy()

    def __update(value):
        w.data = value
        if X:
            out = obj.forward(X).data
        else:
            out = obj.forward().data
        return out

    def __recover():
        w.data = old

    return __update, __recover


def invoke_simple_layer_output(obj):
    x = SimpleVariable(None)

    def __inner(value):
        x.data = value
        return obj.forward(x).data

    return __inner


def invoke_simple_loss_output(obj, target: SimpleVariable):
    x = SimpleVariable(None)

    def __inner(value):
        x.data = value
        return obj.forward(x, target).data

    return __inner
