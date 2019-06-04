import numpy as np
from NNetWork.Util import eval_numerical_gradient

# x = np.random.randn(5,5)
# w = np.random.randn(5,5)

# y = np.dot(x,w)

# dx = w # ...
# h = 1e-5
# dx = (np.dot(x + h, w) - np.dot(x - h, w)) / (2 * h)
# oones = np.ones((5,5))
# print(dx)
# print(oones.dot(w))
# exit(1)

# def eval_numerical_gradient(f, x, h=0.00001):
#     return (f(x + h) - f(x - h)) / (2 * h)


x = np.random.randn(5,5)
a = np.random.randint(-10,10)
b = np.random.randint(-10,10)

def forward(x, w, b):
    return x * w + b

def eval_diff(dx, ndx):
    return np.sum(dx - ndx)

fx = lambda x: forward(x, a, b)
fa = lambda w: forward(x, w, b)
fb = lambda q: forward(x, a, q)

y = forward(x,a,b)

# print("f(x)\n", fx(x))
# print("f(a)\n", fa(a))
# print("f(b)\n", fb(b))

dout = np.ones_like(y)

dx = a * dout
da = np.mean(x)
db = np.mean(dout)

# print("dx\n", dx)
# print("da\n", da)
# print("db\n", db)

ndx = eval_numerical_gradient(fx,x)
nda = eval_numerical_gradient(fa,a)
ndb = eval_numerical_gradient(fb,b)

print("dx and ndx")
print(eval_diff(dx, ndx))
print("da and nda")
print(eval_diff(da, nda))
print("db and ndb")
print(eval_diff(db, ndb))
