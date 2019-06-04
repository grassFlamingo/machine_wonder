import numpy as np

from gradient_check import eval_numerical_gradient

y = np.random.randn(50)
y_label = np.random.rand(50)

# 1/2 * (y - y_label)^2
def l2loss(y):
    return 1/2 * np.square(y - y_label)

def l2Grad(y):
    return y - y_label



print(y.shape, y.dtype)
lloss = l2loss(y)
print(lloss.shape, lloss.dtype)

zzr = np.zeros_like(y)
print(zzr.shape, zzr.dtype)
# print("loss l2 of y")
# print(l2loss(y))
print(type(y))

dyfun = l2Grad(y)
dyNum = eval_numerical_gradient(l2loss, y)
print()

diff = np.sum(np.abs(dyfun - dyNum))
print("l2 diff", diff)