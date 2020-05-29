# This is a simple tutorial of numpy

#%%
# (1) first you need to import numpy
import numpy as np


#%%
# (2) list of numpy
np_list = np.array([1, 2, 3, 4, 5])

print(np_list)

#%%
# (3) arange
np_range = np.arange(100)
print(np_range)

#%%
# (4) ones and zeros
np_ones = np.ones((2,3))
print(np_ones)

np_ones_l = np.ones_like(np_list)
print(np_ones)

np_zeros = np.zeros((2,3))
print(np_zeros)

#%%
# (5) linspace, logspace
np_ls = np.linspace(1, 100, 20)
print(np_ls)

np_ls = np.logspace(1, 20, 200)
print(np_ls)


#%%
# (6) eye, diag, trace
np_eye = np.eye(4)
print(np_eye)

print(np.trace(np_eye))
print(np.diag(np_eye))


#%%
# (7) rand, randn, randint
# uniform distribution U(0,1)
np_rand = np.random.rand(3,4)
print(np_rand)

# normal distribution N(0,1)
np_rand = np.random.randn(3,4)
print(np_rand)

# uniform distrubution U \in [low, high)
np_rand = np.random.randint(0, 4, (3,4))
print(np_rand)


#%%
# (8) reshape, transpose

arr = np.random.rand(3*5)
print(arr)
arr = np.reshape(arr, (3,5))
print(arr)
arr = arr.reshape(3,5)
print(arr)
# transpose
print(arr.T)

arr = np.random.rand(3*4*5)
print(arr)
arr = np.reshape(arr, (3,4,5))
print(arr)
arr = np.transpose(arr, (0,2,1))
print(arr)

#%%
# (9) max, min, std, var, sum, mean
arr = np.random.rand(16)
print(np.max(arr))
print(np.min(arr))
print(np.std(arr))
print(np.var(arr))
print(np.sum(arr))
print(np.mean(arr))

#%%
# (10) indexing 
arr = np.random.rand(3,4,5)
print(arr[1])
print(arr[-1])
print(arr[0,1])
print(arr[0:1, 2])
print(arr[0,0,0])

arr[0] = 1
print(arr)
arr[1, 1:2] += 1
print(arr)

#%% 
# (11) + - * /
arr = np.random.rand(4)
print(arr + 3)
print(arr - 3)
print(arr * 2)
print(arr / 2)
aii = np.random.rand(4)

print(arr + aii)
print(arr - aii)
print(arr * aii)
print(arr / aii)

#%% 
# (12) broadcast
arr = np.random.rand(1,3,4)
aii = np.random.rand(2,3,1)

print(arr * aii)

#%% 
# (13) matmul
arr = np.random.rand(3,4)
aii = np.random.rand(4,5)

print(np.matmul(arr, aii))
print(np.dot(arr, aii))

#%% 
# (14) svd, inv

arr = np.random.rand(4,4)

U, S, V = np.linalg.svd(arr)
print(U, S, V, end="\n")

iarr = np.linalg.inv(arr)
print(iarr)

print(np.matmul(iarr, arr))

#%%
# (15) copy
arr = np.random.rand(3,4)
aii = arr.copy()
print(arr)
aii[0,0] = -1
print(aii)
