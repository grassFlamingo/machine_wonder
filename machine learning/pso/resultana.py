
#%% 
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import re

#%%
pattern = re.compile(r'epho (\d+) gloss ([-.\d]+) \[ ([-.\d]+) ([-.\d]+) ([-.\d]+) \]')

#%%
def ana_result_log(logfile):
    record = []
    with open(logfile) as f:
        for line in f.readlines():
            if line[0] != 'e':
                continue
            catch = pattern.match(line)
            record.append([catch.group(i) for i in range(1,6)])
    return np.asarray(record, dtype=np.float16)

#%%
recordpso = ana_result_log('result-square.txt')
recordgrd = ana_result_log('result-square-grad.txt')

#%%
plt.plot(record[:,0], record[:,1])
plt.xlabel("t")
plt.ylabel("$f(x,y,z)$")
plt.show()

#%%
plt.figure(figsize=(16,5))

for i, l in zip([2,3,4], ['x', 'y', 'z']):
    plt.subplot(1,3,i-1)
    plt.plot(record[:,0], record[:,i])
    plt.xlabel("t")
    plt.ylabel(l)
plt.show()


#%% 
# PSO and read on square
plt.plot(recordpso[0:30,0], recordpso[0:30,1], '-s')
plt.plot(recordgrd[0:30,0], recordgrd[0:30,1], "-o")
plt.legend(["pso", "grad"])
plt.xlabel("t")
plt.ylabel("$f(x,y,z)$")
plt.show()


#%% [markdown]
# This is the figure of that crazy sin function
# %%
theX = np.linspace(-5, 5, 800)
theY = np.linspace(-5, 5, 800)
theX, theY = np.meshgrid(theX, theY)

theZ = (theX*theX + theY*theY)/3 + np.sin(theX*3) + np.sin(theY*3)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_surface(theX, theY, theZ, cmap=plt.cm.jet)
plt.xlabel("x")
plt.ylabel('y')
# plt.title("$g = (x^2 + y^2)/3 + \sin(3x) + \sin(3y)$")
plt.show()

#%% [markdown]

#%%
recordpsos = [ana_result_log(f'result-sin-pso{i:d}.txt') for i in range(0,4)]
recordgrds = [ana_result_log(f'result-sin-grad{i:d}.txt') for i in range(0,4)]


marks = ['-o', '-^', '-p', '-d']


#%%
def draw_records(recs, marker, ylabel):
    plt.figure(figsize=(14,8))
    for rec, mak in zip(recs, marker):
        plt.plot(rec[0:40,0], rec[0:40,1], mak)
    plt.xlabel('t')
    plt.ylabel(ylabel)
    plt.show()

#%%
draw_records(recordpsos, marks, 'PSO g')
draw_records(recordgrds, marks, 'Grad g')

#%%
def draw_record4(recs, name):
    plt.figure(figsize=(25,10))
    for i, rec in enumerate(recs):
        plt.subplot(2,2,i+1)
        plt.plot(rec[0:150,0], rec[0:150,2:5])
        plt.xlabel('t')
        plt.title(f"{i}")
    plt.title(name)
    plt.show()


# %%
draw_record4(recordpsos, "PSO")
draw_record4(recordgrds, "Grad")
