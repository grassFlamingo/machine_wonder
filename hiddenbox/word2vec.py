import SNet
import numpy as np
import os
import random

BookPath = "../Dataset/Bed Time Stories"


def iter_books(bpath):
    rmstr = """()<"“!?-}{»#«&+0123456789]‘[©:|*~—”™’;=@"""
    smap = str.maketrans("", "", rmstr)
    
    for b in sorted(os.listdir(bpath)): 
        bfpath = os.path.join(bpath, b)
        book = open(bfpath)
        while True:
            line = book.readline()
            if line == "": break
            line = line.translate(smap)
            for w in line.split():
                w = w.lower()
                if w[-1] in ['.', ',']:
                    yield w[0:-1]
                    yield w[-1]
                else:
                    yield w
                

StoryDictW2I = dict()
StoryDictI2W = dict()
StoryIndex = 0
for w in iter_books(BookPath):
    # print(w, end="\n\n" if w == "." else " ")
    if w in StoryDictW2I: continue
    StoryDictW2I[w] = StoryIndex
    StoryDictI2W[StoryIndex] = w
    StoryIndex += 1

# print(StoryDictW2I)

model = SNet.Layers.Sequences(
    SNet.Layers.Embbed(StoryIndex-1, 16),
    SNet.Layers.Linear(16, 1),
    SNet.Layers.SoftMax(),
)

indx = SNet.Var(np.array([1, 2, 3, 4]).reshape(1,4))

print(model(indx))