# reference from
# https://nathanrooy.github.io/posts/2018-03-22/word2vec-from-scratch-with-python-and-numpy/


import matplotlib.pyplot as plt
import itertools
import numpy as np
import re
import os
from collections import defaultdict
import argparse

__Args = argparse.ArgumentParser(description="Word2Vec")
__Args.add_argument("-e", "--embbed_dim", type=int,
                    default=2, help="dimension of embbed vector")
__Args.add_argument("-l", "--learning_rate", type=float,
                    default=1e-2, help="learning rate")
__Args.add_argument("-p", "--ephos", type=int, default=200, help="ephos")
__Args.add_argument("-w", "--window_size", type=int,
                    default=2, help="size of window")
__Args.add_argument("-n", "--negative_sample", type=int, default=5,
                    help="number of negative words to use during training")
__Args.add_argument("-c", "--corpus", type=str, default="the quick brown fox jumps over the lazy dog",
                    help="file or sentence of crops")


class __InnerArg:
    def __init__(self):
        self.word2Index = dict()
        self.index2Word = dict()
        self.wordcount = 0

    def preprocess_data(self, corpus: iter):
        """
        corpus is a iterator of word
        """
        index = 0
        for w in corpus:
            if w in self.word2Index:
                continue
            self.word2Index[w] = index
            self.index2Word[index] = w
            index += 1
        self.wordcount = max(0, index)


def softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex, axis=1, keepdims=True)


class Word2Vec:
    """
    Skip-gram model

    ```
                -> w(t-2)
                -> w(t-1)
    w(t) -> P
                -> w(t+1)
                -> w(t+2)
    ```
    """

    def __init__(self, word_count, embbed_dim):
        """
        Word2Vec

        Requires:
        - word_count: how many words in your dictionary
        - embbed_dim: embbed dimension
        """

        self.embbed = np.random.randn(word_count, embbed_dim)
        self.dense = np.random.randn(embbed_dim, word_count)

        class __inner:
            pass
        self.cache = __inner()
        self.cache.ddense = 0
        self.cache.demb = 0

    def forward(self, x):
        """
        - x: shape [batch, 1], dtype np.int32
        """
        # pick vectors from
        h = self.embbed[x]  # [batch, 1, embbed dim]
        h = np.squeeze(h, axis=1)
        u = np.matmul(h, self.dense)  # [batch, word_count]
        self.cache.h = h
        return u

    def loss(self, prerdict: np.ndarray, target: np.ndarray) -> np.ndarray:
        """

        ```
        loss = C \log \sum_i e^{u_i} - \sum_{j} u_j
        ```
        where j is the index of target vector

        - predict is the output of self.forward
        - target: shape [batch, window*2-1], dtype np.int32
        """
        C = target.shape[1]
        prerdict = prerdict - np.max(prerdict)
        eui = np.exp(prerdict)
        sumeui = np.sum(eui, axis=1, keepdims=True)
        loss = C * np.log(sumeui) - np.sum(prerdict[:, target], axis=1, keepdims=True)

        self.cache.target = target
        self.cache.dloss = eui / sumeui
        self.cache.dloss[:, target] -= 1
        self.cache.dloss /= target.size
        return np.mean(loss)

    def update(self):
        ddense = np.matmul(self.cache.h.T, self.cache.dloss)
        demb = np.matmul(self.cache.dloss, self.dense.T)

        # momentum optimizer
        self.cache.ddense = 0.6 * self.cache.ddense + 0.4 * ddense
        self.cache.demb = 0.6 * self.cache.demb + 0.4 * demb

        self.dense -= args.learning_rate * self.cache.ddense
        self.embbed[self.cache.target] -= args.learning_rate * self.cache.demb


args = __Args.parse_args()
innerArg = __InnerArg()


def circle_iter(corpus: iter):
    def __innerloop(buffer, bi, ans, bs):
        fl = bi - 1
        for i in range(ws):
            fl = (fl + 1) % bs
            if buffer[fl] < 0:
                continue
            ans.append(buffer[fl])

        fl = (fl+1) % bs
        w = buffer[fl]

        for i in range(ws):
            fl = (fl + 1) % bs
            if buffer[fl] < 0:
                continue
            ans.append(buffer[fl])
        return w

    itc = iter(corpus)
    ws = args.window_size
    bs = 2 * ws + 1
    ans = []
    buffer = [-1 for _ in range(bs)]
    bi = 0
    try:
        for _ in range(ws+1):
            buffer[bi] = innerArg.word2Index[next(itc)]
            bi += 1

        while True:
            w = __innerloop(buffer, bi, ans, bs)
            yield w, ans
            ans.clear()
            buffer[bi] = innerArg.word2Index[next(itc)]
            bi = (bi + 1) % bs

    except StopIteration:
        while True:
            buffer[bi] = -1
            bi = (bi + 1) % bs
            w = __innerloop(buffer, bi, ans, bs)
            if w == -1:
                return
            yield w, ans
            ans.clear()


BookPath = "../Dataset/Bed Time Stories"


def iter_books(bpath):
    rmstr = """()<"“!?-}{»#«&+0123456789]‘[©:|*~—”™’;=@"""
    smap = str.maketrans("", "", rmstr)

    for b in sorted(os.listdir(bpath)):
        bfpath = os.path.join(bpath, b)
        book = open(bfpath)
        while True:
            line = book.readline()
            if line == "":
                break
            line = line.translate(smap)
            for w in line.split():
                w = w.lower()
                if w[-1] in ['.', ',']:
                    yield w[0:-1]
                    yield w[-1]
                else:
                    yield w



innerArg.preprocess_data(iter_books(BookPath))
w2v = Word2Vec(innerArg.wordcount, 2)
# w2v.forward()


print(w2v.embbed)

for _, (c, cw) in itertools.product(range(64), circle_iter(iter_books(BookPath))):
    # print(innerArg.index2Word[c], "<-->",
    #       ",".join([innerArg.index2Word[i] for i in cw]))
    c = np.array([[c]])
    cw = np.asarray(cw).reshape(1, -1)
    u = w2v.forward(c)
    los = w2v.loss(u, cw)
    w2v.update()
    # print(los)


plt.scatter(w2v.embbed[:, 0], w2v.embbed[:, 1])

for k, v in innerArg.index2Word.items():
    xy = w2v.embbed[k]
    if np.sum(np.abs(xy)) > 1e+2: continue
    plt.annotate(v, xy)
plt.show()

print(innerArg.word2Index)
print(w2v.embbed)
