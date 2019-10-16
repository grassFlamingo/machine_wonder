# %%
# import os
# os.chdir("machine learning/decision tree")

# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# %matplotlib notebook

# %%
trainData = pd.read_csv(
    "machine learning/decision tree/mobile-price-classification/train.csv",
    sep=",",
)

# %%


class DecisionNode:
    def __init__(self, feature: str, label, value):
        """
        feature
        """
        self.feature = feature
        self.label = label
        self.value = value
        self.leaves = []
        self.subfeature = "--"

    def append_node(self, node):
        self.leaves.append(node)
        self.subfeature = node.feature

    def has_nodes(self) -> bool:
        return len(self.leaves) > 0

    def nodes_iterator(self):
        yield from self.leaves

    def predict(self, data: pd.DataFrame):
        if len(self.leaves) <= 0:
            return self.label
        val = data[self.subfeature]
        for l in self.leaves:
            if l.value == val:
                return l.predict(data)
        return self.label

    def __str__(self, depth=0):
        pres = ".."*depth
        if self.has_nodes():
            leaves = "\n"+"\n".join([l.__str__(depth+1) for l in self.leaves])
            label = ""
        else:
            leaves = ""
            label = f" [{self.label}]"

        return f"|{pres}{self.feature} {self.value}{label}{leaves}"

# %%


def column_entropy(d: pd.DataFrame, colname):
    count = d[colname].value_counts().to_numpy()
    prob = count / count.sum()
    ent = - np.sum(prob * np.log(prob))
    return ent

# %%
def ID3(dataset: pd.DataFrame, features: list, value, oldfearute, epsilon=1e-5):
    """
    dataset: Dataset, the last column are labels
    features: feature set
    epsilon: error bound
    """
    # if all instances in D are same category
    valueCounts = dataset.iloc[:, -1].value_counts()
    label = valueCounts.idxmax()
    if len(valueCounts) <= 1 or len(features) <= 0:
        # print(f"end with {len(valueCounts)} or {len(features)}")
        return DecisionNode(str(dataset.columns[-1]), label, value)

    # compute information Gain
    minent, minfea = np.inf, None

    for fea in features:
        ent = column_entropy(dataset, fea)
        if ent < minent:
            minent = ent
            minfea = fea

    entD = column_entropy(dataset, dataset.columns[-1])
    # print(f"min entropy is {minent}, feature is {minfea} and gain is {entD-minent}")

    if entD - minent < epsilon:
        return DecisionNode(minfea, label, value)

    choices = dataset[minfea].unique()

    dropfea = features.copy()
    dropfea.remove(minfea)

    node = DecisionNode(oldfearute, label, value)
    dropset = dataset.drop(columns=minfea)
    for co in choices:
        subnode = ID3(dropset[dataset[minfea] == co], dropfea, co, minfea)
        node.append_node(subnode)

    return node
#%%
def Dtree_C45(dataset: pd.DataFrame, features: list, value, oldfearute, epsilon=1e-2):
    """
    dataset: Dataset, the last column are labels
    features: feature set
    epsilon: error bound
    """
    # if all instances in D are same category
    valueCounts = dataset.iloc[:, -1].value_counts()
    label = valueCounts.idxmax()
    if len(valueCounts) <= 1 or len(features) <= 0:
        # print(f"end with {len(valueCounts)} or {len(features)}")
        return DecisionNode(str(dataset.columns[-1]), label, value)

    # compute information Gain
    maxgain, selectfea = -np.inf, None

    entD = column_entropy(dataset, dataset.columns[-1])
    # TODO: C4.5
    for fea in features:
        ent = column_entropy(dataset, fea)
        gain = (entD - ent) / (ent + 1e-8)
        if gain > maxgain:
            maxgain = gain
            selectfea = fea

    # print(f"min entropy is {minent}, feature is {selectfea} and gain is {entD-minent}")

    if maxgain < epsilon:
        return DecisionNode(selectfea, label, value)

    choices = dataset[selectfea].unique()

    dropfea = features.copy()
    dropfea.remove(selectfea)

    node = DecisionNode(oldfearute, label, value)
    dropset = dataset.drop(columns=selectfea)
    for co in choices:
        subnode = Dtree_C45(dropset[dataset[selectfea] == co], dropfea, co, selectfea)
        node.append_node(subnode)

    return node


#%%
featureSet = list(trainData.columns[0:-1])
print(featureSet)

realTrainSet = trainData.loc[0:1800]
realTestSet = trainData.loc[1800::]

tree = ID3(trainData.loc[0:100], featureSet, '--', '--')

print(tree)

#%%
# predict
realout, predout = [], []
for i, row in trainData.loc[0:100].iterrows():
    preded = tree.predict(row)
    # print(f"{i} real {row[-1]} pred {preded}")
    realout.append(row[-1])
    predout.append(preded)

print((np.asarray(realout) == np.asarray(predout)).mean())

# predict
realout, predout = [], []
for i, row in trainData.loc[1800:2000].iterrows():
    preded = tree.predict(row)
    # print(f"{i} real {row[-1]} pred {preded}")
    realout.append(row[-1])
    predout.append(preded)

print((np.asarray(realout) == np.asarray(predout)).mean())


#%%
tree = Dtree_C45(trainData.loc[0:1800], featureSet, '--', '--')

print(tree)

#%%
# predict
realout, predout = [], []
for i, row in trainData.loc[0:100].iterrows():
    preded = tree.predict(row)
    # print(f"{i} real {row[-1]} pred {preded}")
    realout.append(row[-1])
    predout.append(preded)

print((np.asarray(realout) == np.asarray(predout)).mean())

# predict
realout, predout = [], []
for i, row in trainData.loc[1800:2000].iterrows():
    preded = tree.predict(row)
    # print(f"{i} real {row[-1]} pred {preded}")
    realout.append(row[-1])
    predout.append(preded)

print((np.asarray(realout) == np.asarray(predout)).mean())
