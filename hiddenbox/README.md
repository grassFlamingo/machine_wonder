# This is a toy implementation of ML algorithm

## Directory Details

- Dataset
  - MNIST | The MNIST Dataset
  - mnistReader.py | A mnist file reader
- NNetWork | Network implementation
  - Layers.py | Linear, ReLU, Sigmoid ...
  - LossFunc.py | L2, svmLoss, softMax
  - Optim.py | Optimizers, Adam, Momentum, RMSProp
  - Variables.py | The Variable in NNetWork

## Layer Supported

- Linear
- Serial
- MaxPool
- Dropout
- Reshape
- BatchNormalization
- Conv2d
- ReLU, LeakyReLU, Sigmoid, Tanh, Tanhshrink

For More Please Read [NNetWork/Layers.py](NNetWork/Layers.py)

You can call the forward method by calling it as a function.

```python
FC = nn.Linear(2,3)
FC.forward(x)
FC(x)
```

## Jupyter Notebook Files

### AnotherMNIST.ipynb [</>]

Another Vertion of [mnist_lunch](https://github.com/grassFlamingo/mnist_lunch)

### ConvMnist.ipynb [</>]

Use Convolution Layers on MINST dataset

### ConvMnistTF.ipynb [</>]

The Convnet of NNetWork is too slow...

Try Tensorflow

### Information_Picker.ipynb [>/<]

An attempt

### linear_regression.ipynb

Simple Linear Regression (@_@)

### NetWorkInTIme.ipynb [</>]

To see the changes of weights through time

### CheckMetroCity.ipynb [</>]

Use Dataset/MetroAreas.CSV
