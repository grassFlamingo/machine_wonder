import numpy as np
import NNetWork.Layers as nn
import NNetWork.Variables as uvar

def svmLoss(output, target):
    """
    Input: 
    - output: model forward output
    - target: labels
    Return:
    - loss
    - the gradent of output is stortage in output.grad
    """
    x = output.data
    y = target.data
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    output.grad = dx
    return loss

def softMax(output, target):
    """
    Input: 
    - output: model forward output
    - target: labels
    Return:
    - loss
    - the gradent of output is stortage in output.grad
    """
    x = output.data
    y = target.data
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -1 * np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    output.grad = dx
    return loss

def sigmoidCrossEntropyWithLogits(output, target):
    """
    Input: 
    - output: model forward output
    - target: labels
    Return:
    - loss 
    $L = - z \log(\sigma(x))) - (1-z)\log(1 - \sigma(x))
       = x - zx + \log(1 + e^{-x})$
    - dx
    $dx = \dfrac{1}{1 + e^{-x}} - z$
    - the gradent of output is stortage in output.grad
    """
    x = output.data
    z = target.data
    sigma = 1 / (1 + np.exp(-x))
    loss = x - z*x - np.log(sigma)
    output.grad = sigma - z
    return loss

def leastSquare(output, target):
    diff = output.data - target.data
    loss = np.square(diff)
    output.grad = 2 * diff
    return loss

    