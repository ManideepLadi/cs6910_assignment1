import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    numerator_values = np.exp(x - np.max(x))
    probabilities = numerator_values / np.sum(numerator_values)
    return probabilities


def sigmoid_backward_propagation(dh, a):
    sig = sigmoid(a)
    da = dh * sig * (1 - sig)
    return da


def relu_backward_propagation(dh, a):
    da = np.array(dh, copy=True)
    da[a <= 0] = 0
    return da
