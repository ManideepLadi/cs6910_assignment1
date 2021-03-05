import numpy as np
from ActivationFunctions import sigmoid
from ActivationFunctions import relu
from ActivationFunctions import sigmoid_backward_propagation
from ActivationFunctions import relu_backward_propagation


class Layer:
    def __init__(self, n_inputs, n_neurons):
        np.random.seed(100)
        self.w = 0.10 * np.random.randn(n_neurons, n_inputs)
        self.b = np.zeros((n_neurons, 1))
        self.a = np.zeros((n_neurons, 1))
        self.h = np.zeros((n_neurons, 1))
        self.dw = np.zeros((n_neurons, n_inputs))
        self.db = np.zeros((n_neurons, 1))
        self.da = np.zeros((n_neurons, 1))

    def getWeights(self):
        return self.w

    def getBiases(self):
        return self.b

    def forward(self, input, activation):
        self.a = np.dot(input, self.w) + self.b
        if activation is 'relu':
            self.h = relu(self.a)
        elif activation is 'sigmoid':
            self.h = sigmoid(self.a)
        return self.h

    def backward(self, da_curr, h_prev, a_prev, activation):
        self.da = da_curr
        self.dw = np.dot(self.da, h_prev.T)
        self.db = self.da
        dh_prev = np.dot(self.w, self.da)
        if activation is 'sigmoid':
            da_prev = sigmoid_backward_propagation(dh_prev.T, a_prev)
        elif activation is 'relu':
            da_prev = relu_backward_propagation(dh_prev.T, a_prev)
        return da_prev

    def update_StochsticGradientDescent(self, eta):
        self.w = self.w - eta * self.dw
        self.b = self.b - eta * self.db

    def printLayerProperties(self):
        print("weights ", self.w.shape)
        print("biases ", self.b.shape)
        print("ouput-a", self.a.shape)
        print("deltaWeights", self.dw.shape)
        print("deltaBiases", self.db.shape)
        print("delta-a", self.da.shape)

    def printLayerdelta(self):
        print("Biases", self.b)
