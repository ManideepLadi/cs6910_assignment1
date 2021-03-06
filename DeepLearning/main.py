from GradientDescent import StochasticGradientDescent
from Layer_initializations import Layer_initializations
from backwardPropogation import backwardPropagation
from crossEntropyLoss import crossEntropyLoss
from forwardPropagation import feedForward
from keras.datasets import fashion_mnist
import numpy as np
from numpy import random


def train_model(X_train, y_train, epoch, layer_dim, learning_rate, NoOfLayers):
    # Step 1: Initialize parameters
    Layers = Layer_initializations(layer_dim)

    for i in range(1, epoch):
        index = random.randint(59999)
        # Forward propagation
        hL, Layers = feedForward(X_train[index:index + 1].T, Layers)
        # Calculate cost
        cost = crossEntropyLoss(hL, y_train[index:index + 1])

        # backward Propogation
        da_cur = - (y_train[index:index + 1].T - hL)
        Layers = backwardPropagation(da_cur, X_train[index:index + 1], NoOfLayers, Layers)
        # Update weights and biases according to stocahstic gradient descent
        Layers = StochasticGradientDescent(Layers, NoOfLayers, learning_rate)
    return Layers


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    # plot_images()

    # Preprocessing
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = np.reshape(X_train, (X_train.shape[0], 28 * 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 28 * 28))
    # converting output vectors to one-hot encoders
    y_train = np.eye(len(labels))[y_train]
    y_test = np.eye(len(labels))[y_test]
    NoOfLayers = 2
    layerDim = [784, 100, 10]
    Layers = train_model(X_train, y_train, 20, layerDim, 0.0001, NoOfLayers)
    for i in range(1000, 1001):
        hL, Layers = feedForward(X_test[i:i + 1].T, Layers)
        print(np.argmax(hL))
        print(np.argmax(y_test[i:i+1]))
