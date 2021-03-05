import numpy as np
from Layer import Layer


def Layer_initializations(layer_dim):
    np.random.seed(100)
    Layers = []
    Length = len(layer_dim)

    for i in range(1, Length):
        Layers.append(Layer(layer_dim[i], layer_dim[i - 1]))

    return Layers


# Test
test_parameters = Layer_initializations([784, 784, 10])
print(test_parameters)
