import numpy as np


def backwardPropagation(da_cur, x, NoOfLayers, Layers):
    # have to pass as  parameter da_cur = - (y_train[index:index+1].T-hL.T)
    for i in reversed(range(1, NoOfLayers)):
        h_prev = Layers[i - 1].h
        a_prev = Layers[i - 1].a
        da_cur = Layers[i].backward(da_cur, h_prev, a_prev, 'sigmoid')

    Layers[0].w = np.dot(da_cur, x.T)
    Layers[0].db = da_cur
    return Layers
