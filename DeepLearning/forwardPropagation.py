

def     feedForward(x, Layers):
    Length = len(Layers)
    h_prev = x
    # For 1 to N-1 layers, apply sigmoid activation function
    for i in range(0, Length-1):
        h = Layers[i].forward(h_prev, 'sigmoid')
        h_prev = h
    hL = Layers[Length-1].forward(h_prev, 'softmax')
    return hL,Layers