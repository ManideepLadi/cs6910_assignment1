def StochasticGradientDescent(Layers, NoOfLayers, learning_rate):
    for i in (range(0, NoOfLayers)):
        Layers[i].w = Layers[i].w - learning_rate * Layers[i].dw
        Layers[i].b = Layers[i].b - learning_rate * Layers[i].db
        Layers[i].printLayerProperties()

    return Layers
