def add_linear_activation(model):
    if(model.layers[-1][0]):
        model.layers.append([False,"linear"])
    for i in range(len(model.layers)):
        if(i != 0):
            if(model.layers[i-1][0]):
                if(model.layers[i][0]):
                    model.layers.insert(i,[False,"linear"])
    return 0      
 