class DenseLayer():
    def __init__(self, number_of_neurons = None ,input_shape= None, weights= None, decorelation=False, learning=False, bias = False, name_of_layer="dense_layer"):
        self.number_of_neurons = number_of_neurons
        self.input_shape = input_shape
        self.decorelation = decorelation
        self.learning = learning
        self.bias = bias
        self.weigts = weights
        self.name_of_layer = name_of_layer
    