class ConvolutionLayer():
    def __init__(self, number_of_kernels = None,input_shape = None, weights = None, decorelation=False, learning=False, bias = False, 
                 name_of_layer="dense_layer", is_first_layer= False, activation_function = "linear"):
        self.number_of_kernels = number_of_kernels
        self.input_shape = input_shape
        self.decorelation = decorelation
        self.learning = learning
        self.bias = bias
        self.weigts = weights
        self.name_of_layer = name_of_layer
        self.is_first_layer = is_first_layer
        self.activation_function = activation_function
        self.hls_file_string = ""

    def set_first_layer_declerations(self):
        pass


    def set_initial_declerations():
        pass
    
    def set_functions():
        pass

    def set_output():
        pass

    def set_test_functions():
        pass

    def get_hls_file_string(self):
        if(self.mode=="inference"):
            return 0

    