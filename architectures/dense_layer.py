from activation.activation_functions import Activation_functions
from utils.random_generator import Random_Matrix_Generator

class DenseLayer():
    def __init__(self,ai_model, precision = "float",number_of_kernels = None, weights = None, mode="train", learning=False, bias = False, 
                 name_of_layer="dense_layer", is_first_layer= False, activation_function = "linear", inject_weights= False): 
        self.learning = learning
        self.bias = bias
        self.weigts = weights
        self.name_of_layer = name_of_layer
        self.is_first_layer = is_first_layer
        self.activation_function = activation_function
        self.hls_file_string = ""
        self.mode = mode
        self.precision = precision
        self.ai_model = ai_model
        self.inject_weights = inject_weights
        self.activation_function_obj = Activation_functions(activation_function=activation_function,name_of_layer = self.name_of_layer,precision=self.precision)
        self.input_shape, self.output_shape = self.get_weigth_shape()
    
    def get_hls_file_string(self):
        if(self.is_first_layer):
            self.hls_file_string += self.get_hls_first_layer_initializer()
        self.hls_file_string += self.get_hls_layer_initializers()
        self.hls_file_string += self.activation_function_obj.get_activation_function()
        return(self.hls_file_string)
        
    def get_weigth_shape(self):
         input_shape = self.weigts.shape[1]
         output_shape = self.weigts.shape[0]
         return (input_shape, output_shape)
    
    def get_hls_learning_rate_initializer(self):
        return "const float lr = "  + str(self.ai_model.learning_rate) + ";\n"
    
    def get_hls_first_layer_initializer(self):
        first_layer_initializer = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
"""
        first_layer_initializer += "#define number_of_inputs " + str(self.ai_model.number_of_input_nodes) + "\n"
        first_layer_initializer += "#define number_of_outputs " + str(self.ai_model.number_of_output_nodes) + "\n"
        
        

        first_layer_initializer += self.get_hls_learning_rate_initializer()
        return first_layer_initializer
    
    def get_hls_last_layer_initializer(self):
        last_layer_initializer = "int real_output_matrix["+str(self.ai_model.number_of_input_nodes)+"];\n"
        return last_layer_initializer

         
        
    def get_hls_layer_initializers(self):
        layer_initializer = ""
        if(self.mode =="inference"):
            layer_initializer = "int input_matrix_"+self.name_of_layer+"["+str(self.input_shape)+"];\n"
            layer_initializer += "int output_matrix_"+self.name_of_layer+"["+str(self.output_shape)+"];\n"
        if(self.mode =="train"):
            layer_initializer = "int input_matrix_"+self.name_of_layer+"["+str(self.input_shape)+"];\n"
            layer_initializer += "int output_matrix_"+self.name_of_layer+"["+str(self.output_shape)+"];\n"
            layer_initializer += "int delta_output_matrix"+self.name_of_layer+"["+str(self.output_shape)+"];\n"

            if(self.inject_weights):
                layer_initializer += self.precision +' '+ self.name_of_layer +'_weights[' + str(self.input_shape* self.output_shape) + "];" + '\n'
            else:
                layer_initializer += self.precision +' '+ self.name_of_layer +'_weights[' + str(self.input_shape* self.output_shape) + "];" + '\n'


        return layer_initializer
    

    def get_input_shape(self):
        return self.weigts.shape()

    