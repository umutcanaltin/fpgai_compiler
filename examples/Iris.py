from architectures.convolution_layer import ConvolutionLayer
from architectures.dense_layer import DenseLayer
from implementations import dense_layer_imp, conv_layer_imp
from utils.random_generator import weight_generator
from utils.model_to import model_to_hls, model_to_cpp

class My_Model():
    def __init__(self):
      self.layers = []
      #Layers must be in order to compile codes!
      #Input and output shape information for the layer must be with weights argument!
      self.layers.append(ConvolutionLayer(weights= weight_generator(layer_type = "convolution", input_shape=, shape=(5,10,10),precision = "float")))
      # Convolution Layer 1
      self.layers.append(DenseLayer(activation_function = "relu", precision = "float",name_of_layer = 'my_first_dense_layer', weights= weight_generator(layer_type = "dense",precision = "float", input_shape= 100, output_shape = 10)))
      # Dense Layer 1 with relu activation

      dense_layer_3 = DenseLayer(activation_function = "linear",precision = "float")
      random_generatad_weights = np.random(100,10)
      dense_layer_3.inject_weights(weights = random_generatad_weights)
      self.layers.append(dense_layer_3)
      # Dense Layer 3 with linear activation
