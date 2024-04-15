from onnx_handler.onnx_reader import get_model_weights,get_model_arch,verify_model
from architectures.convolution_layer import ConvolutionLayer
from architectures.dense_layer import DenseLayer
import onnx

class fpgai_engine():
    def __init__(self,onnx_file_name = "my_image_classifier.onnx", precision = "float64",vitis_hls_location= "default",hls_project_name="default",hls_solution_name="default"
                 ,hardware_optimization = False, use_BRAM = True, use_DMA = True,user_DDR= True,memory_option_weights="default"):
        self.use_BRAM = use_BRAM
        self.use_DMA = use_DMA
        self.hardware_optimization = hardware_optimization
        self.onnx_file_name = onnx_file_name
        self.model = onnx.load(self.onnx_file_name)
        verify_str = self.verify_onnx_model()
        if (verify_str!= 1):
            raise Exception(verify_str)
        self.layers = self.get_model_arch()
        self.weights= self.get_weights()
        self.obj_arch_rep = []
        self.generate_obj_rep()
      
        
        
    def verify_onnx_model(self):
        return verify_model(self.model)

    def get_model_arch(self):
        return get_model_arch(self.model)
    
    def get_weights(self):
        return get_model_weights(self.model)
    
    def add_linear_activation(self):
     
        if(not self.layers[-1][0]):
            self.layers.append([False,"linear"])
        for i in range(len(self.layers)):
            if(i != 0):
                if(self.layers[i-1][0]):
                    if(self.layers[i][0]):
                        pass      
    
    def generate_obj_rep(self):
        print(self.layers)
        for i in range(len(self.layers)):
            #print(self.layers[i])
            if(self.layers[i][0]):
                layer_weights = self.weights[i]
                layer_bias = self.weights[i+1]
                #skip the activation layer
                if(self.layers[i][1] == "conv"):

                    self.obj_arch_rep.append(ConvolutionLayer(weights=layer_weights,bias=layer_bias))
                elif(self.layers[i][1]== "dense"):
                    self.obj_arch_rep.append(DenseLayer())
                else:
                    raise Exception
            else:
                pass
        return 0

          

    
    def compile_hls_codes(self):
        return 0
    
    def generate_hls_test_codes(self):
        return 0
    
    def compile_hls_test(self):
        return 0


