from onnx_handlers.onnx_reader import get_model_weights,get_model_arch,verify_model
from architectures.convolution_layer import ConvolutionLayer
from architectures.dense_layer import DenseLayer
import onnx

class AI_Model():
    def __init__(self,onnx_file_name = "my_image_classifier.onnx", hardware_optimization = False, use_BRAM = True, use_DMA = True):
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
    
    def generate_obj_rep(self):
        for i in range(len(self.layers)):
            print("a")

          

    
    def compile_hls_codes(self):
        return 0
    
    def generate_hls_test_codes(self):
        return 0
    
    def compile_hls_test(self):
        return 0


