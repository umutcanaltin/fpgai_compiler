from onnx_handler.onnx_reader import get_model_weights,get_model_arch,verify_model
from architectures.convolution_layer import ConvolutionLayer
from architectures.dense_layer import DenseLayer
from implementations.implementations import dense_layer_imp, conv_layer_imp
import onnx
from utils.main_function_handler import construct_main_function
from utils.generate_object_rep import generate_obj_rep
from utils.get_nodes import get_number_of_input_nodes,get_number_of_output_nodes
from utils.get_pragma import get_pragmas
from utils.write_cpp_file import write_cpp_file
from utils.read_stream import read_input_stream
from utils.write_stream import write_output_stream
from utils.add_linear_act import add_linear_activation


#modes = inference , training
#precisions = int , float

class fpgai_engine():
    def __init__(self,learning_rate= 0.1,mode="inference",onnx_file_name = "my_image_classifier.onnx", precision = "float", quantization=False):
        self.main_func_name = "deeplearn"
        self.quantization = quantization
        if(self.quantization):
            precision = "int"
        self.precision = precision
        self.learning_rate = learning_rate
        self.mode = mode
        self.onnx_file_name = onnx_file_name
        self.obj_arch_rep = []
        self.model = onnx.load(self.onnx_file_name)
        verify_str = self.verify_onnx_model()
        if (verify_str!= 1):
            raise Exception(verify_str)
        self.layers = self.get_model_arch()
        self.weights= self.get_weights()
        self.layer_function_implementations = ""


        add_linear_activation(self)
        generate_obj_rep(self)
        self.number_of_input_nodes = get_number_of_input_nodes(self)
        self.number_of_output_nodes = get_number_of_output_nodes(self)
        generated_hls_code = self.generate_hls_codes()
        print(generated_hls_code)
        write_cpp_file("new_cpp_file",generated_hls_code )
      
    def verify_onnx_model(self):
        return verify_model(self.model)
    
    def create_test_branch_cpp(self):
        pass

    def get_model_arch(self):
        return get_model_arch(self.model)
    
    def get_weights(self):
        return get_model_weights(self.model, quantization=self.quantization)
    

    def generate_hls_codes(self):
        generated_hls_codes = self.layer_function_implementations
        for i in range(len(self.obj_arch_rep)):
            generated_hls_codes += self.obj_arch_rep[i].get_hls_file_string()
        generated_hls_codes += construct_main_function(self)
        generated_hls_codes += get_pragmas(self)
        generated_hls_codes += read_input_stream(self)
        if(self.mode == "inference"):
            for i in range(len(self.obj_arch_rep)):
                generated_hls_codes += self.obj_arch_rep[i].get_inference_func()
        if(self.mode == "training"):
            for i in range(len(self.obj_arch_rep)):
                generated_hls_codes += self.obj_arch_rep[i].get_inference_func()
            for i in range(len(self.obj_arch_rep)-1, -1, -1):
                generated_hls_codes += self.obj_arch_rep[i].get_delta_calculation_func()
            for i in range(len(self.obj_arch_rep)-1, 0, -1):
                generated_hls_codes += self.obj_arch_rep[i].get_change_weights_func()
        generated_hls_codes += write_output_stream(self)
        return generated_hls_codes         
    

    




