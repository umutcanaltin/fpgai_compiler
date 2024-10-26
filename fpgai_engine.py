from onnx_handler.onnx_reader import get_model_weights,get_model_arch,verify_model
import onnx
from utils.main_function_handler import construct_main_function
from utils.generate_object_rep import generate_obj_rep
from utils.get_nodes import get_number_of_input_nodes,get_number_of_output_nodes
from utils.get_pragma import get_pragmas
from utils.write_cpp_file import write_cpp_file,write_header_file, write_tcl_file
from utils.read_stream import read_input_stream
from utils.write_stream import write_output_stream
from utils.add_linear_act import add_linear_activation
from utils.vitis_tcl import vitis_tcl_generator
from utils.vivado_tcl import vivado_tcl_generator
from utils.testbench import generate_testbench_codes
import os
import numpy as np
from onnx_inference import onnx_inference_pytorch, onnx_train_pytorch

class fpgai_engine():
    def __init__(self,learning_rate= 0.1,mode="inference",onnx_file_name = "mlp.onnx", precision = "float"):
        self.main_func_name = "deeplearn"
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
        self.strides = []
        self.kernels = []
        self.weights= self.get_weights()
        self.layer_function_implementations = ""
        self.first_layer_shape= [5,5]

        add_linear_activation(self)
        generate_obj_rep(self)
        print(self.layers)
        self.number_of_input_nodes = get_number_of_input_nodes(self)
        self.number_of_output_nodes = get_number_of_output_nodes(self)

        generated_hls_code = self.generate_hls_codes()
        generated_header_code = self.generate_header_codes()
        generated_vitis_tcl = self.generate_vitis_tcl_codes()
        generated_vivado_tcl = self.generate_vivado_tcl_codes()


        input_shape = (5, 5)
        increment = 0.1
        total_elements = np.prod(input_shape)
        input_data = np.arange(0.1, total_elements * increment + 0.1, increment, dtype=np.float32)
        input_data = input_data.reshape(1, 1, 5,5)
        onnx_inference_pytorch(onnx_file_name,input_data=input_data)
        

        output_shape = (1, 10)
        increment = 0.1
        total_elements = np.prod(output_shape)
        target_output = np.arange(0.1, total_elements * increment + 0.1, increment, dtype=np.float32)
        target_output = target_output.reshape(1, 1, 1,10)

        target_output = target_output.flatten()
        input_data = input_data.flatten()


        generated_testbench_code = generate_testbench_codes(input_data= input_data, output_file_dest= os.getcwd()+"/generated_files", target_output=target_output,model=self)

        write_cpp_file("generated_files/main",generated_hls_code )
        write_cpp_file("generated_files/testbench", generated_testbench_code)
        write_header_file("generated_files/deeplearn",generated_header_code)
        write_tcl_file("generated_files/tcl_for_vitis",generated_vitis_tcl)
        write_tcl_file("generated_files/tcl_for_vivado",generated_vivado_tcl)

      
    def verify_onnx_model(self):
        return verify_model(self.model)
    

    def get_model_arch(self):
        return get_model_arch(self.model)
    
    def get_weights(self):
        return get_model_weights(self.model)
    

    def generate_hls_codes(self):
        generated_hls_codes = ""
        for i in range(len(self.obj_arch_rep)):
            generated_hls_codes += self.obj_arch_rep[i].get_hls_file_string()
        generated_hls_codes += self.layer_function_implementations
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
            for i in range(len(self.obj_arch_rep)-1, -1, -1):
                generated_hls_codes += self.obj_arch_rep[i].get_update_weights_func()
        generated_hls_codes += write_output_stream(self)
        return generated_hls_codes         
    
    def generate_header_codes(self):
        generated_header_code = ""
        for i in range(len(self.obj_arch_rep)):
            generated_header_code += self.obj_arch_rep[i].get_header_file_string()
        return generated_header_code
    
    def generate_vitis_tcl_codes(self):
        generated_vitis_tcl = vitis_tcl_generator(src_file_dir= os.getcwd()+"/generated_files",project_name="compiler_hls_project",project_dir=os.getcwd()+"/generated_files/")
        return generated_vitis_tcl
    
    def generate_vivado_tcl_codes(self):
        generated_vivado_tcl = vivado_tcl_generator(project_name="compiler_vivado_project", project_dir=os.getcwd()+"generated_files/",ip_repo_dir=os.getcwd()+"generated_files/compiler_hls_project")
        return generated_vivado_tcl
        
    
    




