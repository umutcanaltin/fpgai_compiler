from onnx_handler.onnx_reader import get_model_weights,get_model_arch,verify_model
from architectures.convolution_layer import ConvolutionLayer
from architectures.dense_layer import DenseLayer
from implementations.implementations import dense_layer_imp, conv_layer_imp
import onnx

class fpgai_engine():
    def __init__(self,learning_rate= 0.1,mode="inference",onnx_file_name = "my_image_classifier.onnx", precision = "float",vitis_hls_location= "default",
                 hls_project_name="default",hls_solution_name="default"):
        self.main_func_name = "deeplearn"
        self.precision = precision
        self.number_of_input_nodes = 4
        self.number_of_output_nodes = 10
        self.learning_rate = learning_rate
        self.mode = mode
        self.onnx_file_name = onnx_file_name
        self.model = onnx.load(self.onnx_file_name)
        verify_str = self.verify_onnx_model()
        if (verify_str!= 1):
            raise Exception(verify_str)
        self.layers = self.get_model_arch()
        self.weights= self.get_weights()
        self.obj_arch_rep = []
        self.layer_function_implementations = ""
        self.add_linear_activation()
        self.generate_obj_rep()
        print(self.layers)
        print(self.generate_hls_codes())
      
    def verify_onnx_model(self):
        return verify_model(self.model)
    
    def create_test_branch_cpp(self):
        pass

    def get_model_arch(self):
        return get_model_arch(self.model)
    
    def get_weights(self):
        return get_model_weights(self.model)
    
    def add_linear_activation(self):
     
        if(self.layers[-1][0]):
            self.layers.append([False,"linear"])
        for i in range(len(self.layers)):
            if(i != 0):
                if(self.layers[i-1][0]):
                    if(self.layers[i][0]):
                        self.layers.insert(i,[False,"linear"])
        return 0      
    
    def generate_obj_rep(self):
        for i in range(len(self.layers)):
            first_layer= True
            first_dense_layer = True
            first_conv_layer = True
            if(self.layers[i][0]):
                if(i!=0):
                    first_layer = False
                layer_weights = self.weights[i]
                layer_bias = self.weights[i+1]
                #skip the activation layer
                if(self.layers[i][1] == "conv"):
                    if(first_conv_layer):
                        new_implementation = conv_layer_imp().get_hls_function()
                        self.layer_function_implementations += new_implementation
                        first_conv_layer = False
                    #self.obj_arch_rep.append(ConvolutionLayer(weights=layer_weights,bias=layer_bias,is_first_layer=first_layer, activation_function=self.layers[i+1][1]))
                elif(self.layers[i][1]== "dense"):
                    if(first_dense_layer):
                        new_implementation = dense_layer_imp().get_hls_function()
                        self.layer_function_implementations += new_implementation
                        first_dense_layer = False 
                    new_layer = DenseLayer(ai_model=self, activation_function=self.layers[i+1][1], weights=layer_weights,bias=layer_bias, is_first_layer=first_layer, name_of_layer="layer_"+str(int(i/2)))
                    self.obj_arch_rep.append(new_layer)
                else:
                    raise Exception
            else:
                pass
        return 0

          

    
    def compile_hls_codes(self):
        return 0
    
    def generate_hls_codes(self):
        generated_hls_codes = self.layer_function_implementations
        for i in range(len(self.obj_arch_rep)):
            generated_hls_codes += self.obj_arch_rep[i].get_hls_file_string()
       
        generated_hls_codes += self.construct_main_function()
        generated_hls_codes += self.get_pragmas()
        return generated_hls_codes         

            
    def construct_main_function(self):
        main_func_string = ""
        main_func_string += "void " + self.main_func_name + "( "
        for i in range(len(self.obj_arch_rep)):
            if(isinstance(self.obj_arch_rep[i], ConvolutionLayer)):
                main_func_string += self.precision + " MEM_"+ str(i) +"["+self.obj_arch_rep[i]
                #dolacak

            if(isinstance(self.obj_arch_rep[i], DenseLayer)):
                main_func_string += self.precision + " MEM_"+ str(i) +"["+str(len(self.obj_arch_rep[i].weights)*len(self.obj_arch_rep[i].weights[0]) + len(self.obj_arch_rep[i].bias)) + "],"
                
        if(self.mode == "inference"):
            main_func_string += self.precision+"& output_var){" + "\n"
        if(self.mode == "training"):
            main_func_string += " int mode_var, "
            main_func_string += self.precision+"& output_var{"+ "\n"
                    #todo list input eklenecek
        return main_func_string
    

    def get_pragmas(self):
        pragma_string = "#pragma HLS INTERFACE ap_ctrl_none port=return \n"
        for i in range(len(self.obj_arch_rep)):
            pragma_string += "#pragma HLS INTERFACE bram port =  MEM_" + str(i) + "\n"
        return pragma_string


    
    def compile_hls_test(self):
        return 0


