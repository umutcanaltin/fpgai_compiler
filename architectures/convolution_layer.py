from activation.activation_functions import Activation_functions

class ConvolutionLayer():
    def __init__(self,ai_model,kernel_shape,stride,input_shape,precision = "float", weights = None, mode="inference", learning=False, bias = None, 
                 name_of_layer="convolution_layer", is_first_layer= False,is_last_layer=False, activation_function = "linear", inject_weights= False,layer_order=0): 
        
        self.learning = learning
        self.bias = bias
        self.weights = weights

        self.name_of_layer = name_of_layer
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer
        self.activation_function = activation_function
        self.layer_order = layer_order
        self.hls_file_string = ""
        self.hls_header_string = ""
        self.mode = mode
        self.precision = precision
        self.ai_model = ai_model
        self.inject_weights = inject_weights
        self.activation_function_obj = Activation_functions(activation_function=activation_function,name_of_layer = self.name_of_layer,precision=self.precision)
        self.kernel_shape = [int(kernel_shape[0]),int(kernel_shape[1])]
        self.stride = [int(stride[0]),int(stride[1])]
        self.input_shape = input_shape
        self.output_shape = self.get_output_shape()
        self.filter_num = self.get_filter_num()
    
    
    def get_hls_file_string(self):
        if(self.is_first_layer):
            self.hls_file_string += self.get_hls_first_layer_initializer()
        self.hls_file_string += self.get_hls_layer_initializers()
        self.hls_file_string += self.activation_function_obj.get_activation_function(mode=self.mode)

        return(self.hls_file_string)
    def get_header_file_string(self):
        if(self.is_first_layer):
            self.hls_header_string += self.get_header_first_layer_initializer()
        return(self.hls_header_string)
    
    def get_output_shape(self):
        first_dim = int((int(self.input_shape[0]) - int(self.kernel_shape[0]) )/ int(self.stride[0])) +1
        second_dim = int(int((self.input_shape[1]) - int(self.kernel_shape[1]))/ int(self.stride[1])) +1
        return [first_dim,second_dim]
    
    def get_filter_num(self):
        return self.weights.shape[1]
        

    
    def get_hls_learning_rate_initializer(self):
        return "const float learning_rate = "  + str(self.ai_model.learning_rate) + ";\n"
    

    def get_header_first_layer_initializer(self):
        first_layer_initializer = """
// deeplearn.h

        #ifndef DEEPLEARN_H
        #define DEEPLEARN_H
        #include <hls_stream.h>
        #include <ap_axi_sdata.h>
        #include <ap_int.h>
        typedef ap_axis<32, 2, 5, 6> floatSdCh;
        int export_weihts;

        """
        first_layer_initializer += "#define number_of_inputs " + str(self.ai_model.number_of_input_nodes) + "\n"
        first_layer_initializer += "#define number_of_outputs " + str(self.ai_model.number_of_output_nodes) + "\n"
        first_layer_initializer += self.get_hls_learning_rate_initializer()
        first_layer_initializer += "void deeplearn(hls::stream<floatSdCh> &inStream, hls::stream<floatSdCh> &outStream);\n#endif \n"
        return first_layer_initializer   
    
    def get_hls_first_layer_initializer(self):
        first_layer_initializer = '#include "deeplearn.h"  \n'
        return first_layer_initializer
    
    def get_hls_last_layer_initializer(self):
        last_layer_initializer = "int real_output_matrix["+str(self.ai_model.number_of_input_nodes)+"];\n"
        return last_layer_initializer

    def get_hls_layer_initializers(self):
        layer_initializer = ""
        if(self.mode =="inference"):
            if(self.is_first_layer):
                layer_initializer = self.precision+" "+self.name_of_layer+"_input["+str(self.input_shape[0]*self.input_shape[1])+"];\n"
            else:
                layer_initializer = ""
            layer_initializer += "int "+self.name_of_layer+"_filter_num ="+str(self.filter_num)+";\n"
            layer_initializer += self.precision+" "+self.name_of_layer+"_output["+str(self.output_shape[0]*self.output_shape[1])+"];\n"
            layer_initializer += self.precision +' '+ self.name_of_layer +'_weights[' + str(self.kernel_shape[0]* self.kernel_shape[1]) + "] = {"
            for i in range(len(self.weights[0][0])):
                for k in range(len(self.weights[0][0][i])):
                    layer_initializer += str(self.weights[0][0][i][k])
                    if(not (i== len(self.weights[0][0])-1 and k == len(self.weights[0][0][i])-1)):
                        layer_initializer += " ,"
            layer_initializer += "};\n"
           
            layer_initializer += self.precision +' '+ self.name_of_layer +'_bias[' + str(self.bias.shape[0]) + "] = {"
            for i in range(len(self.bias)):
                layer_initializer += str(self.bias[i])
                if(i!= len(self.bias)-1):
                    layer_initializer += " ,"
            layer_initializer += "};\n"
        if(self.mode == "training"):
            if(self.is_first_layer):
                layer_initializer = self.precision+" "+self.name_of_layer+"_input["+str(self.input_shape[0]*self.input_shape[1])+"];\n"
                layer_initializer += self.precision+" target_output["+str(self.ai_model.number_of_output_nodes)+"];\n"
                layer_initializer += self.precision+" error_buffer["+str(self.ai_model.number_of_output_nodes)+"];\n"
            else:
                layer_initializer = ""
            layer_initializer += self.precision+" "+self.name_of_layer+"_delta["+str(self.output_shape[0]*self.output_shape[1])+"];\n"            
            layer_initializer += "int "+self.name_of_layer+"_filter_num ="+str(self.filter_num)+";\n"
            layer_initializer += self.precision+" "+self.name_of_layer+"_output["+str(self.output_shape[0]*self.output_shape[1])+"];\n"
            layer_initializer += self.precision +' '+ self.name_of_layer +'_weights[' + str(self.kernel_shape[0]* self.kernel_shape[1]) + "] = {"
            for i in range(len(self.weights[0][0])):
                for k in range(len(self.weights[0][0][i])):
                    layer_initializer += str(self.weights[0][0][i][k])
                    if(not (i== len(self.weights[0][0])-1 and k == len(self.weights[0][0][i])-1)):
                        layer_initializer += " ,"
            layer_initializer += "};\n"
           
            layer_initializer += self.precision +' '+ self.name_of_layer +'_bias[' + str(self.bias.shape[0]) + "] = {"
            for i in range(len(self.bias)):
                layer_initializer += str(self.bias[i])
                if(i!= len(self.bias)-1):
                    layer_initializer += " ,"
            layer_initializer += "};\n"

        return layer_initializer
 
    def get_inference_func(self):
        layer_string = ""
        if(self.is_first_layer):
            layer_string += "convolution_forward_"+self.name_of_layer+"("+self.name_of_layer +"_input,"+self.name_of_layer +"_weights," +self.name_of_layer + "_bias,"+self.name_of_layer + "_filter_num,"+str(self.kernel_shape[0])+","+str(self.kernel_shape[1])+","+str(self.input_shape[0])+","+str(self.input_shape[1])+","+self.name_of_layer +"_output "+");"+ '\n'
        else:
            layer_string += "convolution_forward_"+self.name_of_layer+"("+self.ai_model.obj_arch_rep[self.layer_order-1].name_of_layer+"_output,"+self.name_of_layer +"_weights," +self.name_of_layer + "_bias,"+self.name_of_layer + "_filter_num,"+str(self.kernel_shape[0])+","+str(self.kernel_shape[1])+","+str(self.input_shape[0])+","+str(self.input_shape[1])+","+self.name_of_layer +"_output "+");"+ '\n'

        return layer_string
    
    def get_delta_calculation_func(self):
        calculate_delta_layer_string = ""
        if(self.is_last_layer):
            calculate_delta_layer_string += "calculate_loss("+self.name_of_layer+"_output, target_output, error_buffer,"+str(self.output_shape[0]*self.output_shape[1])+");"+ '\n'
            calculate_delta_layer_string += "convolution_delta_"+self.name_of_layer + "(error_buffer,"+self.name_of_layer+"_output,"+self.name_of_layer+"_delta,"+str(self.output_shape[0]*self.output_shape[1])+");"+ '\n'
        else:
            calculate_delta_layer_string += "convolution_delta_"+self.name_of_layer + "("+self.ai_model.obj_arch_rep[self.layer_order+1].name_of_layer +"_delta,"+self.ai_model.obj_arch_rep[self.layer_order+1].name_of_layer+"_weights,"+self.name_of_layer+"_output,"+self.name_of_layer+"_delta,"+str(self.input_shape[0])+", "+str(self.input_shape[1])+", "+str(self.output_shape[0])+", "+str(self.output_shape[1])+", "+str(self.kernel_shape[0])+", "+str(self.kernel_shape[1])+");"+ '\n'
        return calculate_delta_layer_string
       
    def get_update_weights_func(self):
        if(self.is_first_layer):
            update_weights_layer_string = "convolution_update_weights_"+self.name_of_layer + "("+self.name_of_layer +"_input,"+self.name_of_layer+"_weights,"+self.name_of_layer+"_bias,"+self.name_of_layer+"_delta,"+str(self.kernel_shape[0])+","+str(self.kernel_shape[1])+","+str(self.input_shape[0])+","+str(self.input_shape[1])+", learning_rate);"+ '\n'
        else:
            update_weights_layer_string = "convolution_update_weights_"+self.name_of_layer + "("+self.ai_model.obj_arch_rep[self.layer_order-1].name_of_layer  +"_output,"+self.name_of_layer+"_weights,"+self.name_of_layer+"_bias,"+self.name_of_layer+"_delta,"+str(self.kernel_shape[0])+","+str(self.kernel_shape[1])+","+str(self.input_shape[0])+","+str(self.input_shape[1])+", learning_rate);"+ '\n'
        return update_weights_layer_string
    

