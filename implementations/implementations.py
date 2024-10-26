class dense_layer_imp():
    def __init__(self,precision = "float",name_of_layer="default_dense_name"):
        self.precision= precision
        self.name_of_layer = name_of_layer

    def get_forward_hls_function(self):
        
        forward_pass = "void dense_forward_"+self.name_of_layer+"( " +self.precision+" * layer_input, " + self.precision + "* layer_bias, "+self.precision+" * layer_weights , "+self.precision+""" * layer_output ,int number_of_input_nodes, int number_of_output_nodes){

    for (int i = 0; i < number_of_output_nodes; i++) {
        float activation = layer_bias[i];  // Apply bias to the activation
        for (int j = 0; j < number_of_input_nodes; j++) {
            activation += layer_weights[number_of_input_nodes * i + j] * layer_input[j];
        }
      *(layer_output + i)= activate_"""+self.name_of_layer+"""(activation);

   }

}
        """
        return forward_pass

    def get_last_delta_calc_hls_function(self):
        
        delta_calc = "void dense_delta_"+self.name_of_layer+"( " +self.precision+" * layer_output, " + self.precision + "* target_output, "+ self.precision+""" * layer_delta , int output_size){
            for (int i = 0; i < output_size; i++) {
                float error = target_output[i] - layer_output[i];
                layer_delta[i] = error * dactivate_"""+self.name_of_layer+"""(layer_output[i]);
            }
        }
        """
        return delta_calc 

    def get_delta_calc_hls_function(self):
        
        delta_calc = "void dense_delta_"+self.name_of_layer+"( " +self.precision+" * next_delta, " + self.precision + "* next_weights, "+ self.precision + "* layer_output, "+self.precision + """ * layer_delta ,int input_size, int output_size){
    for (int i = 0; i < input_size; i++) {
        layer_delta[i] = 0;
        for (int j = 0; j < output_size; j++) {
            layer_delta[i] += next_delta[j] * next_weights[j * input_size + i];
        }
        layer_delta[i] *= dactivate_"""+self.name_of_layer+"""(layer_output[i]);
    }
}
        """
        return delta_calc
    



    def get_update_weights_hls_function(self):
        
        update = "void dense_update_weights_"+self.name_of_layer+"( " +self.precision+" * layer_output, " + self.precision + "* layer_weights, "+ self.precision+" * layer_bias ,"+ self.precision+" * layer_delta, "   +""" int input_size, int output_size, float learning_rate){
                for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            layer_weights[i * input_size + j] += learning_rate * layer_delta[i] * layer_output[j];
        }
        layer_bias[i] += learning_rate * layer_delta[i];
    }
}
        """
        return update 

















    
class conv_layer_imp():
    def __init__(self,precision = "float",name_of_layer="default_conv_name"):
        self.precision= precision
        self.name_of_layer = name_of_layer
    def get_forward_hls_function(self):
        forward_pass = "void convolution_forward_"+self.name_of_layer+"( "+self.precision+" * input, "+self.precision+"  * filters,"+self.precision+" * layer_bias"+", int filter_number, int filter_size_1, int filter_size_2,int input_size_1, int input_size_2, "+self.precision+""" * output){
    // Loop over the number of filters
    for (int f = 0; f < filter_number; f++) {
        // Loop over the input feature map (output rows)
        for (int i = 0; i < input_size_1 - filter_size_1 + 1; i++) {
            for (int j = 0; j < input_size_2 - filter_size_2 + 1; j++) {



                float sum = *(layer_bias+f);

                for (int fi = 0; fi < filter_size_1; fi++) {
                    for (int fj = 0; fj < filter_size_2; fj++) {

                    	int filter_idx = (f * filter_size_1 * filter_size_2) + (fi * filter_size_2) + fj;
                        int input_idx = (i + fi) * input_size_2 + (j + fj);
                        sum += *(input + input_idx) * *(filters+filter_idx);
                    }
                }

                int output_idx = f * (input_size_1 - filter_size_1 + 1) * (input_size_2 - filter_size_2 + 1) + i * (input_size_2 - filter_size_2 + 1) + j;
                *(output + output_idx) = activate_""" + self.name_of_layer+"(sum);"
            
        forward_pass += """
            }
        }
    }
}
        """
        return forward_pass




    def get_last_delta_calc_hls_function(self):
        
        delta_calc = "void convolution_delta_"+self.name_of_layer+"( " +self.precision+" * layer_output, " + self.precision + "* target_output, "+ self.precision+""" * layer_delta , int output_size){
            for (int i = 0; i < output_size; i++) {
                float error = target_output[i] - layer_output[i];
                layer_delta[i] = error * dactivate_"""+self.name_of_layer+"""(*layer_output+i);
            }
        }
        """
        return delta_calc
    






    def get_delta_calc_hls_function(self):
        
        delta_calc = "void convolution_delta_"+self.name_of_layer+"( " +self.precision+" * next_delta, " + self.precision + "* next_weights, "+ self.precision + "* layer_output, "+self.precision + """ * layer_delta ,int input_height,int input_width, int output_height, int output_width,int filter_height, int filter_width){
     
    for (int i = 0; i < input_height * input_width; i++) {
        layer_delta[i] = 0.0f;
    }

    // Iterate over each spatial location in the output
    for (int oh = 0; oh < output_height; oh++) {
        for (int ow = 0; ow < output_width; ow++) {
            // Iterate over each position in the filter
            for (int fh = 0; fh < filter_height; fh++) {
                for (int fw = 0; fw < filter_width; fw++) {
                    // Calculate the corresponding input indices
                    int ih = oh + fh; // input height index
                    int iw = ow + fw; // input width index

                    // Check if the input indices are within bounds
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        // Update the delta for the corresponding input position
                        layer_delta[ih * input_width + iw] += 
                            next_delta[oh * output_width + ow] * next_weights[fh * filter_width + fw];
                    }
                }
            }

            // Apply the derivative of the activation function (if needed)
            layer_delta[oh * output_width + ow] *= dactivate_"""+self.name_of_layer+"""(layer_output[oh * output_width + ow]);
        }
    }
}
    
        """
        return delta_calc
    


    def get_update_weights_hls_function(self):
        
        update = "void convolution_update_weights_"+self.name_of_layer+"( " +self.precision+" * layer_output, " + self.precision + "* layer_weights, "+ self.precision+" * layer_bias ,"+ self.precision+" * layer_delta, "   +""" int filter_height, int filter_width, int input_height, int input_width, float learning_rate){
    int output_height = input_height - filter_height + 1;
    int output_width = input_width - filter_width + 1;

    // Loop over the output feature map (output dimensions)
    for (int i = 0; i < output_height; i++) {
        for (int j = 0; j < output_width; j++) {
            // Loop over each element in the filter
            for (int fh = 0; fh < filter_height; fh++) {
                for (int fw = 0; fw < filter_width; fw++) {
                    int weight_idx = fh * filter_width + fw;
                    int input_idx = (i + fh) * input_width + (j + fw);
                    layer_weights[weight_idx] += learning_rate * layer_delta[i * output_width + j] * layer_output[input_idx];
                }
            }
            // Update bias
            layer_bias[0] += learning_rate * layer_delta[i * output_width + j];
        }
    }
}

        """
        return update 
    















