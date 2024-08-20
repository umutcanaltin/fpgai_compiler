class dense_layer_imp():
    def __init__(self,precision = "float",name_of_layer="default_dense_name"):
        self.precision= precision
        self.name_of_layer = name_of_layer
    def get_forward_hls_function(self):
        
        forward_pass = "void dense_forward_"+self.name_of_layer+"( " +self.precision+" * layer_input, " + self.precision + "* layer_bias, "+self.precision+" * layer_weights , "+self.precision+""" * layer_output ,int number_of_input_nodes, int number_of_output_nodes){

   for (int i=0; i<number_of_output_nodes; i++) {

      int activation = *(layer_bias+i);
      for (int j=0; j<number_of_input_nodes; j++) {

         activation += *(layer_weights + number_of_output_nodes*i + j) * (*(layer_input + i));

      }
      *(layer_output + i)= activate_"""+self.name_of_layer+"""(activation);

   }

}
        """
        return forward_pass
    

    
class conv_layer_imp():
    def __init__(self,precision = "float",name_of_layer="default_conv_name"):
        self.precision= precision
        self.name_of_layer = name_of_layer
    def get_forward_hls_function(self):
        forward_pass = "void conv_forward"+self.name_of_layer+"( "+self.precision+" * input, "+self.precision+"  * filters, int filter_number, int filter_size_1, int filter_size_2,int input_size_1, int input_size_2, "+self.precision+""" * output){
    for(int filter_index = 0 ; filter_index < filter_number ;  filter_index++ ){
      
        for(int conv_x_index=0 ; conv_x_index < input_size_1 - filter_size_1 + 1 ; conv_x_index ++ ){ 
            for(int conv_y_index=0 ; conv_y_index < input_size_2 - filter_size_2 + 1 ; conv_y_index ++ ){
           
            float conv_sum = 0;

                for(int kernel_x_index=0 ; kernel_x_index < filter_size_1 ; kernel_x_index ++ ){  
                    for(int kernel_y_index =0; kernel_y_index < filter_size_2 ; kernel_y_index++){ 
                        conv_sum += *(filters + kernel_y_index + filter_size_2*kernel_x_index)* *(input + conv_y_index + kernel_y_index + input_size_2*kernel_x_index);
                        
                    } 
                }
            *(output + conv_y_index  + conv_x_index*input_size_2) = activate_"""+self.name_of_layer+"""(conv_sum); 
            }
        }
    } 
}
        """
        return forward_pass
    