architecture_ = [2,2]
activation_func = "linear" # linear etc.
precision = "int"
mode = 'training' #/ inference / training + inference
output_type = 'prediction class' #/ predictions + final weights / predictions + loss etc
learning_rate_type = "fixed" #/ adjustable
learning_rate = 0.1
test = True
test_file = 'nn_test.py'
file_string = ''
func_declerations = '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define POLY_MASK_32 0xB4BCD35C
#define POLY_MASK_31 0x7A5BC2E3
int lfsr32, lfsr31;
'''
file_string += func_declerations
input_shape = architecture_[0]
output_shape= architecture_[-1]
bram_counts = len(architecture_)

for i in range(len(architecture_)):
    if(i==0):
        file_string += "#define number_of_inputs " + str(architecture_[0]) + '\n'
    elif(i == len(architecture_)-1):
        file_string += "#define number_of_outputs " + str(architecture_[i]) + '\n'
    else:
        file_string += "#define layer_" +  str(i) + "_nodes " + str(architecture_[i]) + '\n'

if(learning_rate_type == "fixed"):
    learning_rate_string = "const float lr = " + str(learning_rate) +";" + '\n'
    file_string += learning_rate_string

input_matrix =  precision + " input_matrix[" + str(architecture_[0]) + "];" + '\n'
file_string += input_matrix
real_output_matrix = precision + " real_output_matrix[" + str(architecture_[-1]) + "];" + '\n'
real_output_matrix += precision + " output_matrix[" + str(architecture_[-1]) + "];" + '\n'
file_string += real_output_matrix
for i in range(len(architecture_)-1):
    weight_matrix = precision + " layer_" + str(i+1) +'_weights[' + str(architecture_[i]*architecture_[i+1]) + "];" + '\n'
    file_string += weight_matrix
    bias_matrix =  precision + " layer_" + str(i+1) +'_bias[' + str(architecture_[i]) + "];" + '\n'
    file_string += bias_matrix
    output_matrix = precision + " layer_" + str(i+1) +'_output[' + str(architecture_[i+1]) + "];" + '\n'
    file_string += output_matrix
    delta_matrix = precision +" layer_" + str(i+1) +'_delta_out[' + str(architecture_[i+1]) + "];" + '\n'
    file_string += delta_matrix



if(activation_func == "sigmoid"):
    related_functions = precision+ " activate( "+precision+" x) { return 1 / (1 + exp(-x)); }\n" + precision +" dactivate( "+precision+" x) { return x * (1 - x); }"
      
elif(activation_func == "linear"):
    related_functions = precision+ " activate( "+precision+" x) { return x; }\n" + precision +" dactivate( "+precision+" x) { return 1; }"
else:
    raise Exception("Missing activation function info!!....")



related_functions += """
int shift_lfsr(int *lfsr,  int polymonial_mask)
{
    int feedback;

    feedback = *lfsr & 1;
    *lfsr >>= 1;
    if (feedback == 1)
        *lfsr ^= polymonial_mask;
    return *lfsr;
}

void init_lfsrs(void)
{
    lfsr32 = 0xABCDE; //seed values
    lfsr31 = 0x23456789;
}

int get_random(void)
{
    /*this random number generator shifts the 32-bit LFSR twice before XORing
      it with the 31-bit LFSR. the bottom 16 bits are used for the random number*/
    shift_lfsr(&lfsr32, POLY_MASK_32);
    return(shift_lfsr(&lfsr32, POLY_MASK_32) ^ shift_lfsr(&lfsr31, POLY_MASK_31));
}



void dense_layer_forward("""+precision+""" * layer_input, """+precision+""" * layer_bias, """+precision+""" * layer_weights , """+precision+""" * layer_output ,int number_of_input_nodes, int number_of_output_nodes){
   
   for (int i=0; i<number_of_output_nodes; i++) {
      
      """+precision+""" activation = *(layer_bias+i);
      for (int j=0; j<number_of_input_nodes; j++) {

         activation += *(layer_weights + number_of_output_nodes*i + j) * (*(layer_input + i));
      
      }
      *(layer_output + i)= activate(activation);
      
   }
   
}

void calculate_delta("""+precision+""" * delta_input_real, """+precision+""" * delta_input_calculated,"""+precision+""" * layer_weights,  """+precision+""" * delta_output, int delta_output_num,int delta_input_num,int last_layer){
   
   if(last_layer){

      for (int i=0; i<delta_output_num; i++) {

         *(delta_output + i)= *(delta_input_real+i) - *(delta_input_calculated+i);

         *(delta_output + i)= *(delta_output + i) * dactivate(*(delta_input_calculated+i));
      }

   }

   else{

      for (int i=0; i<delta_output_num; i++) {
         float errorHidden = 0.0;
         for(int j=0; j<delta_input_num; j++) {
            // delta_input_layer == deltaoutput of the next layer
            // delta_input_calculated == 
            errorHidden += *(delta_input_real+j) *  (*(layer_weights+ i*delta_output_num + j));
         }
         *(delta_output+i) = errorHidden * dactivate(*(delta_input_calculated+i));
      }


   }

}

void change_weights(float * layer_input,float * layer_delta, float * layer_weight, float * layer_bias,int number_of_input_nodes, int number_of_output_nodes, float lr){
   
   for (int i=0; i<number_of_output_nodes; i++) {
     
      *(layer_bias+i) += *(layer_delta+i) * lr;
      for (int j=0; j<number_of_input_nodes; j++) {
         
         *(layer_weight + j*number_of_output_nodes+i) += *(layer_input+j) * (*(layer_delta+i)) * lr;
      }
   }
}


"""
file_string += related_functions

main_function_header = "void deeplearn( "
for i in range(bram_counts -1):
    main_function_header += precision +" MEM_" +str(i+1) + "[" + str(architecture_[i]*architecture_[i+1] + architecture_[i+1]) + '],'
#main_function_header += "float[" + str(input_shape) + "] input_array,"
#main_function_header += "float[" + str(output_shape) + "] output_array,"
if(learning_rate_type == "adjustable"):
    main_function_header += "float learning_rate,"
for i in range(architecture_[0]):
    main_function_header += "int input_var_" + str(i+1)+" ,"
main_function_header += "int mode_var ,"


if(output_type == "prediction class"):
    main_function_header += "int real_output_var ,int& output_var"
if(output_type == "prediction_and_loss"):
    main_function_header += "int real_output_var ,int& output_var , float& loss_val"

main_function_header += """){
#pragma HLS INTERFACE ap_ctrl_none port=return \n
    """
file_string += main_function_header

input_output_pragmas = ''
for i in range(architecture_[0]):
    input_output_pragmas += "#pragma HLS INTERFACE s_axilite port= " + "input_var_" + str(i+1) + '\n'
input_output_pragmas += "#pragma HLS INTERFACE s_axilite port= mode_var" + '\n'
if(output_type == "prediction class"):
    input_output_pragmas += "#pragma HLS INTERFACE s_axilite port= real_output_var"+ '\n'
    input_output_pragmas += "#pragma HLS INTERFACE s_axilite port= output_var"+ '\n'
if(output_type == "prediction_and_loss"):
    input_output_pragmas += "#pragma HLS INTERFACE s_axilite port= real_output_var"+ '\n'
    input_output_pragmas += "#pragma HLS INTERFACE s_axilite port= output_var"+ '\n'
    input_output_pragmas += "#pragma HLS INTERFACE s_axilite port= loss_val"+ '\n'
file_string += input_output_pragmas


memory_pragmas = ''
for i in range(bram_counts -1):
    memory_pragmas += "#pragma HLS INTERFACE bram port =  MEM_" +str(i+1) + '\n'
file_string += memory_pragmas

init_lfsr = "init_lfsrs();" + '\n'
file_string += init_lfsr 

input_matrix_string = ""
for i in range(architecture_[0]):
    input_matrix_string += "input_matrix["+str(i)+"]= input_var_" + str(i+1)+";" + '\n'
file_string += input_matrix_string

real_output_matrix_string = ""
for i in range(architecture_[-1]):
    real_output_matrix_string += "real_output_matrix["+str(i)+"]= 0;" + '\n'
real_output_matrix_string += "real_output_matrix[real_output_var]= 1;" + '\n'
file_string += real_output_matrix_string


mode_string = "if(mode_var==0){"+ '\n'
file_string += mode_string

initialization_string = ''
for i in range(len(architecture_)-1):
    initialization_string += "for(int i = 0; i <"+str(architecture_[i]*architecture_[i+1])+"; i++){\n"
    initialization_string += "layer_" + str(i+1) +"_weights[i]  = (get_random()% 100) /100 ;" + '\n'
    initialization_string += "MEM_" + str(i+1) +"[i] = layer_" + str(i+1) +"_weights[i];" + '\n }\n'

    initialization_string += "for(int i = 0; i <"+str(architecture_[i])+"; i++){\n"
    initialization_string += "layer_" + str(i+1) +"_bias[i]  = (get_random()% 100) /100 ;" + '\n'
    initialization_string += "MEM_" + str(i+1) +"["+str(architecture_[i]*architecture_[i+1])+"+i] = layer_" + str(i+1) +"_bias[i];" + '\n }\n'


    initialization_string += "for(int i = 0; i <"+str(architecture_[i+1])+"; i++){\n"
    initialization_string += "layer_" + str(i+1) +"_output[i]  = 0;" + '\n'
    initialization_string += "layer_" + str(i+1) +"_delta_out[i]  = 0;" + '\n }\n'
  
file_string += initialization_string
initialization_end_string = """

output_var = 0.1;

}
"""
file_string += initialization_end_string










mode_string = "if(mode_var==1){"+ '\n'
file_string += mode_string


initialization_string = ''
for i in range(len(architecture_)-1):
    initialization_string += "for(int i = 0; i <"+str(architecture_[i]*architecture_[i+1])+"; i++){\n"
    initialization_string += "layer_" + str(i+1) +"_weights[i] = MEM_" + str(i+1) +"[i]  ;" + '\n }\n'

    initialization_string += "for(int i = 0; i <"+str(architecture_[i])+"; i++){\n"
    initialization_string += "layer_" + str(i+1) +"_bias[i] = MEM_" + str(i+1) +"["+str(architecture_[i]*architecture_[i+1])+"+i];" + '\n }\n'


    initialization_string += "for(int i = 0; i <"+str(architecture_[i+1])+"; i++){\n"
    initialization_string += "layer_" + str(i+1) +"_output[i]  = 0;" + '\n'
    
  
file_string += initialization_string

dense_layer_string = ""
for i in range(len(architecture_)-1):
    if(i==0):
        dense_layer_string += "dense_layer_forward(input_matrix,layer_"+str(i+1)+"_bias,layer_"+str(i+1)+"_weights,layer_"+str(i+1)+"_output,"+str(architecture_[i])+","+str(architecture_[i+1])+");"+ '\n'
    elif(i== len(architecture_)-2):
        dense_layer_string +=   "dense_layer_forward(layer_"+str(i)+"_output,layer_"+str(i+1)+"_bias,layer_"+str(i+1)+"_weights,layer_"+str(i+1)+"_output,"+str(architecture_[i])+","+str(architecture_[i+1])+");"+ '\n'
    else:
        dense_layer_string += "dense_layer_forward(layer_"+str(i)+"_output,layer_"+str(i+1)+"_bias,layer_"+str(i+1)+"_weights,layer_"+str(i+1)+"_output,"+str(architecture_[i])+","+str(architecture_[i+1])+");"+ '\n'

file_string += dense_layer_string

inference_end_string = """
}
"""
file_string += inference_end_string












mode_string = "if(mode_var==2){"+ '\n'
file_string += mode_string


initialization_string = ''
for i in range(len(architecture_)-1):
    initialization_string += "for(int i = 0; i <"+str(architecture_[i]*architecture_[i+1])+"; i++){\n"
    initialization_string += "layer_" + str(i+1) +"_weights[i] = MEM_" + str(i+1) +"[i]  ;" + '\n }\n'

    initialization_string += "for(int i = 0; i <"+str(architecture_[i])+"; i++){\n"
    initialization_string += "layer_" + str(i+1) +"_bias[i] = MEM_" + str(i+1) +"["+str(architecture_[i]*architecture_[i+1])+"+i];" + '\n }\n'


    initialization_string += "for(int i = 0; i <"+str(architecture_[i+1])+"; i++){\n"
    initialization_string += "layer_" + str(i+1) +"_output[i]  = 0;" + '\n'
    initialization_string += "layer_" + str(i+1) +"_delta_out[i]  = 0;" + '\n }\n'
  
file_string += initialization_string

dense_layer_string = ""
for i in range(len(architecture_)-1):
    if(i==0):
        dense_layer_string += "dense_layer_forward(input_matrix,layer_"+str(i+1)+"_bias,layer_"+str(i+1)+"_weights,layer_"+str(i+1)+"_output,"+str(architecture_[i])+","+str(architecture_[i+1])+");"+ '\n'
    elif(i== len(architecture_)-2):
        dense_layer_string +=   "dense_layer_forward(layer_"+str(i)+"_output,layer_"+str(i+1)+"_bias,layer_"+str(i+1)+"_weights,layer_"+str(i+1)+"_output,"+str(architecture_[i])+","+str(architecture_[i+1])+");"+ '\n'
    else:
        dense_layer_string += "dense_layer_forward(layer_"+str(i)+"_output,layer_"+str(i+1)+"_bias,layer_"+str(i+1)+"_weights,layer_"+str(i+1)+"_output,"+str(architecture_[i])+","+str(architecture_[i+1])+");"+ '\n'

file_string += dense_layer_string


calculate_delta_layer_string = ""
for i in range(len(architecture_)-2, -1, -1):
    if(i==len(architecture_)-2):
        calculate_delta_layer_string += "calculate_delta(real_output_matrix,layer_"+str(i+1)+"_output,layer_1_weights,layer_"+str(i+1)+"_delta_out,"+str(architecture_[i+1])+","+str(architecture_[i+1])+",1);"+ '\n'
    else:
        calculate_delta_layer_string += "calculate_delta(layer_"+str(i+2)+"_delta_out,layer_"+str(i+1)+"_output,layer_"+str(i+2)+"_weights,layer_"+str(i+1)+"_delta_out,"+str(architecture_[i+2])+","+str(architecture_[i+1])+",0);"+ '\n'

file_string += calculate_delta_layer_string



change_weigts_string = ""
for i in range(len(architecture_)-1):
    if(i==0):
        change_weigts_string += "change_weights(input_matrix,layer_"+str(i+1)+"_delta_out,layer_"+str(i+1)+"_weights,layer_"+str(i+1)+"_bias,"+str(architecture_[i])+","+str(architecture_[i+1])+",lr);"+ '\n'
    elif(i== len(architecture_)-2):
        change_weigts_string +=   "change_weights(layer_"+str(i)+"_output,layer_"+str(i+1)+"_delta_out,layer_"+str(i+1)+"_weights,layer_"+str(i+1)+"_bias,"+str(architecture_[i])+","+str(architecture_[i+1])+",lr);"+ '\n'
    else:
        change_weigts_string += "change_weights(layer_"+str(i)+"_output,layer_"+str(i+1)+"_delta_out,layer_"+str(i+1)+"_weights,layer_"+str(i+1)+"_bias,"+str(architecture_[i])+","+str(architecture_[i+1])+",lr);"+ '\n'

file_string += change_weigts_string


save_memory_string = ""

for i in range(len(architecture_)-1):
    save_memory_string += "for(int i = 0; i <"+str(architecture_[i]*architecture_[i+1])+"; i++){\n"
    save_memory_string += "MEM_" + str(i+1) +"[i] = layer_" + str(i+1) +"_weights[i];" + '\n }\n'

    save_memory_string += "for(int i = 0; i <"+str(architecture_[i])+"; i++){\n"
    save_memory_string += "MEM_" + str(i+1) +"["+str(architecture_[i]*architecture_[i+1])+"+i] = layer_" + str(i+1) +"_bias[i];" + '\n }\n'

file_string += save_memory_string

final_output_string = """
                float loss_vall = 0;
				   for(int i = 0; i < number_of_outputs; i++){
					   loss_vall += real_output_matrix[i] - output_matrix[i] ;
				   	}
				   output_var = loss_vall;
"""
file_string += final_output_string

training_end_string = """
}
"""
file_string += training_end_string



deeplearn_end_string = """
}
"""
file_string += deeplearn_end_string
print(file_string)

f = open("deneme.txt", "a")
f.write(file_string)
f.close()






