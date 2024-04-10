def get_initial_arrays():
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

