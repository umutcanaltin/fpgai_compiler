from architectures.dense_layer import DenseLayer
from architectures.convolution_layer import ConvolutionLayer 
def generate_testbench_codes(input_data,output_file_dest,float_to_int_constant = 10000000 ,target_output = None, model = None):
    input_data = input_data.flatten()
    test_code = """
#include "deeplearn.h"  // Make sure this includes the necessary definitions and typedefs
#include <iostream>
#include <iomanip>  // For std::fixed and std::setprecision
#include <fstream>  // For file operations

int main() {
    // Declare streams
    hls::stream<floatSdCh> inStream;
    hls::stream<floatSdCh> outStream;

    // Initialize an array of input data as integers \n"""
    test_code += "int input_data["
    if(target_output is None):
        test_code +="number_of_inputs] = {"
        for i in range(len(input_data)-1):
            test_code += str(int(input_data[i]*float_to_int_constant)) + ","
        test_code += str(int(input_data[-1]*float_to_int_constant)) + "};\n"
    else:
        test_code +="number_of_inputs+number_of_outputs+1] = {"
        for i in range(len(input_data)):
            test_code += str(int(input_data[i]*float_to_int_constant)) + ","
        for i in range(len(target_output)):
            test_code += str(int(target_output[i]*float_to_int_constant)) + ","
        test_code += str(0) + "};\n"
    test_code += """
  // Write the integer input data to the stream
    for (int i = 0; i < """
    if(target_output is None):
        test_code += "number_of_inputs"
    else:
        test_code += "number_of_inputs+number_of_outputs+1"
    test_code += """; i++) {
        floatSdCh valIn;
        valIn.data = ap_int<32>(input_data[i]);  // Set integer data, ap_int<32> is required to match the data type
        valIn.keep = -1;  // Keep all bytes
        valIn.strb = -1;  // All strobes valid
        valIn.user = 0;
        valIn.last = (i == number_of_inputs - 1) ? 1 : 0;
        valIn.id = 0;
        valIn.dest = 0;
        inStream.write(valIn);
    }

    // Call the top function (deeplearn)
    deeplearn(inStream, outStream);

    // Open a file to save the output
        
        """
    test_code +='std::ofstream output_file("'
    test_code +=output_file_dest
    test_code += """/hls_output.txt");


    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open the output file!" << std::endl;
        return 1;
    }

    // Retrieve the output and convert to float
    for (int i = 0; i <"""
    if(target_output is None):
        test_code +="number_of_outputs"

    else:
        output_num = 0
        for i in range(len(model.obj_arch_rep)):
            if(isinstance(model.obj_arch_rep[i], DenseLayer)):
                output_num += model.obj_arch_rep[i].input_shape * model.obj_arch_rep[i].output_shape + model.obj_arch_rep[i].output_shape
            if(isinstance(model.obj_arch_rep[i], ConvolutionLayer)):
                output_num += model.obj_arch_rep[i].kernel_shape[0]* model.obj_arch_rep[i].kernel_shape[1] + model.obj_arch_rep[i].bias.shape[0]
        test_code += str(output_num)

    
    
    test_code +="""; i++) {
        floatSdCh valOut = outStream.read();  // Correctly read the output as floatSdCh

        // Convert integer output back to float
        float outputValue = static_cast<float>(valOut.data.to_int());  // Convert to float

        // Save output to file with fixed precision
        output_file << std::fixed << std::setprecision(6) << outputValue << std::endl;

        // Also print to console for reference
        std::cout << outputValue << std::endl;
    }

    // Close the file
    output_file.close();

    return 0;
}
    """
    return test_code



