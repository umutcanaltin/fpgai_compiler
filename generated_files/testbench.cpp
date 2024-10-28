
#include "deeplearn.h"  // Make sure this includes the necessary definitions and typedefs
#include <iostream>
#include <iomanip>  // For std::fixed and std::setprecision
#include <fstream>  // For file operations

int main() {
    // Declare streams
    hls::stream<floatSdCh> inStream;
    hls::stream<floatSdCh> outStream;

    // Initialize an array of input data as integers 
int input_data[number_of_inputs+number_of_outputs+1] = {1000000,2000000,3000000,4000000,5000000,6000000,7000000,8000000,9000000,10000000,11000000,12000000,13000000,14000000,15000000,16000000,17000000,18000000,19000000,20000000,20999999,22000000,22999999,23999998,25000000,1000000,2000000,3000000,4000000,5000000,6000000,7000000,8000000,9000000,10000000,0};

  // Write the integer input data to the stream
    for (int i = 0; i < number_of_inputs+number_of_outputs+1; i++) {
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
        
        std::ofstream output_file("/home/umutcanaltin/Desktop/github_projects/fpgai_compiler/generated_files/hls_output.txt");


    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open the output file!" << std::endl;
        return 1;
    }

    // Retrieve the output and convert to float
    for (int i = 0; i <10; i++) {
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
    