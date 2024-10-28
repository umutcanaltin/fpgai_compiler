
    #include "deeplearn.h"  // Make sure this includes the necessary definitions and typedefs
    #include <iostream>
    #include <iomanip>  // For std::fixed and std::setprecision
    #include <fstream>  
int input_data[number_of_inputs] = {1000000,2000000,3000000,4000000,5000000,6000000,7000000,8000000,8000000}

    // Write the integer input data to the stream
    for (int i = 0; i < number_of_inputs; i++) {
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
    std::ofstream output_file("/home/umutcanaltin/Desktop/hls_output.txt");

    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open the output file!" << std::endl;
        return 1;
    }

    // Retrieve the output and convert to float
    for (int i = 0; i < number_of_outputs; i++) {
        floatSdCh valOut = outStream.read();  // Correctly read the output as floatSdCh

        // Convert integer output back to float
        int outputValue = valOut.data;  // Convert to float

        // Save output to file with fixed precision
        output_file << outputValue << std::endl;

        // Also print to console for reference
        std::cout << outputValue << std::endl;
    }

    // Close the file
    output_file.close();

    return 0;
}