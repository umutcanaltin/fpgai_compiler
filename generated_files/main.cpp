#include "deeplearn.h"  
float layer_0_input[25];
float target_output[9];
float error_buffer[9];
float layer_0_delta[16];
int layer_0_filter_num =1;
float layer_0_output[16];
float layer_0_weights[4] = {0.35346144 ,-0.3802634 ,-0.30074626 ,-0.057160378};
float layer_0_bias[1] = {-0.19155061};
float activate_layer_0( float x) { return x; } 
float dactivate_layer_0( float x) { return 1; }
float layer_1_delta[9];
int layer_1_filter_num =1;
float layer_1_output[9];
float layer_1_weights[4] = {-0.15677583 ,-0.2294904 ,-0.1686756 ,0.10529846};
float layer_1_bias[1] = {0.23754978};
float activate_layer_1( float x) { return x; } 
float dactivate_layer_1( float x) { return 1; }
void convolution_forward_layer_0( float * input, float  * filters,float * layer_bias, int filter_number, int filter_size_1, int filter_size_2,int input_size_1, int input_size_2, float * output){
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
                *(output + output_idx) = activate_layer_0(sum);
            }
        }
    }
}
        void convolution_delta_layer_0( float * next_delta, float* next_weights, float* layer_output, float * layer_delta ,int input_height,int input_width, int output_height, int output_width,int filter_height, int filter_width){
     
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
            layer_delta[oh * output_width + ow] *= dactivate_layer_0(layer_output[oh * output_width + ow]);
        }
    }
}
    
        void convolution_update_weights_layer_0( float * layer_output, float* layer_weights, float * layer_bias ,float * layer_delta,  int filter_height, int filter_width, int input_height, int input_width, float learning_rate){
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

        void convolution_forward_layer_1( float * input, float  * filters,float * layer_bias, int filter_number, int filter_size_1, int filter_size_2,int input_size_1, int input_size_2, float * output){
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
                *(output + output_idx) = activate_layer_1(sum);
            }
        }
    }
}
        void convolution_delta_layer_1( float * error_buffer, float* layer_output, float * layer_delta , int output_size){
            for (int i = 0; i < output_size; i++) {
                layer_delta[i] = -2 * error_buffer[i] *  dactivate_layer_1(*layer_output+i);
            }
        }
        
float calculate_loss(float *layer_output, float *target_output, float *error_buffer, int output_size) {
    float loss = 0.0f;
    for (int i = 0; i < output_size; i++) {
        float error = target_output[i] - layer_output[i];
        error_buffer[i] = error; 
        loss += error * error;
    }
    return loss / output_size; 
}

void convolution_update_weights_layer_1( float * layer_output, float* layer_weights, float * layer_bias ,float * layer_delta,  int filter_height, int filter_width, int input_height, int input_width, float learning_rate){
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

        void deeplearn( hls::stream<floatSdCh> &inStream,  int mode_var, hls::stream<floatSdCh> &outStream){

#pragma HLS INTERFACE axis port=outStream 

#pragma HLS INTERFACE axis port=inStream  

#pragma HLS INTERFACE ap_ctrl_none port=return 
for (int idx = 0; idx < 25; idx++){
          #pragma HLS PIPELINE
          floatSdCh input_data = inStream.read();
layer_0_input[idx] = static_cast<float>(input_data.data.to_int())/10000000;
}
for (int idx = 0; idx < 9; idx++){
          #pragma HLS PIPELINE
          floatSdCh input_data = inStream.read();
target_output[idx] = static_cast<float>(input_data.data.to_int())/10000000;
}
floatSdCh input_data = inStream.read();
int export_weihts = input_data.data.to_int();
convolution_forward_layer_0(layer_0_input,layer_0_weights,layer_0_bias,layer_0_filter_num,2,2,5,5,layer_0_output );
convolution_forward_layer_1(layer_0_output,layer_1_weights,layer_1_bias,layer_1_filter_num,2,2,4,4,layer_1_output );
calculate_loss(layer_1_output, target_output, error_buffer,9);
convolution_delta_layer_1(error_buffer,layer_1_output,layer_1_delta,9);
convolution_delta_layer_0(layer_1_delta,layer_1_weights,layer_0_output,layer_0_delta,5, 5, 4, 4, 2, 2);
convolution_update_weights_layer_1(layer_0_output,layer_1_weights,layer_1_bias,layer_1_delta,2,2,4,4, learning_rate);
convolution_update_weights_layer_0(layer_0_input,layer_0_weights,layer_0_bias,layer_0_delta,2,2,5,5, learning_rate);

		if(export_weihts == 0){
		for (int idx = 0; idx < number_of_outputs; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(layer_1_output[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = (idx == number_of_outputs - 1) ? 1 : 0;
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			} } 
			else{
		

			for (int idx = 0; idx < 4; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(layer_0_weights[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = 0;
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			}


			for (int idx = 0; idx < 1; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(layer_0_bias[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = 0;
				
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			}

			

			for (int idx = 0; idx < 4; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(layer_1_weights[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = 0;
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			}


			for (int idx = 0; idx < 1; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(layer_1_bias[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = (idx == number_of_outputs - 1) ? 1 : 0;
				
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			}

				} }
			#include "deeplearn.h"  
float layer_0_input[25];
float target_output[9];
float error_buffer[9];
float layer_0_delta[16];
int layer_0_filter_num =1;
float layer_0_output[16];
float layer_0_weights[4] = {0.35346144 ,-0.3802634 ,-0.30074626 ,-0.057160378};
float layer_0_bias[1] = {-0.19155061};
float activate_layer_0( float x) { return x; } 
float dactivate_layer_0( float x) { return 1; }
float layer_1_delta[9];
int layer_1_filter_num =1;
float layer_1_output[9];
float layer_1_weights[4] = {-0.15677583 ,-0.2294904 ,-0.1686756 ,0.10529846};
float layer_1_bias[1] = {0.23754978};
float activate_layer_1( float x) { return x; } 
float dactivate_layer_1( float x) { return 1; }
void convolution_forward_layer_0( float * input, float  * filters,float * layer_bias, int filter_number, int filter_size_1, int filter_size_2,int input_size_1, int input_size_2, float * output){
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
                *(output + output_idx) = activate_layer_0(sum);
            }
        }
    }
}
        void convolution_delta_layer_0( float * next_delta, float* next_weights, float* layer_output, float * layer_delta ,int input_height,int input_width, int output_height, int output_width,int filter_height, int filter_width){
     
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
            layer_delta[oh * output_width + ow] *= dactivate_layer_0(layer_output[oh * output_width + ow]);
        }
    }
}
    
        void convolution_update_weights_layer_0( float * layer_output, float* layer_weights, float * layer_bias ,float * layer_delta,  int filter_height, int filter_width, int input_height, int input_width, float learning_rate){
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

        void convolution_forward_layer_1( float * input, float  * filters,float * layer_bias, int filter_number, int filter_size_1, int filter_size_2,int input_size_1, int input_size_2, float * output){
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
                *(output + output_idx) = activate_layer_1(sum);
            }
        }
    }
}
        void convolution_delta_layer_1( float * error_buffer, float* layer_output, float * layer_delta , int output_size){
            for (int i = 0; i < output_size; i++) {
                layer_delta[i] = -2 * error_buffer[i] *  dactivate_layer_1(*layer_output+i);
            }
        }
        
float calculate_loss(float *layer_output, float *target_output, float *error_buffer, int output_size) {
    float loss = 0.0f;
    for (int i = 0; i < output_size; i++) {
        float error = target_output[i] - layer_output[i];
        error_buffer[i] = error; 
        loss += error * error;
    }
    return loss / output_size; 
}

void convolution_update_weights_layer_1( float * layer_output, float* layer_weights, float * layer_bias ,float * layer_delta,  int filter_height, int filter_width, int input_height, int input_width, float learning_rate){
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

        void deeplearn( hls::stream<floatSdCh> &inStream,  int mode_var, hls::stream<floatSdCh> &outStream){

#pragma HLS INTERFACE axis port=outStream 

#pragma HLS INTERFACE axis port=inStream  

#pragma HLS INTERFACE ap_ctrl_none port=return 
for (int idx = 0; idx < 25; idx++){
          #pragma HLS PIPELINE
          floatSdCh input_data = inStream.read();
layer_0_input[idx] = static_cast<float>(input_data.data.to_int())/10000000;
}
for (int idx = 0; idx < 9; idx++){
          #pragma HLS PIPELINE
          floatSdCh input_data = inStream.read();
target_output[idx] = static_cast<float>(input_data.data.to_int())/10000000;
}
floatSdCh input_data = inStream.read();
int export_weihts = input_data.data.to_int();
convolution_forward_layer_0(layer_0_input,layer_0_weights,layer_0_bias,layer_0_filter_num,2,2,5,5,layer_0_output );
convolution_forward_layer_1(layer_0_output,layer_1_weights,layer_1_bias,layer_1_filter_num,2,2,4,4,layer_1_output );
calculate_loss(layer_1_output, target_output, error_buffer,9);
convolution_delta_layer_1(error_buffer,layer_1_output,layer_1_delta,9);
convolution_delta_layer_0(layer_1_delta,layer_1_weights,layer_0_output,layer_0_delta,5, 5, 4, 4, 2, 2);
convolution_update_weights_layer_1(layer_0_output,layer_1_weights,layer_1_bias,layer_1_delta,2,2,4,4, learning_rate);
convolution_update_weights_layer_0(layer_0_input,layer_0_weights,layer_0_bias,layer_0_delta,2,2,5,5, learning_rate);

		if(export_weihts == 0){
		for (int idx = 0; idx < number_of_outputs; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(layer_1_output[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = (idx == number_of_outputs - 1) ? 1 : 0;
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			} } 
			else{
		

			for (int idx = 0; idx < 4; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(layer_0_weights[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = 0;
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			}


			for (int idx = 0; idx < 1; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(layer_0_bias[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = 0;
				
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			}

			

			for (int idx = 0; idx < 4; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(layer_1_weights[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = 0;
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			}


			for (int idx = 0; idx < 1; idx++) {
				#pragma HLS PIPELINE
				floatSdCh output_data;
				output_data.data = static_cast<ap_int<32>>(layer_1_bias[idx]*10000000); // Convert float to integer
				output_data.keep = -1; // Keep all bytes
				output_data.strb = -1; // All strobes valid
				output_data.user = 0;
				output_data.last = (idx == number_of_outputs - 1) ? 1 : 0;
				
				output_data.id = 0;
				output_data.dest = 0;
				outStream.write(output_data);
			}

				} }
			