#include "deeplearn.h"  
float layer_0_input[8];
float target_output[4];
float error_buffer[4];
float layer_0_delta[8];
float layer_0_output[8];
float layer_0_weights[64] = {-0.15182585 ,-0.03462022 ,0.09832354 ,-0.29703742 ,-0.12048688 ,-0.022387309 ,-0.20842573 ,-0.26812863 ,0.25798157 ,-0.3389822 ,0.2626855 ,-0.19214621 ,-0.32115448 ,0.19602874 ,-0.30015382 ,0.27372253 ,0.19669963 ,0.21543449 ,-0.21563584 ,-0.21899663 ,0.06774162 ,0.22668636 ,0.08529713 ,0.18634242 ,-0.09950984 ,-0.059502456 ,0.26011854 ,0.027506888 ,-0.04416581 ,-0.22155805 ,-0.25671798 ,-0.0017957086 ,0.13930376 ,-0.2710417 ,-0.043686092 ,0.19736336 ,0.23245487 ,-0.23468755 ,-0.30476907 ,-0.2680945 ,-0.14524828 ,-0.0550606 ,0.29959282 ,0.019742131 ,0.3277419 ,0.15922043 ,0.045848437 ,0.2640662 ,-0.26034212 ,-0.29558873 ,0.110009804 ,-0.045941453 ,0.046171322 ,0.13240866 ,0.26131976 ,-0.31351134 ,-0.1939091 ,-0.2552551 ,0.18289119 ,-0.19952103 ,0.040513698 ,0.35053068 ,-0.16458432 ,-0.30321044};
float layer_0_bias[8] = {0.22061093 ,0.0043839887 ,-0.014109753 ,-0.3107805 ,0.25174916 ,-0.23282623 ,0.080935225 ,-0.035229284};
float activate_layer_0( float x) { return x; } 
float dactivate_layer_0( float x) { return 1; }
float layer_1_delta[4];
float layer_1_output[4];
float layer_1_weights[32] = {-0.14573553 ,0.26840276 ,-0.24434555 ,0.023005392 ,0.2683581 ,0.35191232 ,0.32535833 ,0.0835219 ,-0.061053965 ,-0.26182514 ,-0.0056677395 ,-0.09830036 ,0.26998013 ,0.19467166 ,0.2043398 ,-0.31316498 ,-0.02325507 ,-0.15796272 ,0.17621554 ,-0.1979068 ,-0.24195655 ,0.163575 ,0.2563326 ,0.1720112 ,-0.16953316 ,-0.2715046 ,-0.020208402 ,0.008109264 ,-0.021195313 ,0.12566942 ,-0.13855709 ,-0.06585436};
float layer_1_bias[4] = {0.10091447 ,-0.3128493 ,-0.32337645 ,0.07267959};
float activate_layer_1( float x) { return x; } 
float dactivate_layer_1( float x) { return 1; }
void dense_forward_layer_0( float * layer_input, float* layer_bias, float * layer_weights , float * layer_output ,int number_of_input_nodes, int number_of_output_nodes){

    for (int i = 0; i < number_of_output_nodes; i++) {
        float activation = layer_bias[i];  // Apply bias to the activation
        for (int j = 0; j < number_of_input_nodes; j++) {
            activation += layer_weights[number_of_input_nodes * i + j] * layer_input[j];
        }
      *(layer_output + i)= activate_layer_0(activation);

   }

}
        void dense_delta_layer_0( float * next_delta, float* next_weights, float* layer_output, float * layer_delta ,int input_size, int output_size){
    for (int i = 0; i < input_size; i++) {
        layer_delta[i] = 0;
        for (int j = 0; j < output_size; j++) {
            layer_delta[i] += next_delta[j] * next_weights[j * input_size + i];
        }
        layer_delta[i] *= dactivate_layer_0(layer_output[i]);
    }
}
        void dense_update_weights_layer_0( float * layer_output, float* layer_weights, float * layer_bias ,float * layer_delta,  int input_size, int output_size, float learning_rate){
                for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            layer_weights[i * input_size + j] += learning_rate * layer_delta[i] * layer_output[j];
        }
        layer_bias[i] += learning_rate * layer_delta[i];
    }
}
        void dense_forward_layer_1( float * layer_input, float* layer_bias, float * layer_weights , float * layer_output ,int number_of_input_nodes, int number_of_output_nodes){

    for (int i = 0; i < number_of_output_nodes; i++) {
        float activation = layer_bias[i];  // Apply bias to the activation
        for (int j = 0; j < number_of_input_nodes; j++) {
            activation += layer_weights[number_of_input_nodes * i + j] * layer_input[j];
        }
      *(layer_output + i)= activate_layer_1(activation);

   }

}
        void dense_delta_layer_1( float * error_buffer, float* layer_output, float * layer_delta , int output_size){
            for (int i = 0; i < output_size; i++) {
                layer_delta[i] = -2 * error_buffer[i] * dactivate_layer_1(layer_output[i]);
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

void dense_update_weights_layer_1( float * layer_output, float* layer_weights, float * layer_bias ,float * layer_delta,  int input_size, int output_size, float learning_rate){
                for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            layer_weights[i * input_size + j] += learning_rate * layer_delta[i] * layer_output[j];
        }
        layer_bias[i] += learning_rate * layer_delta[i];
    }
}
        void deeplearn( hls::stream<floatSdCh> &inStream,  int mode_var, hls::stream<floatSdCh> &outStream){

#pragma HLS INTERFACE axis port=outStream 

#pragma HLS INTERFACE axis port=inStream  

#pragma HLS INTERFACE ap_ctrl_none port=return 
for (int idx = 0; idx < 8; idx++){
          #pragma HLS PIPELINE
          floatSdCh input_data = inStream.read();
layer_0_input[idx] = static_cast<float>(input_data.data.to_int())/10000000;
}
for (int idx = 0; idx < 4; idx++){
          #pragma HLS PIPELINE
          floatSdCh input_data = inStream.read();
target_output[idx] = static_cast<float>(input_data.data.to_int())/10000000;
}
floatSdCh input_data = inStream.read();
int export_weihts = input_data.data.to_int();
dense_forward_layer_0(layer_0_input,layer_0_bias,layer_0_weights,layer_0_output,8,8);
dense_forward_layer_1(layer_0_output,layer_1_bias,layer_1_weights,layer_1_output,8,4);
calculate_loss(layer_1_output, target_output, error_buffer,4);
dense_delta_layer_1(error_buffer,layer_1_output,layer_1_delta,4);
dense_delta_layer_0(layer_1_delta,layer_1_weights,layer_0_output,layer_0_delta,8, 4);
dense_update_weights_layer_1(layer_0_output,layer_1_weights,layer_1_bias,layer_1_delta,8, 4, learning_rate);
dense_update_weights_layer_0(layer_0_input,layer_0_weights,layer_0_bias,layer_0_delta,8, 8, learning_rate);

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
		

			for (int idx = 0; idx < 64; idx++) {
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


			for (int idx = 0; idx < 8; idx++) {
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

			

			for (int idx = 0; idx < 32; idx++) {
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


			for (int idx = 0; idx < 4; idx++) {
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
			