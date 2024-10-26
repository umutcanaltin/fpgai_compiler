#include "deeplearn.h"  
float layer_0_input[8];
float layer_0_output[8];
float layer_0_weights[64] = {-0.15182585 ,-0.03462022 ,0.09832354 ,-0.29703742 ,-0.12048688 ,-0.022387309 ,-0.20842573 ,-0.26812863 ,0.25798157 ,-0.3389822 ,0.2626855 ,-0.19214621 ,-0.32115448 ,0.19602874 ,-0.30015382 ,0.27372253 ,0.19669963 ,0.21543449 ,-0.21563584 ,-0.21899663 ,0.06774162 ,0.22668636 ,0.08529713 ,0.18634242 ,-0.09950984 ,-0.059502456 ,0.26011854 ,0.027506888 ,-0.04416581 ,-0.22155805 ,-0.25671798 ,-0.0017957086 ,0.13930376 ,-0.2710417 ,-0.043686092 ,0.19736336 ,0.23245487 ,-0.23468755 ,-0.30476907 ,-0.2680945 ,-0.14524828 ,-0.0550606 ,0.29959282 ,0.019742131 ,0.3277419 ,0.15922043 ,0.045848437 ,0.2640662 ,-0.26034212 ,-0.29558873 ,0.110009804 ,-0.045941453 ,0.046171322 ,0.13240866 ,0.26131976 ,-0.31351134 ,-0.1939091 ,-0.2552551 ,0.18289119 ,-0.19952103 ,0.040513698 ,0.35053068 ,-0.16458432 ,-0.30321044};
float layer_0_bias[8] = {0.22061093 ,0.0043839887 ,-0.014109753 ,-0.3107805 ,0.25174916 ,-0.23282623 ,0.080935225 ,-0.035229284};

float layer_1_output[4];
float layer_1_weights[32] = {-0.14573553 ,0.26840276 ,-0.24434555 ,0.023005392 ,0.2683581 ,0.35191232 ,0.32535833 ,0.0835219 ,-0.061053965 ,-0.26182514 ,-0.0056677395 ,-0.09830036 ,0.26998013 ,0.19467166 ,0.2043398 ,-0.31316498 ,-0.02325507 ,-0.15796272 ,0.17621554 ,-0.1979068 ,-0.24195655 ,0.163575 ,0.2563326 ,0.1720112 ,-0.16953316 ,-0.2715046 ,-0.020208402 ,0.008109264 ,-0.021195313 ,0.12566942 ,-0.13855709 ,-0.06585436};
float layer_1_bias[4] = {0.10091447 ,-0.3128493 ,-0.32337645 ,0.07267959};

float target_output[4];  // Target output for training

float learning_rate = 0.01;  // Learning rate for weight updates

// Activation function (ReLU example)
float activate_layer_0(float x) { return x; }
float activate_layer_1(float x) { return x; }

// Derivative of ReLU for backpropagation
float activate_layer_0_derivative(float x) { return (x > 0) ? 1 : 0; }
float activate_layer_1_derivative(float x) { return (x > 0) ? 1 : 0; }

// Forward pass for Layer 0
void dense_forward_layer_0(float *layer_input, float *layer_bias, float *layer_weights, float *layer_output, int number_of_input_nodes, int number_of_output_nodes) {
    for (int i = 0; i < number_of_output_nodes; i++) {
        float activation = layer_bias[i];
        for (int j = 0; j < number_of_input_nodes; j++) {
            activation += layer_weights[number_of_input_nodes * i + j] * layer_input[j];
        }
        layer_output[i] = activate_layer_0(activation);
    }
}

// Forward pass for Layer 1
void dense_forward_layer_1(float *layer_input, float *layer_bias, float *layer_weights, float *layer_output, int number_of_input_nodes, int number_of_output_nodes) {
    for (int i = 0; i < number_of_output_nodes; i++) {
        float activation = layer_bias[i];
        for (int j = 0; j < number_of_input_nodes; j++) {
            activation += layer_weights[number_of_input_nodes * i + j] * layer_input[j];
        }
        layer_output[i] = activate_layer_1(activation);
    }
}

// Backpropagation: Layer 1 delta calculation
void delta_layer_1(float *layer_1_output, float *target_output, float *delta_1, int output_size) {
    for (int i = 0; i < output_size; i++) {
        float error = target_output[i] - layer_1_output[i];
        delta_1[i] = error * activate_layer_1_derivative(layer_1_output[i]);
    }
}

// Backpropagation: Layer 1 weight update
void update_weights_layer_1(float *layer_0_output, float *layer_1_weights, float *layer_1_bias, float *delta_1, int input_size, int output_size, float learning_rate) {
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            layer_1_weights[i * input_size + j] += learning_rate * delta_1[i] * layer_0_output[j];
        }
        layer_1_bias[i] += learning_rate * delta_1[i];
    }
}

// Backpropagation: Layer 0 delta calculation
void delta_layer_0(float *delta_1, float *layer_1_weights, float *layer_0_output, float *delta_0, int input_size, int output_size) {
    for (int i = 0; i < input_size; i++) {
        delta_0[i] = 0;
        for (int j = 0; j < output_size; j++) {
            delta_0[i] += delta_1[j] * layer_1_weights[j * input_size + i];
        }
        delta_0[i] *= activate_layer_0_derivative(layer_0_output[i]);
    }
}

// Backpropagation: Layer 0 weight update
void update_weights_layer_0(float *input, float *layer_0_weights, float *layer_0_bias, float *delta_0, int input_size, int output_size, float learning_rate) {
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            layer_0_weights[i * input_size + j] += learning_rate * delta_0[i] * input[j];
        }
        layer_0_bias[i] += learning_rate * delta_0[i];
    }
}

// Main function for HLS
void deeplearn(hls::stream<floatSdCh> &inStream, hls::stream<floatSdCh> &outStream) {

#pragma HLS INTERFACE axis port=outStream
#pragma HLS INTERFACE axis port=inStream
#pragma HLS INTERFACE ap_ctrl_none port=return

    // Input reading
    for (int idx = 0; idx < 8; idx++) {
        #pragma HLS PIPELINE
        floatSdCh input_data = inStream.read();
        layer_0_input[idx] = static_cast<float>(input_data.data.to_int()) / 10000000;
    }

    // Forward pass
    dense_forward_layer_0(layer_0_input, layer_0_bias, layer_0_weights, layer_0_output, 8, 8);
    dense_forward_layer_1(layer_0_output, layer_1_bias, layer_1_weights, layer_1_output, 8, 4);

    // Target output (For simplicity, this could be read from input as well)
    float target_output[4] = {0.5, 0.1, 0.8, 0.6};  // Example targets

    // Backpropagation
    float delta_1[4];  // Delta for layer 1
    float delta_0[8];  // Delta for layer 0

    // Delta and weight update for Layer 1
    delta_layer_1(layer_1_output, target_output, delta_1, 4);
    update_weights_layer_1(layer_1_input, layer_1_weights, layer_1_bias, delta_1, 8, 4, learning_rate);

    // Delta and weight update for Layer 0
    delta_layer_0(delta_1, layer_1_weights, layer_0_output, delta_0, 8, 4);
    update_weights_layer_0(layer_0_input, layer_0_weights, layer_0_bias, delta_0, 8, 8);

    // Output results (for testing or further use)
    for (int i = 0; i < 4; i++) {
        #pragma HLS PIPELINE
        floatSdCh output_data;
        output_data.data = layer_1_output[i] * 10000000;  // Scale back output if needed
        outStream.write(output_data);
    }
}
