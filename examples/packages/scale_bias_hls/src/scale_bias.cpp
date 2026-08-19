#include "scale_bias.hpp"

void scale_bias_hls(
    const float* input,
    float* output,
    int count,
    float scale,
    float bias
) {
    for (int index = 0; index < count; ++index) {
#pragma HLS PIPELINE II=1
        output[index] = input[index] * scale + bias;
    }
}

void scale_bias_backward_input_hls(
    const float* grad_output,
    float* grad_input,
    int count,
    float scale,
    float bias
) {
    (void)bias;
    for (int index = 0; index < count; ++index) {
#pragma HLS PIPELINE II=1
        grad_input[index] = grad_output[index] * scale;
    }
}
