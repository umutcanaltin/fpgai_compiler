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
