#include "split_scale_tensor_ports.hpp"

void split_scale_tensor_ports_hls(
    const float* input,
    float* identity,
    float* scaled,
    int count,
    float scale
) {
    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        const float value = input[i];
        identity[i] = value;
        scaled[i] = value * scale;
    }
}
