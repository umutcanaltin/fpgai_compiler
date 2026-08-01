#include "external_wrapper.h"

void scale_bias_hls(const float* input, float* output, int count, float scale, float bias);

extern "C" void deeplearn(
    const float input[4],
    float output[4]
) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    scale_bias_hls(input, output, 4, 2.0f, 1.0f);
}
