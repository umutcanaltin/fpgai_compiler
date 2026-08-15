#include "add_tensor_ports.hpp"
void add_tensor_ports_hls(const float* lhs, const float* rhs, float* output, int count) {
    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = lhs[i] + rhs[i];
    }
}
