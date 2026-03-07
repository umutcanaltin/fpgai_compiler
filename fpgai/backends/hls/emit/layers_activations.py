def emit_activations_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"
#include <hls_math.h> // For exp()

namespace fpgai {

template<int N>
void relu_inplace(act_t x[N]) {
    for(int i=0; i<N; i++) {
        #pragma HLS PIPELINE
        if(x[i] < 0) x[i] = 0;
    }
}

template<int N>
void leaky_relu_inplace(act_t x[N], float alpha) {
    act_t a = (act_t)alpha;
    for(int i=0; i<N; i++) {
        #pragma HLS PIPELINE
        if(x[i] < 0) x[i] = x[i] * a;
    }
}

// Softmax (casts to float for accuracy, then back)
template<int N>
void softmax_inplace(act_t x[N]) {
    float tmp[N];
    float max_val = -1e9f;

    // 1. Find Max (for numerical stability)
    for(int i=0; i<N; i++) {
        float f = (float)x[i];
        if(f > max_val) max_val = f;
        tmp[i] = f;
    }

    // 2. Exp and Sum
    float sum = 0.0f;
    for(int i=0; i<N; i++) {
        #pragma HLS PIPELINE
        tmp[i] = hls::exp(tmp[i] - max_val);
        sum += tmp[i];
    }

    // 3. Normalize and Write Back
    for(int i=0; i<N; i++) {
        #pragma HLS PIPELINE
        x[i] = (act_t)(tmp[i] / sum);
    }
}

} // namespace
'''

def emit_activations_cpp() -> str:
    return '#include "layers/activations.h"\n'