def emit_activations_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"
#include <hls_math.h>

namespace fpgai {

template<int N>
void relu_inplace(act_t x[N]) {
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        if (x[i] < (act_t)0) {
            x[i] = (act_t)0;
        }
    }
}

template<int N>
void leaky_relu_inplace(act_t x[N], act_t alpha) {
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        if (x[i] <= (act_t)0) {
            x[i] = (act_t)(x[i] * alpha);
        }
    }
}

template<int N>
void sigmoid_inplace(act_t x[N]) {
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        float xf = (float)x[i];
        float yf = 1.0f / (1.0f + hls::expf(-xf));
        x[i] = (act_t)yf;
    }
}

template<int N>
void softmax_inplace(act_t x[N]) {
    float tmp[N];
    float max_val = -1e30f;

    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        float f = (float)x[i];
        tmp[i] = f;
        if (f > max_val) {
            max_val = f;
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        tmp[i] = hls::expf(tmp[i] - max_val);
        sum += tmp[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        x[i] = (act_t)(tmp[i] * inv_sum);
    }
}

} // namespace fpgai
'''


def emit_activations_cpp() -> str:
    return r'''
#include "layers/activations.h"
'''