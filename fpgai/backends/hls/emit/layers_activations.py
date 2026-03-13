def emit_activations_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"
#include <hls_math.h>

namespace fpgai {

// -----------------------------------------------------------------------------
// New typed kernels
// -----------------------------------------------------------------------------
template<typename ACT_T, int N>
void relu_inplace_typed(ACT_T x[N]) {
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        if (x[i] < (ACT_T)0) {
            x[i] = (ACT_T)0;
        }
    }
}

template<typename ACT_T, int N>
void leaky_relu_inplace_typed(ACT_T x[N], ACT_T alpha) {
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        if (x[i] <= (ACT_T)0) {
            x[i] = (ACT_T)(x[i] * alpha);
        }
    }
}

template<typename ACT_T, int N>
void sigmoid_inplace_typed(ACT_T x[N]) {
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        float xf = (float)x[i];
        float yf = 1.0f / (1.0f + hls::expf(-xf));
        x[i] = (ACT_T)yf;
    }
}

template<typename ACT_T, int N>
void softmax_inplace_typed(ACT_T x[N]) {
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
        x[i] = (ACT_T)(tmp[i] * inv_sum);
    }
}

// -----------------------------------------------------------------------------
// Backward-compatible wrappers for old top emitter
// -----------------------------------------------------------------------------
template<int N>
void relu_inplace(act_t x[N]) {
    relu_inplace_typed<act_t, N>(x);
}

template<int N>
void leaky_relu_inplace(act_t x[N], act_t alpha) {
    leaky_relu_inplace_typed<act_t, N>(x, alpha);
}

template<int N>
void sigmoid_inplace(act_t x[N]) {
    sigmoid_inplace_typed<act_t, N>(x);
}

template<int N>
void softmax_inplace(act_t x[N]) {
    softmax_inplace_typed<act_t, N>(x);
}

} // namespace fpgai
'''


def emit_activations_cpp() -> str:
    return r'''
#include "layers/activations.h"
'''