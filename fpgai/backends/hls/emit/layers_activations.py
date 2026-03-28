from __future__ import annotations


def emit_activations_h() -> str:
    return r'''#pragma once
#include "fpgai_types.h"

namespace fpgai {

static inline act_t fpga_abs(act_t x) {
    return (x < (act_t)0) ? (act_t)(-x) : x;
}

// Cheap sigmoid approximation.
// Piecewise linear:
//   x <= -4 -> 0
//   x >=  4 -> 1
//   else    -> 0.5 + x/8
static inline act_t sigmoid_approx_scalar(act_t x) {
    if (x <= (act_t)-4) return (act_t)0;
    if (x >= (act_t)4)  return (act_t)1;
    return (act_t)0.5 + (x / (act_t)8);
}

// Cheap exp approximation around 0 for negative values.
// Input is assumed <= 0 in softmax after subtracting max.
// Clamp far negatives to near-zero.
// Use: exp(x) ~= 1 + x + x^2/2 for x in [-4, 0]
static inline act_t exp_approx_neg_scalar(act_t x) {
    if (x <= (act_t)-8) return (act_t)0;
    if (x >= (act_t)0)  return (act_t)1;

    acc_t x1 = (acc_t)x;
    acc_t x2 = x1 * x1;
    acc_t y = (acc_t)1 + x1 + (x2 * (acc_t)0.5);

    if (y < (acc_t)0) y = (acc_t)0;
    return (act_t)y;
}

template<int N>
void relu(const act_t* x, act_t* y) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        y[i] = (x[i] > (act_t)0) ? x[i] : (act_t)0;
    }
}

template<int N>
void relu_inplace(act_t* x) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        x[i] = (x[i] > (act_t)0) ? x[i] : (act_t)0;
    }
}

template<int N>
void leaky_relu(const act_t* x, act_t* y, act_t alpha = (act_t)0.1) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        y[i] = (x[i] > (act_t)0) ? x[i] : (act_t)(alpha * x[i]);
    }
}

template<int N>
void leaky_relu_inplace(act_t* x, act_t alpha = (act_t)0.1) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        x[i] = (x[i] > (act_t)0) ? x[i] : (act_t)(alpha * x[i]);
    }
}

template<int N>
void sigmoid(const act_t* x, act_t* y) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        y[i] = sigmoid_approx_scalar(x[i]);
    }
}

template<int N>
void sigmoid_inplace(act_t* x) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        x[i] = sigmoid_approx_scalar(x[i]);
    }
}

template<int N>
void softmax(const act_t* x, act_t* y) {
#pragma HLS INLINE off

    act_t maxv = x[0];
    for (int i = 1; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        if (x[i] > maxv) maxv = x[i];
    }

    act_t tmp[N];
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=FPGAI_PARTITION_FACTOR

    acc_t sum = 0;
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        act_t shifted = x[i] - maxv;   // <= 0
        act_t e = exp_approx_neg_scalar(shifted);
        tmp[i] = e;
        sum += (acc_t)e;
    }

    if (sum <= (acc_t)0) sum = (acc_t)1;

    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        y[i] = (act_t)((acc_t)tmp[i] / sum);
    }
}

template<int N>
void softmax_inplace(act_t* x) {
#pragma HLS INLINE off

    act_t maxv = x[0];
    for (int i = 1; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        if (x[i] > maxv) maxv = x[i];
    }

    act_t tmp[N];
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=FPGAI_PARTITION_FACTOR

    acc_t sum = 0;
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        act_t shifted = x[i] - maxv;
        act_t e = exp_approx_neg_scalar(shifted);
        tmp[i] = e;
        sum += (acc_t)e;
    }

    if (sum <= (acc_t)0) sum = (acc_t)1;

    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        x[i] = (act_t)((acc_t)tmp[i] / sum);
    }
}

template<int N>
void reshape_copy(const act_t* x, act_t* y) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        y[i] = x[i];
    }
}

} // namespace fpgai
'''


def emit_activations_cpp() -> str:
    return '#include "layers/activations.h"\n'