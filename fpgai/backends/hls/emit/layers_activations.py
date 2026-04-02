from __future__ import annotations


def emit_activations_h() -> str:
    return r'''#pragma once
#include "fpgai_types.h"

#ifndef FPGAI_PIPELINE_II
#define FPGAI_PIPELINE_II 1
#endif

#ifndef FPGAI_ACT_UNROLL
#define FPGAI_ACT_UNROLL 1
#endif

namespace fpgai {

static inline act_t sigmoid_approx_scalar(act_t x) {
    if (x <= (act_t)-4) return (act_t)0;
    if (x >= (act_t)4)  return (act_t)1;
    return (act_t)0.5 + (x / (act_t)8);
}

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
    for (int i0 = 0; i0 < N; i0 += FPGAI_ACT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        for (int u = 0; u < FPGAI_ACT_UNROLL; ++u) {
#pragma HLS UNROLL
            int i = i0 + u;
            if (i < N) y[i] = (x[i] > (act_t)0) ? x[i] : (act_t)0;
        }
    }
}

template<int N>
void relu_backward_from_output(const act_t* y, const grad_act_t* dY, grad_act_t* dX) {
#pragma HLS INLINE off
    for (int i0 = 0; i0 < N; i0 += FPGAI_ACT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        for (int u = 0; u < FPGAI_ACT_UNROLL; ++u) {
#pragma HLS UNROLL
            int i = i0 + u;
            if (i < N) dX[i] = (y[i] > (act_t)0) ? dY[i] : (grad_act_t)0;
        }
    }
}

template<int N>
void leaky_relu(const act_t* x, act_t* y, act_t alpha = (act_t)0.1) {
#pragma HLS INLINE off
    for (int i0 = 0; i0 < N; i0 += FPGAI_ACT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        for (int u = 0; u < FPGAI_ACT_UNROLL; ++u) {
#pragma HLS UNROLL
            int i = i0 + u;
            if (i < N) y[i] = (x[i] > (act_t)0) ? x[i] : (act_t)(alpha * x[i]);
        }
    }
}

template<int N>
void leaky_relu_backward_from_input(const act_t* x, const grad_act_t* dY, grad_act_t* dX, act_t alpha = (act_t)0.1) {
#pragma HLS INLINE off
    for (int i0 = 0; i0 < N; i0 += FPGAI_ACT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        for (int u = 0; u < FPGAI_ACT_UNROLL; ++u) {
#pragma HLS UNROLL
            int i = i0 + u;
            if (i < N) dX[i] = (x[i] > (act_t)0) ? dY[i] : (grad_act_t)((acc_t)alpha * (acc_t)dY[i]);
        }
    }
}

template<int N>
void sigmoid(const act_t* x, act_t* y) {
#pragma HLS INLINE off
    for (int i0 = 0; i0 < N; i0 += FPGAI_ACT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        for (int u = 0; u < FPGAI_ACT_UNROLL; ++u) {
#pragma HLS UNROLL
            int i = i0 + u;
            if (i < N) y[i] = sigmoid_approx_scalar(x[i]);
        }
    }
}

template<int N>
void sigmoid_backward_from_output(const act_t* y, const grad_act_t* dY, grad_act_t* dX) {
#pragma HLS INLINE off
    for (int i0 = 0; i0 < N; i0 += FPGAI_ACT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        for (int u = 0; u < FPGAI_ACT_UNROLL; ++u) {
#pragma HLS UNROLL
            int i = i0 + u;
            if (i < N) {
                acc_t yy = (acc_t)y[i];
                dX[i] = (grad_act_t)((acc_t)dY[i] * yy * ((acc_t)1 - yy));
            }
        }
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
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=FPGAI_ACT_UNROLL

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
        y[i] = (act_t)((acc_t)tmp[i] / sum);
    }
}

template<int N>
void softmax_backward(const act_t* y, const grad_act_t* dY, grad_act_t* dX) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        acc_t acc = 0;
        for (int j = 0; j < N; ++j) {
            acc_t jac = (i == j)
                ? ((acc_t)y[i] * ((acc_t)1 - (acc_t)y[i]))
                : (-(acc_t)y[i] * (acc_t)y[j]);
            acc += jac * (acc_t)dY[j];
        }
        dX[i] = (grad_act_t)acc;
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

template<int N>
void add_vec(const act_t* a, const act_t* b, act_t* y) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        y[i] = (act_t)((acc_t)a[i] + (acc_t)b[i]);
    }
}

template<int N>
void add_backward(const grad_act_t* dY, grad_act_t* dA, grad_act_t* dB) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        dA[i] += dY[i];
        dB[i] += dY[i];
    }
}

} // namespace fpgai
'''


def emit_activations_cpp() -> str:
    return '#include "layers/activations.h"\n'