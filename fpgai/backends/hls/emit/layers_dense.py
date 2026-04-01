from __future__ import annotations


def emit_dense_h() -> str:
    return r'''#pragma once
#include "fpgai_types.h"
#include <math.h>

namespace fpgai {

template<int N>
void zero_vec(acc_t x[N]) {
#pragma HLS INLINE
    for (int i = 0; i < N; ++i) {
#pragma HLS UNROLL
        x[i] = (acc_t)0;
    }
}

template<int N>
void copy_act_vec(const act_t src[N], act_t dst[N]) {
#pragma HLS INLINE
    for (int i = 0; i < N; ++i) {
#pragma HLS UNROLL
        dst[i] = src[i];
    }
}

template<int IN, int OUT, typename ACT_T = act_t, typename WGT_T = wgt_t, typename BIAS_T = bias_t, typename ACC_T = acc_t>
void dense_out_in_typed(
    const ACT_T x[IN],
    ACT_T y[OUT],
    const WGT_T W[OUT * IN],
    const BIAS_T B[OUT]
) {
#pragma HLS INLINE off
#if FPGAI_DENSE_OUT_UNROLL > 1
#pragma HLS ARRAY_PARTITION variable=y cyclic factor=FPGAI_DENSE_OUT_UNROLL
#pragma HLS ARRAY_PARTITION variable=B cyclic factor=FPGAI_DENSE_OUT_UNROLL
#endif

OUT_TILE:
    for (int o0 = 0; o0 < OUT; o0 += FPGAI_DENSE_OUT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        ACC_T acc[FPGAI_DENSE_OUT_UNROLL];
#pragma HLS ARRAY_PARTITION variable=acc complete

INIT_ACC:
        for (int oo = 0; oo < FPGAI_DENSE_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
            int o = o0 + oo;
            if (o < OUT) acc[oo] = (ACC_T)B[o];
        }

IN_LOOP:
        for (int i = 0; i < IN; ++i) {
#pragma HLS PIPELINE II=1
MAC_OUT:
            for (int oo = 0; oo < FPGAI_DENSE_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
                int o = o0 + oo;
                if (o < OUT) {
                    const int w_idx = o * IN + i;
                    acc[oo] += (ACC_T)x[i] * (ACC_T)W[w_idx];
                }
            }
        }

WRITE_OUT:
        for (int oo = 0; oo < FPGAI_DENSE_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
            int o = o0 + oo;
            if (o < OUT) y[o] = (ACT_T)acc[oo];
        }
    }
}

template<int IN, int OUT>
void dense_out_in(
    const act_t x[IN],
    act_t y[OUT],
    const wgt_t W[OUT * IN],
    const bias_t B[OUT]
) {
#pragma HLS INLINE off
    dense_out_in_typed<IN, OUT, act_t, wgt_t, bias_t, acc_t>(x, y, W, B);
}

template<int IN, int OUT>
void dense_backward_input(
    const acc_t dY[OUT],
    const wgt_t W[OUT * IN],
    acc_t dX[IN]
) {
#pragma HLS INLINE off
#if FPGAI_DENSE_IN_UNROLL > 1
#pragma HLS ARRAY_PARTITION variable=dX cyclic factor=FPGAI_DENSE_IN_UNROLL
#endif
INIT_DX:
    for (int i = 0; i < IN; ++i) {
#pragma HLS PIPELINE II=1
        dX[i] = (acc_t)0;
    }

BWD_O:
    for (int o = 0; o < OUT; ++o) {
#pragma HLS PIPELINE II=1
        acc_t gy = dY[o];
    BWD_I:
        for (int i = 0; i < IN; ++i) {
#pragma HLS UNROLL factor=FPGAI_DENSE_IN_UNROLL
            dX[i] += gy * (acc_t)W[o * IN + i];
        }
    }
}

template<int IN, int OUT>
void dense_weight_grad(
    const act_t x[IN],
    const acc_t dY[OUT],
    acc_t dW[OUT * IN]
) {
#pragma HLS INLINE off
GRAD_O:
    for (int o = 0; o < OUT; ++o) {
#pragma HLS PIPELINE II=1
        acc_t gy = dY[o];
    GRAD_I:
        for (int i = 0; i < IN; ++i) {
#pragma HLS UNROLL factor=FPGAI_DENSE_IN_UNROLL
            dW[o * IN + i] = gy * (acc_t)x[i];
        }
    }
}

template<int OUT>
void dense_bias_grad(
    const acc_t dY[OUT],
    acc_t dB[OUT]
) {
#pragma HLS INLINE off
BIAS_G:
    for (int o = 0; o < OUT; ++o) {
#pragma HLS PIPELINE II=1
        dB[o] = dY[o];
    }
}

template<int N>
void sgd_update_wgt(
    wgt_t W[N],
    const acc_t dW[N],
    const acc_t lr
) {
#pragma HLS INLINE off
UPD_W:
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        acc_t tmp = (acc_t)W[i] - lr * dW[i];
        W[i] = (wgt_t)tmp;
    }
}

template<int N>
void sgd_update_bias(
    bias_t B[N],
    const acc_t dB[N],
    const acc_t lr
) {
#pragma HLS INLINE off
UPD_B:
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        acc_t tmp = (acc_t)B[i] - lr * dB[i];
        B[i] = (bias_t)tmp;
    }
}

template<int N>
void relu_forward(const act_t x[N], act_t y[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        y[i] = (x[i] > (act_t)0) ? x[i] : (act_t)0;
    }
}

template<int N>
void relu_backward_from_output(const act_t y[N], const acc_t dY[N], acc_t dX[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        dX[i] = (y[i] > (act_t)0) ? dY[i] : (acc_t)0;
    }
}

template<int N>
void sigmoid_forward(const act_t x[N], act_t y[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        float xf = (float)x[i];
        float yf = 1.0f / (1.0f + expf(-xf));
        y[i] = (act_t)yf;
    }
}

template<int N>
void sigmoid_backward_from_output(const act_t y[N], const acc_t dY[N], acc_t dX[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        acc_t yy = (acc_t)y[i];
        dX[i] = dY[i] * yy * ((acc_t)1 - yy);
    }
}

template<int N>
void mse_loss_grad(
    const act_t pred[N],
    const act_t target[N],
    acc_t dY[N],
    acc_t &loss_out
) {
#pragma HLS INLINE off
    loss_out = (acc_t)0;
MSE_G:
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        acc_t diff = (acc_t)pred[i] - (acc_t)target[i];
        dY[i] = ((acc_t)2 / (acc_t)N) * diff;
        loss_out += diff * diff;
    }
    loss_out = loss_out / (acc_t)N;
}

} // namespace fpgai
'''


def emit_dense_cpp() -> str:
    return '#include "layers/dense.h"\n'