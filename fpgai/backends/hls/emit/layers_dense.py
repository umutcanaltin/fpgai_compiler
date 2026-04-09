from __future__ import annotations


def emit_dense_h() -> str:
    return r'''#pragma once
#include "fpgai_types.h"

#ifndef FPGAI_PIPELINE_II
#define FPGAI_PIPELINE_II 1
#endif

#ifndef FPGAI_DENSE_IN_UNROLL
#define FPGAI_DENSE_IN_UNROLL 1
#endif

#ifndef FPGAI_DENSE_OUT_UNROLL
#define FPGAI_DENSE_OUT_UNROLL 1
#endif

namespace fpgai {

template<int IN_F, int OUT_F>
void dense_out_in(
    const act_t x[IN_F],
    act_t y[OUT_F],
    const wgt_t W[OUT_F * IN_F],
    const bias_t B[OUT_F]
) {
#pragma HLS INLINE off
    for (int o0 = 0; o0 < OUT_F; o0 += FPGAI_DENSE_OUT_UNROLL) {
        acc_t acc[FPGAI_DENSE_OUT_UNROLL];
#pragma HLS ARRAY_PARTITION variable=acc complete

        for (int ou = 0; ou < FPGAI_DENSE_OUT_UNROLL; ++ou) {
#pragma HLS UNROLL
            int o = o0 + ou;
            if (o < OUT_F) acc[ou] = (acc_t)B[o];
        }

        for (int i0 = 0; i0 < IN_F; i0 += FPGAI_DENSE_IN_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
            for (int iu = 0; iu < FPGAI_DENSE_IN_UNROLL; ++iu) {
#pragma HLS UNROLL
                int i = i0 + iu;
                if (i < IN_F) {
                    act_t xv = x[i];
                    for (int ou = 0; ou < FPGAI_DENSE_OUT_UNROLL; ++ou) {
#pragma HLS UNROLL
                        int o = o0 + ou;
                        if (o < OUT_F) {
                            acc[ou] += (acc_t)xv * (acc_t)W[o * IN_F + i];
                        }
                    }
                }
            }
        }

        for (int ou = 0; ou < FPGAI_DENSE_OUT_UNROLL; ++ou) {
#pragma HLS UNROLL
            int o = o0 + ou;
            if (o < OUT_F) y[o] = (act_t)acc[ou];
        }
    }
}

template<int IN_F, int OUT_F>
void dense_weight_grad(
    const act_t x[IN_F],
    const grad_act_t dY[OUT_F],
    grad_wgt_t dW[OUT_F * IN_F]
) {
#pragma HLS INLINE off

    // Conservative implementation for compile time:
    // one final store per weight element, no pipelined multi-write conflict.
    for (int o = 0; o < OUT_F; ++o) {
        for (int i = 0; i < IN_F; ++i) {
            dW[o * IN_F + i] = (grad_wgt_t)((acc_t)dY[o] * (acc_t)x[i]);
        }
    }
}

template<int OUT_F>
void dense_bias_grad(
    const grad_act_t dY[OUT_F],
    grad_bias_t dB[OUT_F]
) {
#pragma HLS INLINE off
    for (int o = 0; o < OUT_F; ++o) {
#pragma HLS PIPELINE II=1
        dB[o] = (grad_bias_t)dY[o];
    }
}

template<int IN_F, int OUT_F>
void dense_backward_input(
    const grad_act_t dY[OUT_F],
    const wgt_t W[OUT_F * IN_F],
    grad_act_t dX[IN_F]
) {
#pragma HLS INLINE off
    for (int i = 0; i < IN_F; ++i) {
        acc_t acc = 0;
        for (int o = 0; o < OUT_F; ++o) {
            acc += (acc_t)dY[o] * (acc_t)W[o * IN_F + i];
        }
        dX[i] = (grad_act_t)acc;
    }
}

template<int N>
void sgd_update_wgt(
    wgt_t W[N],
    const grad_wgt_t dW[N],
    const upd_t lr
) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        W[i] = (wgt_t)((acc_t)W[i] - (acc_t)lr * (acc_t)dW[i]);
    }
}

template<int N>
void sgd_update_bias(
    bias_t B[N],
    const grad_bias_t dB[N],
    const upd_t lr
) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        B[i] = (bias_t)((acc_t)B[i] - (acc_t)lr * (acc_t)dB[i]);
    }
}

} // namespace fpgai
'''


def emit_dense_cpp() -> str:
    return '#include "layers/dense.h"\n'