from __future__ import annotations


def emit_dense_h() -> str:
    return r'''#pragma once
#include "fpgai_types.h"
#include <math.h>

#ifndef FPGAI_PIPELINE_II
#define FPGAI_PIPELINE_II 1
#endif

#ifndef FPGAI_DENSE_OUT_UNROLL
#define FPGAI_DENSE_OUT_UNROLL 1
#endif

#ifndef FPGAI_DENSE_IN_UNROLL
#define FPGAI_DENSE_IN_UNROLL 1
#endif

#ifndef FPGAI_DENSE_BWD_OUT_UNROLL
#define FPGAI_DENSE_BWD_OUT_UNROLL FPGAI_DENSE_OUT_UNROLL
#endif

#ifndef FPGAI_DENSE_BWD_IN_UNROLL
#define FPGAI_DENSE_BWD_IN_UNROLL FPGAI_DENSE_IN_UNROLL
#endif

#ifndef FPGAI_DENSE_UPD_UNROLL
#define FPGAI_DENSE_UPD_UNROLL 1
#endif

#ifndef FPGAI_DENSE_PARTITION_INPUT
#define FPGAI_DENSE_PARTITION_INPUT 1
#endif

#ifndef FPGAI_DENSE_PARTITION_OUTPUT
#define FPGAI_DENSE_PARTITION_OUTPUT 1
#endif

#ifndef FPGAI_DENSE_PARTITION_WEIGHTS
#define FPGAI_DENSE_PARTITION_WEIGHTS 1
#endif

#ifndef FPGAI_DENSE_PARTITION_GRADS
#define FPGAI_DENSE_PARTITION_GRADS 1
#endif

namespace fpgai {

template<int N>
void zero_vec_acc(acc_t x[N]) {
#pragma HLS INLINE
    for (int i = 0; i < N; ++i) {
#pragma HLS UNROLL
        x[i] = (acc_t)0;
    }
}

template<int N>
void zero_vec_act(act_t x[N]) {
#pragma HLS INLINE
    for (int i = 0; i < N; ++i) {
#pragma HLS UNROLL
        x[i] = (act_t)0;
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
#if FPGAI_DENSE_PARTITION_INPUT > 1
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=FPGAI_DENSE_PARTITION_INPUT
#endif
#if FPGAI_DENSE_PARTITION_OUTPUT > 1
#pragma HLS ARRAY_PARTITION variable=y cyclic factor=FPGAI_DENSE_PARTITION_OUTPUT
#pragma HLS ARRAY_PARTITION variable=B cyclic factor=FPGAI_DENSE_PARTITION_OUTPUT
#endif
#if FPGAI_DENSE_PARTITION_WEIGHTS > 1
#pragma HLS ARRAY_PARTITION variable=W cyclic factor=FPGAI_DENSE_PARTITION_WEIGHTS
#endif

    for (int o0 = 0; o0 < OUT; o0 += FPGAI_DENSE_OUT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        ACC_T acc[FPGAI_DENSE_OUT_UNROLL];
#pragma HLS ARRAY_PARTITION variable=acc complete

        for (int oo = 0; oo < FPGAI_DENSE_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
            const int o = o0 + oo;
            if (o < OUT) acc[oo] = (ACC_T)B[o];
        }

        for (int i0 = 0; i0 < IN; i0 += FPGAI_DENSE_IN_UNROLL) {
            for (int oo = 0; oo < FPGAI_DENSE_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
                const int o = o0 + oo;
                if (o < OUT) {
                    for (int ii = 0; ii < FPGAI_DENSE_IN_UNROLL; ++ii) {
#pragma HLS UNROLL
                        const int i = i0 + ii;
                        if (i < IN) {
                            const int w_idx = o * IN + i;
                            acc[oo] += (ACC_T)x[i] * (ACC_T)W[w_idx];
                        }
                    }
                }
            }
        }

        for (int oo = 0; oo < FPGAI_DENSE_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
            const int o = o0 + oo;
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

template<int IN, int OUT, typename GRAD_T = grad_act_t, typename WGT_T = wgt_t, typename ACC_T = acc_t>
void dense_backward_input_typed(
    const GRAD_T dY[OUT],
    const WGT_T W[OUT * IN],
    GRAD_T dX[IN]
) {
#pragma HLS INLINE off
#if FPGAI_DENSE_PARTITION_GRADS > 1
#pragma HLS ARRAY_PARTITION variable=dY cyclic factor=FPGAI_DENSE_PARTITION_GRADS
#pragma HLS ARRAY_PARTITION variable=dX cyclic factor=FPGAI_DENSE_PARTITION_GRADS
#endif
#if FPGAI_DENSE_PARTITION_WEIGHTS > 1
#pragma HLS ARRAY_PARTITION variable=W cyclic factor=FPGAI_DENSE_PARTITION_WEIGHTS
#endif

    for (int i = 0; i < IN; ++i) {
#pragma HLS PIPELINE II=1
        dX[i] = (GRAD_T)0;
    }

    for (int o0 = 0; o0 < OUT; o0 += FPGAI_DENSE_BWD_OUT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        for (int oo = 0; oo < FPGAI_DENSE_BWD_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
            const int o = o0 + oo;
            if (o < OUT) {
                const ACC_T gy = (ACC_T)dY[o];
                for (int i0 = 0; i0 < IN; i0 += FPGAI_DENSE_BWD_IN_UNROLL) {
                    for (int ii = 0; ii < FPGAI_DENSE_BWD_IN_UNROLL; ++ii) {
#pragma HLS UNROLL
                        const int i = i0 + ii;
                        if (i < IN) dX[i] += (GRAD_T)(gy * (ACC_T)W[o * IN + i]);
                    }
                }
            }
        }
    }
}

template<int IN, int OUT>
void dense_backward_input(
    const grad_act_t dY[OUT],
    const wgt_t W[OUT * IN],
    grad_act_t dX[IN]
) {
#pragma HLS INLINE off
    dense_backward_input_typed<IN, OUT, grad_act_t, wgt_t, acc_t>(dY, W, dX);
}

template<int IN, int OUT, typename ACT_T = act_t, typename GRAD_T = grad_act_t, typename G_T = grad_wgt_t>
void dense_weight_grad_typed(
    const ACT_T x[IN],
    const GRAD_T dY[OUT],
    G_T dW[OUT * IN]
) {
#pragma HLS INLINE off
#if FPGAI_DENSE_PARTITION_INPUT > 1
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=FPGAI_DENSE_PARTITION_INPUT
#endif
#if FPGAI_DENSE_PARTITION_GRADS > 1
#pragma HLS ARRAY_PARTITION variable=dY cyclic factor=FPGAI_DENSE_PARTITION_GRADS
#pragma HLS ARRAY_PARTITION variable=dW cyclic factor=FPGAI_DENSE_PARTITION_GRADS
#endif

    for (int o0 = 0; o0 < OUT; o0 += FPGAI_DENSE_BWD_OUT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        for (int oo = 0; oo < FPGAI_DENSE_BWD_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
            const int o = o0 + oo;
            if (o < OUT) {
                const acc_t gy = (acc_t)dY[o];
                for (int i0 = 0; i0 < IN; i0 += FPGAI_DENSE_BWD_IN_UNROLL) {
                    for (int ii = 0; ii < FPGAI_DENSE_BWD_IN_UNROLL; ++ii) {
#pragma HLS UNROLL
                        const int i = i0 + ii;
                        if (i < IN) dW[o * IN + i] = (G_T)(gy * (acc_t)x[i]);
                    }
                }
            }
        }
    }
}

template<int IN, int OUT>
void dense_weight_grad(
    const act_t x[IN],
    const grad_act_t dY[OUT],
    grad_wgt_t dW[OUT * IN]
) {
#pragma HLS INLINE off
    dense_weight_grad_typed<IN, OUT, act_t, grad_act_t, grad_wgt_t>(x, dY, dW);
}

template<int OUT, typename GRAD_T = grad_act_t, typename G_T = grad_bias_t>
void dense_bias_grad_typed(
    const GRAD_T dY[OUT],
    G_T dB[OUT]
) {
#pragma HLS INLINE off
#if FPGAI_DENSE_PARTITION_GRADS > 1
#pragma HLS ARRAY_PARTITION variable=dY cyclic factor=FPGAI_DENSE_PARTITION_GRADS
#pragma HLS ARRAY_PARTITION variable=dB cyclic factor=FPGAI_DENSE_PARTITION_GRADS
#endif

    for (int o0 = 0; o0 < OUT; o0 += FPGAI_DENSE_BWD_OUT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        for (int oo = 0; oo < FPGAI_DENSE_BWD_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
            const int o = o0 + oo;
            if (o < OUT) dB[o] = (G_T)dY[o];
        }
    }
}

template<int OUT>
void dense_bias_grad(
    const grad_act_t dY[OUT],
    grad_bias_t dB[OUT]
) {
#pragma HLS INLINE off
    dense_bias_grad_typed<OUT, grad_act_t, grad_bias_t>(dY, dB);
}

template<int N, typename W_T = wgt_t, typename G_T = grad_wgt_t, typename LR_T = upd_t>
void sgd_update_wgt_typed(
    W_T W[N],
    const G_T dW[N],
    const LR_T lr
) {
#pragma HLS INLINE off
    for (int i0 = 0; i0 < N; i0 += FPGAI_DENSE_UPD_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        for (int uu = 0; uu < FPGAI_DENSE_UPD_UNROLL; ++uu) {
#pragma HLS UNROLL
            const int i = i0 + uu;
            if (i < N) {
                acc_t tmp = (acc_t)W[i] - (acc_t)lr * (acc_t)dW[i];
                W[i] = (W_T)tmp;
            }
        }
    }
}

template<int N>
void sgd_update_wgt(
    wgt_t W[N],
    const grad_wgt_t dW[N],
    const upd_t lr
) {
#pragma HLS INLINE off
    sgd_update_wgt_typed<N, wgt_t, grad_wgt_t, upd_t>(W, dW, lr);
}

template<int N, typename B_T = bias_t, typename G_T = grad_bias_t, typename LR_T = upd_t>
void sgd_update_bias_typed(
    B_T B[N],
    const G_T dB[N],
    const LR_T lr
) {
#pragma HLS INLINE off
    for (int i0 = 0; i0 < N; i0 += FPGAI_DENSE_UPD_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        for (int uu = 0; uu < FPGAI_DENSE_UPD_UNROLL; ++uu) {
#pragma HLS UNROLL
            const int i = i0 + uu;
            if (i < N) {
                acc_t tmp = (acc_t)B[i] - (acc_t)lr * (acc_t)dB[i];
                B[i] = (B_T)tmp;
            }
        }
    }
}

template<int N>
void sgd_update_bias(
    bias_t B[N],
    const grad_bias_t dB[N],
    const upd_t lr
) {
#pragma HLS INLINE off
    sgd_update_bias_typed<N, bias_t, grad_bias_t, upd_t>(B, dB, lr);
}

template<int N>
void mse_loss_grad(
    const act_t pred[N],
    const act_t target[N],
    grad_act_t dY[N],
    loss_t &loss_out
) {
#pragma HLS INLINE off
    loss_out = (loss_t)0;
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
        acc_t diff = (acc_t)pred[i] - (acc_t)target[i];
        dY[i] = (grad_act_t)(((acc_t)2 / (acc_t)N) * diff);
        loss_out += (loss_t)(diff * diff);
    }
    loss_out = (loss_t)(loss_out / (loss_t)N);
}

} // namespace fpgai
'''


def emit_dense_cpp() -> str:
    return '#include "layers/dense.h"\n'