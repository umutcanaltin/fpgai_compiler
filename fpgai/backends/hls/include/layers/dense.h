#pragma once

#include <hls_stream.h>

namespace fpgai {

template<typename T>
static inline T relu_apply(T x) {
#pragma HLS INLINE
    return (x > (T)0) ? x : (T)0;
}

template<typename T>
static inline T relu_grad_from_output(T y, T gy) {
#pragma HLS INLINE
    return (y > (T)0) ? gy : (T)0;
}

template<
    typename ACT_T,
    typename WGT_T,
    typename BIAS_T,
    typename ACC_T,
    int IN_F,
    int OUT_F,
    int PE = 1,
    int SIMD = 1
>
void dense_forward(
    const ACT_T x[IN_F],
    ACT_T y[OUT_F],
    const WGT_T W[OUT_F * IN_F],
    const BIAS_T B[OUT_F]
) {
#pragma HLS INLINE off

    const int PE_EFF = (PE > 0) ? PE : 1;
    const int SIMD_EFF = (SIMD > 0) ? SIMD : 1;

    for (int o0 = 0; o0 < OUT_F; o0 += PE_EFF) {
#pragma HLS PIPELINE II=1
        ACC_T acc[PE_EFF];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

        for (int po = 0; po < PE_EFF; ++po) {
#pragma HLS UNROLL
            int o = o0 + po;
            acc[po] = (o < OUT_F) ? (ACC_T)B[o] : (ACC_T)0;
        }

        for (int i0 = 0; i0 < IN_F; i0 += SIMD_EFF) {
            for (int po = 0; po < PE_EFF; ++po) {
#pragma HLS UNROLL
                int o = o0 + po;
                if (o < OUT_F) {
                    for (int si = 0; si < SIMD_EFF; ++si) {
#pragma HLS UNROLL
                        int i = i0 + si;
                        if (i < IN_F) {
                            int w_idx = o * IN_F + i;
                            acc[po] += (ACC_T)x[i] * (ACC_T)W[w_idx];
                        }
                    }
                }
            }
        }

        for (int po = 0; po < PE_EFF; ++po) {
#pragma HLS UNROLL
            int o = o0 + po;
            if (o < OUT_F) y[o] = (ACT_T)acc[po];
        }
    }
}

template<
    typename ACT_T,
    typename WGT_T,
    typename GRAD_T,
    typename ACC_T,
    int IN_F,
    int OUT_F,
    int PE = 1,
    int SIMD = 1
>
void dense_backward_input(
    const GRAD_T gy[OUT_F],
    GRAD_T gx[IN_F],
    const WGT_T W[OUT_F * IN_F]
) {
#pragma HLS INLINE off

    for (int i = 0; i < IN_F; ++i) {
#pragma HLS PIPELINE II=1
        ACC_T acc = 0;
        for (int o = 0; o < OUT_F; ++o) {
            int w_idx = o * IN_F + i;
            acc += (ACC_T)gy[o] * (ACC_T)W[w_idx];
        }
        gx[i] += (GRAD_T)acc;
    }
}

template<
    typename ACT_T,
    typename GRAD_T,
    typename GWT_T,
    typename GBIAS_T,
    int IN_F,
    int OUT_F,
    int PE = 1,
    int SIMD = 1
>
void dense_backward_params(
    const ACT_T x[IN_F],
    const GRAD_T gy[OUT_F],
    GWT_T dW[OUT_F * IN_F],
    GBIAS_T dB[OUT_F]
) {
#pragma HLS INLINE off

    for (int o = 0; o < OUT_F; ++o) {
#pragma HLS PIPELINE II=1
        dB[o] += (GBIAS_T)gy[o];
        for (int i = 0; i < IN_F; ++i) {
            int w_idx = o * IN_F + i;
            dW[w_idx] += (GWT_T)((GWT_T)gy[o] * (GWT_T)x[i]);
        }
    }
}

template<
    typename WGT_T,
    typename BIAS_T,
    typename GWT_T,
    typename GBIAS_T,
    int IN_F,
    int OUT_F,
    int UPDATE_UNROLL = 1
>
void dense_update_sgd(
    WGT_T W[OUT_F * IN_F],
    BIAS_T B[OUT_F],
    const GWT_T dW[OUT_F * IN_F],
    const GBIAS_T dB[OUT_F],
    float lr
) {
#pragma HLS INLINE off

    const int UW = (UPDATE_UNROLL > 0) ? UPDATE_UNROLL : 1;

    for (int idx0 = 0; idx0 < OUT_F * IN_F; idx0 += UW) {
#pragma HLS PIPELINE II=1
        for (int u = 0; u < UW; ++u) {
#pragma HLS UNROLL
            int idx = idx0 + u;
            if (idx < OUT_F * IN_F) {
                W[idx] = (WGT_T)((float)W[idx] - lr * (float)dW[idx]);
            }
        }
    }

    for (int o = 0; o < OUT_F; ++o) {
#pragma HLS PIPELINE II=1
        B[o] = (BIAS_T)((float)B[o] - lr * (float)dB[o]);
    }
}

template<
    typename ACT_T,
    typename WGT_T,
    typename BIAS_T,
    typename GRAD_T,
    typename GWT_T,
    typename GBIAS_T,
    typename ACC_T,
    int IN_F,
    int OUT_F,
    int PE = 1,
    int SIMD = 1,
    int UPDATE_UNROLL = 1
>
void dense_train_step(
    const ACT_T x[IN_F],
    const GRAD_T gy[OUT_F],
    GRAD_T gx[IN_F],
    WGT_T W[OUT_F * IN_F],
    BIAS_T B[OUT_F],
    GWT_T dW[OUT_F * IN_F],
    GBIAS_T dB[OUT_F],
    float lr
) {
#pragma HLS INLINE off
    dense_backward_params<ACT_T, GRAD_T, GWT_T, GBIAS_T, IN_F, OUT_F, PE, SIMD>(x, gy, dW, dB);
    dense_backward_input<ACT_T, WGT_T, GRAD_T, ACC_T, IN_F, OUT_F, PE, SIMD>(gy, gx, W);
    dense_update_sgd<WGT_T, BIAS_T, GWT_T, GBIAS_T, IN_F, OUT_F, UPDATE_UNROLL>(W, B, dW, dB, lr);
}

} // namespace fpgai