from __future__ import annotations


def emit_pool_h() -> str:
    return r'''#pragma once
#include "fpgai_types.h"

namespace fpgai {

// Core 5-parameter version
template<int IN_H, int IN_W, int IN_C, int K, int STRIDE>
void maxpool2d(const act_t* x, act_t* y) {
#pragma HLS INLINE off

    const int OUT_H = IN_H / STRIDE;
    const int OUT_W = IN_W / STRIDE;

    for (int oh = 0; oh < OUT_H; ++oh) {
        for (int ow = 0; ow < OUT_W; ++ow) {
            for (int c = 0; c < IN_C; ++c) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
                act_t best = x[((oh * STRIDE) * IN_W + (ow * STRIDE)) * IN_C + c];
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        const int ih = oh * STRIDE + kh;
                        const int iw = ow * STRIDE + kw;
                        const int idx = (ih * IN_W + iw) * IN_C + c;
                        if (x[idx] > best) best = x[idx];
                    }
                }
                const int out_idx = (oh * OUT_W + ow) * IN_C + c;
                y[out_idx] = best;
            }
        }
    }
}

// Backward-compatible 7-parameter version:
// <IN_H, IN_W, IN_C, K, STRIDE, OUT_H, OUT_W>
template<int IN_H, int IN_W, int IN_C, int K, int STRIDE, int OUT_H, int OUT_W>
void maxpool2d(const act_t* x, act_t* y) {
#pragma HLS INLINE
    maxpool2d<IN_H, IN_W, IN_C, K, STRIDE>(x, y);
}

template<int IN_H, int IN_W, int IN_C, int K, int STRIDE>
void avgpool2d(const act_t* x, act_t* y) {
#pragma HLS INLINE off

    const int OUT_H = IN_H / STRIDE;
    const int OUT_W = IN_W / STRIDE;

    for (int oh = 0; oh < OUT_H; ++oh) {
        for (int ow = 0; ow < OUT_W; ++ow) {
            for (int c = 0; c < IN_C; ++c) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
                acc_t sum = 0;
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        const int ih = oh * STRIDE + kh;
                        const int iw = ow * STRIDE + kw;
                        const int idx = (ih * IN_W + iw) * IN_C + c;
                        sum += (acc_t)x[idx];
                    }
                }
                const int out_idx = (oh * OUT_W + ow) * IN_C + c;
                y[out_idx] = (act_t)(sum / (K * K));
            }
        }
    }
}

// Backward-compatible 7-parameter version:
// <IN_H, IN_W, IN_C, K, STRIDE, OUT_H, OUT_W>
template<int IN_H, int IN_W, int IN_C, int K, int STRIDE, int OUT_H, int OUT_W>
void avgpool2d(const act_t* x, act_t* y) {
#pragma HLS INLINE
    avgpool2d<IN_H, IN_W, IN_C, K, STRIDE>(x, y);
}

} // namespace fpgai
'''


def emit_pool_cpp() -> str:
    return '#include "layers/pool.h"\n'