from __future__ import annotations


def emit_pool_h() -> str:
    return r'''#pragma once
#include "fpgai_types.h"

#ifndef FPGAI_PIPELINE_II
#define FPGAI_PIPELINE_II 1
#endif

namespace fpgai {

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

template<int IN_H, int IN_W, int IN_C, int K, int STRIDE, int OUT_H, int OUT_W>
void avgpool2d(const act_t* x, act_t* y) {
#pragma HLS INLINE
    avgpool2d<IN_H, IN_W, IN_C, K, STRIDE>(x, y);
}

template<int IN_H, int IN_W, int IN_C, int K, int STRIDE>
void maxpool2d_backward(const act_t* x, const act_t* y, const grad_act_t* dY, grad_act_t* dX) {
#pragma HLS INLINE off
    const int OUT_H = IN_H / STRIDE;
    const int OUT_W = IN_W / STRIDE;

INIT_DX:
    for (int i = 0; i < IN_H * IN_W * IN_C; ++i) {
#pragma HLS PIPELINE II=1
        dX[i] = (grad_act_t)0;
    }

    for (int oh = 0; oh < OUT_H; ++oh) {
        for (int ow = 0; ow < OUT_W; ++ow) {
            for (int c = 0; c < IN_C; ++c) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
                const int out_idx = (oh * OUT_W + ow) * IN_C + c;
                const act_t pooled = y[out_idx];
                bool routed = false;

                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        const int ih = oh * STRIDE + kh;
                        const int iw = ow * STRIDE + kw;
                        const int in_idx = (ih * IN_W + iw) * IN_C + c;
                        if (!routed && x[in_idx] == pooled) {
                            dX[in_idx] += dY[out_idx];
                            routed = true;
                        }
                    }
                }
            }
        }
    }
}

template<int IN_H, int IN_W, int IN_C, int K, int STRIDE>
void avgpool2d_backward(const grad_act_t* dY, grad_act_t* dX) {
#pragma HLS INLINE off
    const int OUT_H = IN_H / STRIDE;
    const int OUT_W = IN_W / STRIDE;
    const acc_t scale = (acc_t)1 / (acc_t)(K * K);

INIT_DX:
    for (int i = 0; i < IN_H * IN_W * IN_C; ++i) {
#pragma HLS PIPELINE II=1
        dX[i] = (grad_act_t)0;
    }

    for (int oh = 0; oh < OUT_H; ++oh) {
        for (int ow = 0; ow < OUT_W; ++ow) {
            for (int c = 0; c < IN_C; ++c) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
                const int out_idx = (oh * OUT_W + ow) * IN_C + c;
                const grad_act_t g = (grad_act_t)((acc_t)dY[out_idx] * scale);
                for (int kh = 0; kh < K; ++kh) {
                    for (int kw = 0; kw < K; ++kw) {
                        const int ih = oh * STRIDE + kh;
                        const int iw = ow * STRIDE + kw;
                        const int in_idx = (ih * IN_W + iw) * IN_C + c;
                        dX[in_idx] += g;
                    }
                }
            }
        }
    }
}

} // namespace fpgai
'''


def emit_pool_cpp() -> str:
    return '#include "layers/pool.h"\n'