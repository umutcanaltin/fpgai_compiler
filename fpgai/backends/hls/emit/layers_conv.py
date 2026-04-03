from __future__ import annotations


def emit_conv_h() -> str:
    return r'''#pragma once
#include "fpgai_types.h"

#ifndef FPGAI_PIPELINE_II
#define FPGAI_PIPELINE_II 1
#endif

#ifndef FPGAI_CONV_OC_UNROLL
#define FPGAI_CONV_OC_UNROLL 1
#endif

#ifndef FPGAI_CONV_IC_UNROLL
#define FPGAI_CONV_IC_UNROLL 1
#endif

namespace fpgai {

template<
    int IN_H, int IN_W, int IN_C,
    int OUT_H, int OUT_W, int OUT_C,
    int K, int STRIDE, int PAD,
    typename ACT_T = act_t,
    typename WGT_T = wgt_t,
    typename BIAS_T = bias_t,
    typename ACC_T = acc_t
>
void conv2d(
    const ACT_T x[IN_H * IN_W * IN_C],
    ACT_T y[OUT_H * OUT_W * OUT_C],
    const WGT_T W[OUT_C * IN_C * K * K],
    const BIAS_T B[OUT_C]
) {
#pragma HLS INLINE off
    for (int oc0 = 0; oc0 < OUT_C; oc0 += FPGAI_CONV_OC_UNROLL) {
        for (int oh = 0; oh < OUT_H; ++oh) {
            for (int ow = 0; ow < OUT_W; ++ow) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
                ACC_T acc[FPGAI_CONV_OC_UNROLL];
#pragma HLS ARRAY_PARTITION variable=acc complete

                for (int oco = 0; oco < FPGAI_CONV_OC_UNROLL; ++oco) {
#pragma HLS UNROLL
                    int oc = oc0 + oco;
                    if (oc < OUT_C) acc[oco] = (ACC_T)B[oc];
                }

                for (int ic0 = 0; ic0 < IN_C; ic0 += FPGAI_CONV_IC_UNROLL) {
                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            for (int ici = 0; ici < FPGAI_CONV_IC_UNROLL; ++ici) {
#pragma HLS UNROLL
                                int ic = ic0 + ici;
                                if (ic < IN_C) {
                                    int ih = oh * STRIDE + kh - PAD;
                                    int iw = ow * STRIDE + kw - PAD;
                                    if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                                        int x_idx = (ih * IN_W + iw) * IN_C + ic;
                                        ACT_T xv = x[x_idx];
                                        for (int oco = 0; oco < FPGAI_CONV_OC_UNROLL; ++oco) {
#pragma HLS UNROLL
                                            int oc = oc0 + oco;
                                            if (oc < OUT_C) {
                                                int w_idx = (((oc * IN_C + ic) * K + kh) * K + kw);
                                                acc[oco] += (ACC_T)xv * (ACC_T)W[w_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                for (int oco = 0; oco < FPGAI_CONV_OC_UNROLL; ++oco) {
#pragma HLS UNROLL
                    int oc = oc0 + oco;
                    if (oc < OUT_C) {
                        int y_idx = (oh * OUT_W + ow) * OUT_C + oc;
                        y[y_idx] = (ACT_T)acc[oco];
                    }
                }
            }
        }
    }
}

template<
    int IN_H, int IN_W, int IN_C,
    int OUT_H, int OUT_W, int OUT_C,
    int K, int STRIDE, int PAD
>
void conv2d_backward_input(
    const grad_act_t dY[OUT_H * OUT_W * OUT_C],
    const wgt_t W[OUT_C * IN_C * K * K],
    grad_act_t dX[IN_H * IN_W * IN_C]
) {
#pragma HLS INLINE off
    for (int i = 0; i < IN_H * IN_W * IN_C; ++i) {
#pragma HLS PIPELINE II=1
        dX[i] = (grad_act_t)0;
    }

    for (int oh = 0; oh < OUT_H; ++oh) {
        for (int ow = 0; ow < OUT_W; ++ow) {
            for (int oc = 0; oc < OUT_C; ++oc) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
                const int y_idx = (oh * OUT_W + ow) * OUT_C + oc;
                const acc_t gy = (acc_t)dY[y_idx];
                for (int ic = 0; ic < IN_C; ++ic) {
                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            const int ih = oh * STRIDE + kh - PAD;
                            const int iw = ow * STRIDE + kw - PAD;
                            if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                                const int x_idx = (ih * IN_W + iw) * IN_C + ic;
                                const int w_idx = (((oc * IN_C + ic) * K + kh) * K + kw);
                                dX[x_idx] += (grad_act_t)(gy * (acc_t)W[w_idx]);
                            }
                        }
                    }
                }
            }
        }
    }
}

template<
    int IN_H, int IN_W, int IN_C,
    int OUT_H, int OUT_W, int OUT_C,
    int K, int STRIDE, int PAD
>
void conv2d_weight_grad(
    const act_t x[IN_H * IN_W * IN_C],
    const grad_act_t dY[OUT_H * OUT_W * OUT_C],
    grad_wgt_t dW[OUT_C * IN_C * K * K]
) {
#pragma HLS INLINE off
    for (int i = 0; i < OUT_C * IN_C * K * K; ++i) {
#pragma HLS PIPELINE II=1
        dW[i] = (grad_wgt_t)0;
    }

    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int ic = 0; ic < IN_C; ++ic) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
                    acc_t acc = 0;
                    for (int oh = 0; oh < OUT_H; ++oh) {
                        for (int ow = 0; ow < OUT_W; ++ow) {
                            const int ih = oh * STRIDE + kh - PAD;
                            const int iw = ow * STRIDE + kw - PAD;
                            if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                                const int x_idx = (ih * IN_W + iw) * IN_C + ic;
                                const int y_idx = (oh * OUT_W + ow) * OUT_C + oc;
                                acc += (acc_t)x[x_idx] * (acc_t)dY[y_idx];
                            }
                        }
                    }
                    dW[(((oc * IN_C + ic) * K + kh) * K + kw)] = (grad_wgt_t)acc;
                }
            }
        }
    }
}

template<int OUT_C, int OUT_H, int OUT_W>
void conv2d_bias_grad(
    const grad_act_t dY[OUT_H * OUT_W * OUT_C],
    grad_bias_t dB[OUT_C]
) {
#pragma HLS INLINE off
    for (int oc = 0; oc < OUT_C; ++oc) {
#pragma HLS PIPELINE II=1
        dB[oc] = (grad_bias_t)0;
    }

    for (int oh = 0; oh < OUT_H; ++oh) {
        for (int ow = 0; ow < OUT_W; ++ow) {
            for (int oc = 0; oc < OUT_C; ++oc) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
                const int y_idx = (oh * OUT_W + ow) * OUT_C + oc;
                dB[oc] += (grad_bias_t)dY[y_idx];
            }
        }
    }
}

} // namespace fpgai
'''


def emit_conv_cpp() -> str:
    return '#include "layers/conv.h"\n'