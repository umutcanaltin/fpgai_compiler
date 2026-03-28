from __future__ import annotations


def emit_conv_h() -> str:
    return r'''#pragma once
#include "fpgai_types.h"

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

    OC_TILE:
    for (int oc0 = 0; oc0 < OUT_C; oc0 += FPGAI_CONV_OC_UNROLL) {
        OH_LOOP:
        for (int oh = 0; oh < OUT_H; ++oh) {
            OW_LOOP:
            for (int ow = 0; ow < OUT_W; ++ow) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

                ACC_T acc[FPGAI_CONV_OC_UNROLL];
#pragma HLS ARRAY_PARTITION variable=acc complete

                INIT_ACC:
                for (int oco = 0; oco < FPGAI_CONV_OC_UNROLL; ++oco) {
#pragma HLS UNROLL
                    int oc = oc0 + oco;
                    if (oc < OUT_C) acc[oco] = (ACC_T)B[oc];
                }

                IC_LOOP:
                for (int ic = 0; ic < IN_C; ++ic) {
                    KH_LOOP:
                    for (int kh = 0; kh < K; ++kh) {
                        KW_LOOP:
                        for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE II=1
                            int ih = oh * STRIDE + kh - PAD;
                            int iw = ow * STRIDE + kw - PAD;

                            if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                                int x_idx = (ih * IN_W + iw) * IN_C + ic;
                                ACT_T xv = x[x_idx];

                                OC_ACC:
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

                WRITE_OUT:
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

} // namespace fpgai
'''


def emit_conv_cpp() -> str:
    return '#include "layers/conv.h"\n'