from __future__ import annotations


def emit_conv_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"

namespace fpgai {

template<
    int H, int W, int CIN,
    int OH, int OW, int COUT,
    int K, int STRIDE, int PAD,
    typename WGT_T, typename BIAS_T
>
void conv2d(
    const act_t x[H * W * CIN],
    act_t y[OH * OW * COUT],
    const WGT_T w[COUT * CIN * K * K],
    const BIAS_T b[COUT]
) {
#pragma HLS INLINE off

OUT_H:
    for (int oh = 0; oh < OH; oh++) {
    OUT_W:
        for (int ow = 0; ow < OW; ow++) {
        OUT_C:
            for (int co = 0; co < COUT; co++) {
#pragma HLS PIPELINE II=1
                acc_t acc = (acc_t)b[co];

            K_H:
                for (int kh = 0; kh < K; kh++) {
                K_W:
                    for (int kw = 0; kw < K; kw++) {
                    IN_C:
                        for (int ci = 0; ci < CIN; ci++) {
#pragma HLS UNROLL factor=FPGAI_CONV_IC_UNROLL
                            int ih = oh * STRIDE + kh - PAD;
                            int iw = ow * STRIDE + kw - PAD;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int x_idx = ((ih * W) + iw) * CIN + ci;
                                int w_idx = (((co * CIN) + ci) * K + kh) * K + kw;
                                acc += (acc_t)x[x_idx] * (acc_t)w[w_idx];
                            }
                        }
                    }
                }

                int y_idx = ((oh * OW) + ow) * COUT + co;
                y[y_idx] = (act_t)acc;
            }
        }
    }
}

} // namespace fpgai
'''


def emit_conv_cpp() -> str:
    return '#include "layers/conv.h"\n'