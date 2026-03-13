def emit_conv_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"

namespace fpgai {

// -----------------------------------------------------------------------------
// New typed kernel with explicit NHWC-shaped arrays
// -----------------------------------------------------------------------------
template<
    typename ACT_T,
    typename WGT_T,
    typename BIAS_T,
    typename ACC_T,
    int H, int W, int CIN,
    int KH, int KW,
    int COUT,
    int OH, int OW,
    int STRIDE_H, int STRIDE_W,
    int PAD_H, int PAD_W
>
void conv2d_nhwc_typed(
    const ACT_T x[H][W][CIN],
    ACT_T y[OH][OW][COUT],
    const WGT_T w[COUT][KH][KW][CIN],
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
                ACC_T acc = (ACC_T)b[co];

            K_H:
                for (int kh = 0; kh < KH; kh++) {
                K_W:
                    for (int kw = 0; kw < KW; kw++) {
                    IN_C:
                        for (int ci = 0; ci < CIN; ci++) {
                            int ih = oh * STRIDE_H + kh - PAD_H;
                            int iw = ow * STRIDE_W + kw - PAD_W;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                acc += (ACC_T)x[ih][iw][ci] * (ACC_T)w[co][kh][kw][ci];
                            }
                        }
                    }
                }

                y[oh][ow][co] = (ACT_T)acc;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Backward-compatible flat-buffer wrapper for current deeplearn.cpp
//
// old call style:
//   conv2d<H,W,CIN,OH,OW,COUT,K,STRIDE,PAD>(x_flat, y_flat, W_flat, B)
//
// buffer layouts:
//   x_flat : [H*W*CIN]        with NHWC flattening
//   y_flat : [OH*OW*COUT]     with NHWC flattening
//   W_flat : [COUT*CIN*K*K]   with layout [co][ci][kh][kw]
// -----------------------------------------------------------------------------
template<
    int H, int W, int CIN,
    int OH, int OW,
    int COUT,
    int K,
    int STRIDE,
    int PAD,
    typename WGT_T,
    typename BIAS_T
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
    return r'''
#include "layers/conv.h"
'''