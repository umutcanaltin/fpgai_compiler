def emit_pool_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"

namespace fpgai {

// -----------------------------------------------------------------------------
// New typed kernels with explicit NHWC-shaped arrays
// -----------------------------------------------------------------------------
template<typename ACT_T, int H, int W, int C, int KH, int KW, int OH, int OW, int SH, int SW>
void maxpool2d_nhwc(
    const ACT_T x[H][W][C],
    ACT_T y[OH][OW][C]
) {
    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            for (int c = 0; c < C; c++) {
#pragma HLS PIPELINE II=1
                ACT_T m = x[oh * SH][ow * SW][c];
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        ACT_T v = x[oh * SH + kh][ow * SW + kw][c];
                        if (v > m) m = v;
                    }
                }
                y[oh][ow][c] = m;
            }
        }
    }
}

template<typename ACT_T, typename ACC_T, int H, int W, int C, int KH, int KW, int OH, int OW, int SH, int SW>
void avgpool2d_nhwc(
    const ACT_T x[H][W][C],
    ACT_T y[OH][OW][C]
) {
    const int DEN = KH * KW;
    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            for (int c = 0; c < C; c++) {
#pragma HLS PIPELINE II=1
                ACC_T acc = 0;
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        acc += (ACC_T)x[oh * SH + kh][ow * SW + kw][c];
                    }
                }
                y[oh][ow][c] = (ACT_T)(acc / (ACC_T)DEN);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Backward-compatible flat-buffer wrappers for current deeplearn.cpp
//
// old call style:
//   maxpool2d<H,W,C,KH,KW,OH,OW>(x_flat, y_flat)
//   avgpool2d<H,W,C,KH,KW,OH,OW>(x_flat, y_flat)
//
// implicit stride = kernel size
// flattening = NHWC
// -----------------------------------------------------------------------------
template<int H, int W, int C, int KH, int KW, int OH, int OW>
void maxpool2d(
    const act_t x[H * W * C],
    act_t y[OH * OW * C]
) {
    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            for (int c = 0; c < C; c++) {
#pragma HLS PIPELINE II=1
                int base_h = oh * KH;
                int base_w = ow * KW;
                int first_idx = ((base_h * W) + base_w) * C + c;
                act_t m = x[first_idx];

                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int ih = base_h + kh;
                        int iw = base_w + kw;
                        int idx = ((ih * W) + iw) * C + c;
                        act_t v = x[idx];
                        if (v > m) m = v;
                    }
                }

                int out_idx = ((oh * OW) + ow) * C + c;
                y[out_idx] = m;
            }
        }
    }
}

template<int H, int W, int C, int KH, int KW, int OH, int OW>
void avgpool2d(
    const act_t x[H * W * C],
    act_t y[OH * OW * C]
) {
    const int DEN = KH * KW;
    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            for (int c = 0; c < C; c++) {
#pragma HLS PIPELINE II=1
                acc_t acc = 0;
                int base_h = oh * KH;
                int base_w = ow * KW;

                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int ih = base_h + kh;
                        int iw = base_w + kw;
                        int idx = ((ih * W) + iw) * C + c;
                        acc += (acc_t)x[idx];
                    }
                }

                int out_idx = ((oh * OW) + ow) * C + c;
                y[out_idx] = (act_t)(acc / (acc_t)DEN);
            }
        }
    }
}

} // namespace fpgai
'''


def emit_pool_cpp() -> str:
    return r'''
#include "layers/pool.h"
'''