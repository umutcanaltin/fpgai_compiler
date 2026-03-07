def emit_conv_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"

namespace fpgai {

// Standard 2D Convolution
template<
    int IN_H, int IN_W, int IN_C,
    int OUT_H, int OUT_W, int OUT_C,
    int K, int STRIDE, int PAD
>
void conv2d(
    const act_t* x,     // Flat Input
    act_t* y,           // Flat Output
    const wgt_t* W,     // Flat Weights
    const bias_t* B     // Flat Bias
) {
    #pragma HLS INLINE off

    for(int oh = 0; oh < OUT_H; oh++) {
        for(int ow = 0; ow < OUT_W; ow++) {
            for(int oc = 0; oc < OUT_C; oc++) {
                #pragma HLS PIPELINE II=1
                
                acc_t acc = (acc_t)B[oc];

                for(int kh = 0; kh < K; kh++) {
                    for(int kw = 0; kw < K; kw++) {
                        for(int ic = 0; ic < IN_C; ic++) {
                            
                            int ih = oh * STRIDE + kh - PAD;
                            int iw = ow * STRIDE + kw - PAD;

                            if(ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                                // Linear Indexing for Input
                                int in_idx = (ih * IN_W + iw) * IN_C + ic;
                                
                                // Linear Indexing for Weights (Out, In, K, K)
                                int w_idx = ((oc * IN_C + ic) * K + kh) * K + kw;
                                
                                acc += (acc_t)x[in_idx] * (acc_t)W[w_idx];
                            }
                        }
                    }
                }
                
                int out_idx = (oh * OUT_W + ow) * OUT_C + oc;
                y[out_idx] = (act_t)acc;
            }
        }
    }
}

} // namespace
'''

def emit_conv_cpp() -> str:
    return '#include "layers/conv.h"\n'