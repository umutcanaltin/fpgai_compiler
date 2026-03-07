def emit_pool_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"

namespace fpgai {

// MaxPool2D
template<int H, int W, int C, int K, int STRIDE, int HO, int WO>
void maxpool2d(act_t* in, act_t* out) {
    #pragma HLS INLINE off

    for(int h=0; h<HO; h++) {
        for(int w=0; w<WO; w++) {
            for(int c=0; c<C; c++) {
                #pragma HLS PIPELINE
                
                // FIX: Init to -128.0 (min for ap_fixed<16,8>)
                // -1e9 causes overflow/garbage in fixed-point.
                act_t max_val = (act_t)-128.0; 

                for(int kh=0; kh<K; kh++) {
                    for(int kw=0; kw<K; kw++) {
                        int ih = h*STRIDE + kh;
                        int iw = w*STRIDE + kw;
                        if(ih < H && iw < W) {
                            act_t val = in[(ih*W + iw)*C + c];
                            if(val > max_val) max_val = val;
                        }
                    }
                }
                out[(h*WO + w)*C + c] = max_val;
            }
        }
    }
}

// AvgPool2D
template<int H, int W, int C, int K, int STRIDE, int HO, int WO>
void avgpool2d(act_t* in, act_t* out) {
    #pragma HLS INLINE off

    for(int h=0; h<HO; h++) {
        for(int w=0; w<WO; w++) {
            for(int c=0; c<C; c++) {
                #pragma HLS PIPELINE
                acc_t sum = 0;
                for(int kh=0; kh<K; kh++) {
                    for(int kw=0; kw<K; kw++) {
                        int ih = h*STRIDE + kh;
                        int iw = w*STRIDE + kw;
                        if(ih < H && iw < W) {
                             sum += (acc_t)in[(ih*W + iw)*C + c];
                        }
                    }
                }
                out[(h*WO + w)*C + c] = (act_t)(sum / (act_t)(K*K));
            }
        }
    }
}

} // namespace
'''

def emit_pool_cpp() -> str:
    return '#include "layers/pool.h"\n'