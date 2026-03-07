def emit_dense_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"

namespace fpgai {

// OUT_IN layout: W[OUT][IN]
template<int IN, int OUT>
inline void dense_out_in(
    const act_t x[IN],
    act_t y[OUT],
    const wgt_t W[OUT][IN],
    const bias_t B[OUT]
) {
#pragma HLS INLINE
    for (int o = 0; o < OUT; o++) {
#pragma HLS PIPELINE II=1
        // Use accumulator type for sum
        acc_t acc = (acc_t)B[o];
        for (int i = 0; i < IN; i++) {
            acc += (acc_t)x[i] * (acc_t)W[o][i];
        }
        y[o] = (act_t)acc;
    }
}

} // namespace fpgai
'''

def emit_dense_cpp() -> str:
    return r'''
#include "layers/dense.h"
'''