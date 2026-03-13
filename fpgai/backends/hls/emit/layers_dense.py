def emit_dense_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"

namespace fpgai {

// -----------------------------------------------------------------------------
// New typed kernel
// -----------------------------------------------------------------------------
template<typename ACT_T, typename WGT_T, typename BIAS_T, typename ACC_T, int OUT, int IN>
void dense_out_in_typed(
    const ACT_T x[IN],
    ACT_T y[OUT],
    const WGT_T W[OUT][IN],
    const BIAS_T B[OUT]
) {
#pragma HLS INLINE off

OUT_LOOP:
    for (int o = 0; o < OUT; o++) {
#pragma HLS PIPELINE II=1
        ACC_T acc = (ACC_T)B[o];

    IN_LOOP:
        for (int i = 0; i < IN; i++) {
            acc += (ACC_T)x[i] * (ACC_T)W[o][i];
        }

        y[o] = (ACT_T)acc;
    }
}

// -----------------------------------------------------------------------------
// Backward-compatible wrapper for old top emitter:
//
// old call style:
//   dense_out_in<IN, OUT>(x, y, (const wgt_t (*)[IN])Wk, Bk);
// -----------------------------------------------------------------------------
template<int IN, int OUT, typename BIAS_T>
void dense_out_in(
    const act_t x[IN],
    act_t y[OUT],
    const wgt_t W[OUT][IN],
    const BIAS_T B[OUT]
) {
    dense_out_in_typed<act_t, wgt_t, BIAS_T, acc_t, OUT, IN>(x, y, W, B);
}

} // namespace fpgai
'''


def emit_dense_cpp() -> str:
    return r'''
#include "layers/dense.h"
'''