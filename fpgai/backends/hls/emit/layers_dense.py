from __future__ import annotations


def emit_dense_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"

namespace fpgai {

template<int IN, int OUT, typename BiasT = bias_t>
void dense_out_in(
    const act_t x[IN],
    act_t y[OUT],
    const wgt_t W[OUT][IN],
    const BiasT B[OUT]
) {
#pragma HLS INLINE off

OUT_LOOP:
    for (int o = 0; o < OUT; o++) {
        acc_t acc = (acc_t)B[o];

    IN_LOOP:
        for (int i = 0; i < IN; i++) {
#if FPGAI_DENSE_IN_UNROLL <= 1
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL off
#else
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=FPGAI_DENSE_IN_UNROLL
#endif
            acc += (acc_t)x[i] * (acc_t)W[o][i];
        }

        y[o] = (act_t)acc;
    }
}

} // namespace fpgai
'''


def emit_dense_cpp() -> str:
    return '#include "layers/dense.h"\n'