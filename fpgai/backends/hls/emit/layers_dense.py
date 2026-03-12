from __future__ import annotations


def emit_dense_h() -> str:
    return r'''
#pragma once
#include "fpgai_types.h"

namespace fpgai {

// Reference-style Dense layer:
// y[o] = B[o] + sum_i x[i] * W[o][i]
//
// This version is intentionally conservative and easy to verify.
// It avoids aggressive unrolling that can hide bugs or introduce
// unexpected HLS behavior during early compiler validation.
template<int IN, int OUT>
void dense_out_in(
    const act_t x[IN],
    act_t y[OUT],
    const wgt_t W[OUT][IN],
    const bias_t B[OUT]
) {
    #pragma HLS INLINE off

    OUT_LOOP:
    for (int o = 0; o < OUT; o++) {
        #pragma HLS PIPELINE II=1

        acc_t acc = (acc_t)B[o];

        IN_LOOP:
        for (int i = 0; i < IN; i++) {
            acc += (acc_t)x[i] * (acc_t)W[o][i];
        }

        y[o] = (act_t)acc;
    }
}

} // namespace fpgai
'''


def emit_dense_cpp() -> str:
    return '#include "layers/dense.h"\n'