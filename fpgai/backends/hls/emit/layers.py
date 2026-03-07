from __future__ import annotations


def emit_dense_h() -> str:
    return r"""\
#pragma once

#include "../fpgai_types.h"

namespace fpgai {

// OUT x IN weights (out_in)
template<int IN, int OUT>
void dense_out_in(
    const act_t x[IN],
    act_t y[OUT],
    const wgt_t W[OUT][IN],
    const bias_t B[OUT]
) {
#pragma HLS INLINE
    for (int o = 0; o < OUT; o++) {
#pragma HLS PIPELINE II=1
        acc_t acc = (acc_t)B[o];
        for (int i = 0; i < IN; i++) {
            acc += (acc_t)x[i] * (acc_t)W[o][i];
        }
        y[o] = (act_t)acc;
    }
}

} // namespace fpgai
"""


def emit_dense_cpp() -> str:
    # Template defined in header; cpp can be empty translation unit or include header.
    return r"""\
#include "dense.h"
"""


def emit_activations_h() -> str:
    return r"""\
#pragma once

#include "../fpgai_types.h"

namespace fpgai {

template<int N>
void relu_inplace(act_t x[N]) {
#pragma HLS INLINE
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        x[i] = (x[i] > 0) ? x[i] : (act_t)0;
    }
}

template<int N>
void leaky_relu_inplace(act_t x[N], float alpha) {
#pragma HLS INLINE
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        x[i] = (x[i] > 0) ? x[i] : (act_t)(x[i] * (act_t)alpha);
    }
}

} // namespace fpgai
"""


def emit_activations_cpp() -> str:
    # Template defined in header; cpp can be empty translation unit or include header.
    return r"""\
#include "activations.h"
"""
