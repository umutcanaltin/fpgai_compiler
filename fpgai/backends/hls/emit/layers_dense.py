from __future__ import annotations


def emit_dense_h() -> str:
    return r'''#pragma once
#include "fpgai_types.h"

namespace fpgai {

template<int IN, int OUT, typename ACT_T = act_t, typename WGT_T = wgt_t, typename BIAS_T = bias_t, typename ACC_T = acc_t>
void dense_out_in_typed(
    const ACT_T x[IN],
    ACT_T y[OUT],
    const WGT_T W[OUT * IN],
    const BIAS_T B[OUT]
) {
#pragma HLS INLINE off

#if FPGAI_DENSE_OUT_UNROLL > 1
#pragma HLS ARRAY_PARTITION variable=y cyclic factor=FPGAI_DENSE_OUT_UNROLL
#pragma HLS ARRAY_PARTITION variable=B cyclic factor=FPGAI_DENSE_OUT_UNROLL
#endif

    OUT_TILE:
    for (int o0 = 0; o0 < OUT; o0 += FPGAI_DENSE_OUT_UNROLL) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        ACC_T acc[FPGAI_DENSE_OUT_UNROLL];
#pragma HLS ARRAY_PARTITION variable=acc complete

        INIT_ACC:
        for (int oo = 0; oo < FPGAI_DENSE_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
            int o = o0 + oo;
            if (o < OUT) acc[oo] = (ACC_T)B[o];
        }

        IN_LOOP:
        for (int i = 0; i < IN; ++i) {
#if FPGAI_DENSE_IN_UNROLL <= 1
#pragma HLS PIPELINE II=1
#else
#pragma HLS PIPELINE II=1
#endif
            MAC_OUT:
            for (int oo = 0; oo < FPGAI_DENSE_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
                int o = o0 + oo;
                if (o < OUT) {
                    const int w_idx = o * IN + i;
                    acc[oo] += (ACC_T)x[i] * (ACC_T)W[w_idx];
                }
            }
        }

        WRITE_OUT:
        for (int oo = 0; oo < FPGAI_DENSE_OUT_UNROLL; ++oo) {
#pragma HLS UNROLL
            int o = o0 + oo;
            if (o < OUT) y[o] = (ACT_T)acc[oo];
        }
    }
}

template<int IN, int OUT>
void dense_out_in(
    const act_t x[IN],
    act_t y[OUT],
    const wgt_t W[OUT * IN],
    const bias_t B[OUT]
) {
    dense_out_in_typed<IN, OUT, act_t, wgt_t, bias_t, acc_t>(x, y, W, B);
}

} // namespace fpgai
'''


def emit_dense_cpp() -> str:
    return '#include "layers/dense.h"\n'