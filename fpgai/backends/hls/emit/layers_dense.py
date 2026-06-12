from __future__ import annotations


def emit_dense_h() -> str:
    return r'''#pragma once

#include "fpgai_types.h"

#ifndef FPGAI_PIPELINE_II
#define FPGAI_PIPELINE_II 1
#endif

#ifndef FPGAI_DENSE_IN_UNROLL
#define FPGAI_DENSE_IN_UNROLL 1
#endif

#ifndef FPGAI_DENSE_OUT_UNROLL
#define FPGAI_DENSE_OUT_UNROLL 1
#endif

namespace fpgai {

template<
    int IN_F,
    int OUT_F,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename WGT_T = wgt_t,
    typename BIAS_T = bias_t,
    typename ACC_T = acc_t,
    int PIPELINE_II = FPGAI_PIPELINE_II,
    int IN_UNROLL = FPGAI_DENSE_IN_UNROLL,
    int OUT_UNROLL = FPGAI_DENSE_OUT_UNROLL,
    int INPUT_PARTITION = 1,
    int OUTPUT_PARTITION = 1,
    int WEIGHT_PARTITION = 1
>
void dense_out_in(
    const IN_T x[IN_F],
    OUT_T y[OUT_F],
    const WGT_T W[OUT_F * IN_F],
    const BIAS_T B[OUT_F]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1

    for (
        int output_base = 0;
        output_base < OUT_F;
        output_base += OUT_UNROLL
    ) {
        ACC_T accumulators[OUT_UNROLL];

#pragma HLS ARRAY_PARTITION variable=accumulators complete

        for (
            int output_lane = 0;
            output_lane < OUT_UNROLL;
            ++output_lane
        ) {
#pragma HLS UNROLL

            const int output_index = (
                output_base + output_lane
            );

            if (output_index < OUT_F) {
                accumulators[output_lane] = (
                    (ACC_T)B[output_index]
                );
            } else {
                accumulators[output_lane] = (
                    (ACC_T)0
                );
            }
        }

        for (
            int input_base = 0;
            input_base < IN_F;
            input_base += IN_UNROLL
        ) {
#pragma HLS PIPELINE II=PIPELINE_II

            for (
                int input_lane = 0;
                input_lane < IN_UNROLL;
                ++input_lane
            ) {
#pragma HLS UNROLL

                const int input_index = (
                    input_base + input_lane
                );

                if (input_index >= IN_F) {
                    continue;
                }

                const ACC_T input_value = (
                    (ACC_T)x[input_index]
                );

                for (
                    int output_lane = 0;
                    output_lane < OUT_UNROLL;
                    ++output_lane
                ) {
#pragma HLS UNROLL

                    const int output_index = (
                        output_base + output_lane
                    );

                    if (output_index >= OUT_F) {
                        continue;
                    }

                    const int weight_index = (
                        output_index * IN_F
                        + input_index
                    );

                    accumulators[output_lane] += (
                        input_value
                        * (ACC_T)W[weight_index]
                    );
                }
            }
        }

        for (
            int output_lane = 0;
            output_lane < OUT_UNROLL;
            ++output_lane
        ) {
#pragma HLS UNROLL

            const int output_index = (
                output_base + output_lane
            );

            if (output_index < OUT_F) {
                y[output_index] = (
                    (OUT_T)accumulators[output_lane]
                );
            }
        }
    }
}


template<
    int IN_F,
    int OUT_F,
    typename ACT_T = act_t,
    typename GRAD_ACT_T = grad_act_t,
    typename GRAD_WGT_T = grad_wgt_t,
    typename ACC_T = acc_t
>
void dense_weight_grad_typed(
    const ACT_T x[IN_F],
    const GRAD_ACT_T dY[OUT_F],
    GRAD_WGT_T dW[OUT_F * IN_F]
) {
#pragma HLS INLINE off

    for (
        int output_index = 0;
        output_index < OUT_F;
        ++output_index
    ) {
        for (
            int input_index = 0;
            input_index < IN_F;
            ++input_index
        ) {
            const int weight_index = (
                output_index * IN_F
                + input_index
            );

            dW[weight_index] = (
                (GRAD_WGT_T)(
                    (ACC_T)dY[output_index]
                    * (ACC_T)x[input_index]
                )
            );
        }
    }
}


template<int IN_F, int OUT_F>
void dense_weight_grad(
    const act_t x[IN_F],
    const grad_act_t dY[OUT_F],
    grad_wgt_t dW[OUT_F * IN_F]
) {
    dense_weight_grad_typed<
        IN_F,
        OUT_F,
        act_t,
        grad_act_t,
        grad_wgt_t,
        acc_t
    >(
        x,
        dY,
        dW
    );
}


template<
    int OUT_F,
    typename GRAD_ACT_T = grad_act_t,
    typename GRAD_BIAS_T = grad_bias_t
>
void dense_bias_grad_typed(
    const GRAD_ACT_T dY[OUT_F],
    GRAD_BIAS_T dB[OUT_F]
) {
#pragma HLS INLINE off

    for (
        int output_index = 0;
        output_index < OUT_F;
        ++output_index
    ) {
#pragma HLS PIPELINE II=1

        dB[output_index] = (
            (GRAD_BIAS_T)dY[output_index]
        );
    }
}


template<int OUT_F>
void dense_bias_grad(
    const grad_act_t dY[OUT_F],
    grad_bias_t dB[OUT_F]
) {
    dense_bias_grad_typed<
        OUT_F,
        grad_act_t,
        grad_bias_t
    >(
        dY,
        dB
    );
}


template<
    int IN_F,
    int OUT_F,
    typename GRAD_OUT_T = grad_act_t,
    typename WGT_T = wgt_t,
    typename GRAD_IN_T = grad_act_t,
    typename ACC_T = acc_t
>
void dense_backward_input_typed(
    const GRAD_OUT_T dY[OUT_F],
    const WGT_T W[OUT_F * IN_F],
    GRAD_IN_T dX[IN_F]
) {
#pragma HLS INLINE off

    for (
        int input_index = 0;
        input_index < IN_F;
        ++input_index
    ) {
        ACC_T accumulator = (ACC_T)0;

        for (
            int output_index = 0;
            output_index < OUT_F;
            ++output_index
        ) {
            const int weight_index = (
                output_index * IN_F
                + input_index
            );

            accumulator += (
                (ACC_T)dY[output_index]
                * (ACC_T)W[weight_index]
            );
        }

        dX[input_index] = (
            (GRAD_IN_T)accumulator
        );
    }
}


template<int IN_F, int OUT_F>
void dense_backward_input(
    const grad_act_t dY[OUT_F],
    const wgt_t W[OUT_F * IN_F],
    grad_act_t dX[IN_F]
) {
    dense_backward_input_typed<
        IN_F,
        OUT_F,
        grad_act_t,
        wgt_t,
        grad_act_t,
        acc_t
    >(
        dY,
        W,
        dX
    );
}


template<
    int N,
    typename WGT_T = wgt_t,
    typename GRAD_WGT_T = grad_wgt_t,
    typename UPDATE_T = upd_t,
    typename ACC_T = acc_t
>
void sgd_update_wgt_typed(
    WGT_T W[N],
    const GRAD_WGT_T dW[N],
    const UPDATE_T learning_rate
) {
#pragma HLS INLINE off

    for (
        int index = 0;
        index < N;
        ++index
    ) {
#pragma HLS PIPELINE II=1

        const ACC_T updated = (
            (ACC_T)W[index]
            - (
                (ACC_T)learning_rate
                * (ACC_T)dW[index]
            )
        );

        W[index] = (WGT_T)updated;
    }
}


template<int N>
void sgd_update_wgt(
    wgt_t W[N],
    const grad_wgt_t dW[N],
    const upd_t learning_rate
) {
    sgd_update_wgt_typed<
        N,
        wgt_t,
        grad_wgt_t,
        upd_t,
        acc_t
    >(
        W,
        dW,
        learning_rate
    );
}


template<
    int N,
    typename BIAS_T = bias_t,
    typename GRAD_BIAS_T = grad_bias_t,
    typename UPDATE_T = upd_t,
    typename ACC_T = acc_t
>
void sgd_update_bias_typed(
    BIAS_T B[N],
    const GRAD_BIAS_T dB[N],
    const UPDATE_T learning_rate
) {
#pragma HLS INLINE off

    for (
        int index = 0;
        index < N;
        ++index
    ) {
#pragma HLS PIPELINE II=1

        const ACC_T updated = (
            (ACC_T)B[index]
            - (
                (ACC_T)learning_rate
                * (ACC_T)dB[index]
            )
        );

        B[index] = (BIAS_T)updated;
    }
}


template<int N>
void sgd_update_bias(
    bias_t B[N],
    const grad_bias_t dB[N],
    const upd_t learning_rate
) {
    sgd_update_bias_typed<
        N,
        bias_t,
        grad_bias_t,
        upd_t,
        acc_t
    >(
        B,
        dB,
        learning_rate
    );
}

} // namespace fpgai
'''


def emit_dense_cpp() -> str:
    return '#include "layers/dense.h"\n'
