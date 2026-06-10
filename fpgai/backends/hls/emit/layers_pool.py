from __future__ import annotations


def emit_pool_h() -> str:
    return r'''#pragma once

#include "fpgai_types.h"

#ifndef FPGAI_PIPELINE_II
#define FPGAI_PIPELINE_II 1
#endif

namespace fpgai {

template<
    int IN_H,
    int IN_W,
    int IN_C,
    int K,
    int STRIDE,
    int OUT_H,
    int OUT_W,
    typename IN_T = act_t,
    typename OUT_T = act_t
>
void maxpool2d_typed(
    const IN_T x[IN_H * IN_W * IN_C],
    OUT_T y[OUT_H * OUT_W * IN_C]
) {
#pragma HLS INLINE off

    for (int output_row = 0; output_row < OUT_H; ++output_row) {
        for (int output_column = 0; output_column < OUT_W; ++output_column) {
            for (int channel = 0; channel < IN_C; ++channel) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

                const int first_input_index = (
                    (
                        output_row * STRIDE * IN_W
                        + output_column * STRIDE
                    )
                    * IN_C
                    + channel
                );

                IN_T maximum = x[first_input_index];

                for (int kernel_row = 0; kernel_row < K; ++kernel_row) {
                    for (int kernel_column = 0; kernel_column < K; ++kernel_column) {
                        const int input_row = (
                            output_row * STRIDE
                            + kernel_row
                        );
                        const int input_column = (
                            output_column * STRIDE
                            + kernel_column
                        );
                        const int input_index = (
                            (
                                input_row * IN_W
                                + input_column
                            )
                            * IN_C
                            + channel
                        );

                        if (x[input_index] > maximum) {
                            maximum = x[input_index];
                        }
                    }
                }

                const int output_index = (
                    (
                        output_row * OUT_W
                        + output_column
                    )
                    * IN_C
                    + channel
                );

                y[output_index] = (OUT_T)maximum;
            }
        }
    }
}


template<
    int IN_H,
    int IN_W,
    int IN_C,
    int K,
    int STRIDE,
    int OUT_H,
    int OUT_W
>
void maxpool2d(
    const act_t* x,
    act_t* y
) {
    maxpool2d_typed<
        IN_H,
        IN_W,
        IN_C,
        K,
        STRIDE,
        OUT_H,
        OUT_W,
        act_t,
        act_t
    >(x, y);
}


template<
    int IN_H,
    int IN_W,
    int IN_C,
    int K,
    int STRIDE,
    int OUT_H,
    int OUT_W,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t
>
void avgpool2d_typed(
    const IN_T x[IN_H * IN_W * IN_C],
    OUT_T y[OUT_H * OUT_W * IN_C]
) {
#pragma HLS INLINE off

    for (int output_row = 0; output_row < OUT_H; ++output_row) {
        for (int output_column = 0; output_column < OUT_W; ++output_column) {
            for (int channel = 0; channel < IN_C; ++channel) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

                ACC_T sum = (ACC_T)0;

                for (int kernel_row = 0; kernel_row < K; ++kernel_row) {
                    for (int kernel_column = 0; kernel_column < K; ++kernel_column) {
                        const int input_row = (
                            output_row * STRIDE
                            + kernel_row
                        );
                        const int input_column = (
                            output_column * STRIDE
                            + kernel_column
                        );
                        const int input_index = (
                            (
                                input_row * IN_W
                                + input_column
                            )
                            * IN_C
                            + channel
                        );

                        sum += (ACC_T)x[input_index];
                    }
                }

                const int output_index = (
                    (
                        output_row * OUT_W
                        + output_column
                    )
                    * IN_C
                    + channel
                );

                y[output_index] = (
                    (OUT_T)(
                        sum / (ACC_T)(K * K)
                    )
                );
            }
        }
    }
}


template<
    int IN_H,
    int IN_W,
    int IN_C,
    int K,
    int STRIDE,
    int OUT_H,
    int OUT_W
>
void avgpool2d(
    const act_t* x,
    act_t* y
) {
    avgpool2d_typed<
        IN_H,
        IN_W,
        IN_C,
        K,
        STRIDE,
        OUT_H,
        OUT_W,
        act_t,
        act_t,
        acc_t
    >(x, y);
}


template<
    int IN_H,
    int IN_W,
    int IN_C,
    int K,
    int STRIDE,
    int OUT_H,
    int OUT_W,
    typename ACT_T = act_t,
    typename GRAD_OUT_T = grad_act_t,
    typename GRAD_IN_T = grad_act_t,
    typename ACC_T = acc_t
>
void maxpool2d_backward_typed(
    const ACT_T x[IN_H * IN_W * IN_C],
    const ACT_T y[OUT_H * OUT_W * IN_C],
    const GRAD_OUT_T dY[OUT_H * OUT_W * IN_C],
    GRAD_IN_T dX[IN_H * IN_W * IN_C]
) {
#pragma HLS INLINE off

    for (int index = 0; index < IN_H * IN_W * IN_C; ++index) {
#pragma HLS PIPELINE II=1
        dX[index] = (GRAD_IN_T)0;
    }

    for (int output_row = 0; output_row < OUT_H; ++output_row) {
        for (int output_column = 0; output_column < OUT_W; ++output_column) {
            for (int channel = 0; channel < IN_C; ++channel) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

                const int output_index = (
                    (
                        output_row * OUT_W
                        + output_column
                    )
                    * IN_C
                    + channel
                );

                const ACT_T pooled_value = y[output_index];
                bool routed = false;

                for (int kernel_row = 0; kernel_row < K; ++kernel_row) {
                    for (int kernel_column = 0; kernel_column < K; ++kernel_column) {
                        const int input_row = (
                            output_row * STRIDE
                            + kernel_row
                        );
                        const int input_column = (
                            output_column * STRIDE
                            + kernel_column
                        );
                        const int input_index = (
                            (
                                input_row * IN_W
                                + input_column
                            )
                            * IN_C
                            + channel
                        );

                        if (
                            !routed
                            && x[input_index] == pooled_value
                        ) {
                            dX[input_index] = (
                                (GRAD_IN_T)(
                                    (ACC_T)dX[input_index]
                                    + (ACC_T)dY[output_index]
                                )
                            );
                            routed = true;
                        }
                    }
                }
            }
        }
    }
}


template<
    int IN_H,
    int IN_W,
    int IN_C,
    int K,
    int STRIDE,
    int OUT_H,
    int OUT_W
>
void maxpool2d_backward(
    const act_t* x,
    const act_t* y,
    const grad_act_t* dY,
    grad_act_t* dX
) {
    maxpool2d_backward_typed<
        IN_H,
        IN_W,
        IN_C,
        K,
        STRIDE,
        OUT_H,
        OUT_W,
        act_t,
        grad_act_t,
        grad_act_t,
        acc_t
    >(x, y, dY, dX);
}


template<
    int IN_H,
    int IN_W,
    int IN_C,
    int K,
    int STRIDE,
    int OUT_H,
    int OUT_W,
    typename GRAD_OUT_T = grad_act_t,
    typename GRAD_IN_T = grad_act_t,
    typename ACC_T = acc_t
>
void avgpool2d_backward_typed(
    const GRAD_OUT_T dY[OUT_H * OUT_W * IN_C],
    GRAD_IN_T dX[IN_H * IN_W * IN_C]
) {
#pragma HLS INLINE off

    const ACC_T scale = (
        (ACC_T)1 / (ACC_T)(K * K)
    );

    for (int index = 0; index < IN_H * IN_W * IN_C; ++index) {
#pragma HLS PIPELINE II=1
        dX[index] = (GRAD_IN_T)0;
    }

    for (int output_row = 0; output_row < OUT_H; ++output_row) {
        for (int output_column = 0; output_column < OUT_W; ++output_column) {
            for (int channel = 0; channel < IN_C; ++channel) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

                const int output_index = (
                    (
                        output_row * OUT_W
                        + output_column
                    )
                    * IN_C
                    + channel
                );

                const ACC_T gradient = (
                    (ACC_T)dY[output_index]
                    * scale
                );

                for (int kernel_row = 0; kernel_row < K; ++kernel_row) {
                    for (int kernel_column = 0; kernel_column < K; ++kernel_column) {
                        const int input_row = (
                            output_row * STRIDE
                            + kernel_row
                        );
                        const int input_column = (
                            output_column * STRIDE
                            + kernel_column
                        );
                        const int input_index = (
                            (
                                input_row * IN_W
                                + input_column
                            )
                            * IN_C
                            + channel
                        );

                        dX[input_index] = (
                            (GRAD_IN_T)(
                                (ACC_T)dX[input_index]
                                + gradient
                            )
                        );
                    }
                }
            }
        }
    }
}


template<
    int IN_H,
    int IN_W,
    int IN_C,
    int K,
    int STRIDE,
    int OUT_H,
    int OUT_W
>
void avgpool2d_backward(
    const grad_act_t* dY,
    grad_act_t* dX
) {
    avgpool2d_backward_typed<
        IN_H,
        IN_W,
        IN_C,
        K,
        STRIDE,
        OUT_H,
        OUT_W,
        grad_act_t,
        grad_act_t,
        acc_t
    >(dY, dX);
}

} // namespace fpgai
'''


def emit_pool_cpp() -> str:
    return '#include "layers/pool.h"\n'