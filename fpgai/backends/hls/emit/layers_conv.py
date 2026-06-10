from __future__ import annotations


def emit_conv_h() -> str:
    return r'''#pragma once

#include "fpgai_types.h"

#ifndef FPGAI_PIPELINE_II
#define FPGAI_PIPELINE_II 1
#endif

#ifndef FPGAI_CONV_OC_UNROLL
#define FPGAI_CONV_OC_UNROLL 1
#endif

#ifndef FPGAI_CONV_IC_UNROLL
#define FPGAI_CONV_IC_UNROLL 1
#endif

namespace fpgai {

template<
    int IN_H,
    int IN_W,
    int IN_C,
    int OUT_H,
    int OUT_W,
    int OUT_C,
    int K,
    int STRIDE,
    int PAD,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename WGT_T = wgt_t,
    typename BIAS_T = bias_t,
    typename ACC_T = acc_t
>
void conv2d(
    const IN_T x[IN_H * IN_W * IN_C],
    OUT_T y[OUT_H * OUT_W * OUT_C],
    const WGT_T W[OUT_C * IN_C * K * K],
    const BIAS_T B[OUT_C]
) {
#pragma HLS INLINE off

    for (
        int output_channel_base = 0;
        output_channel_base < OUT_C;
        output_channel_base += FPGAI_CONV_OC_UNROLL
    ) {
        for (
            int output_row = 0;
            output_row < OUT_H;
            ++output_row
        ) {
            for (
                int output_column = 0;
                output_column < OUT_W;
                ++output_column
            ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

                ACC_T accumulators[
                    FPGAI_CONV_OC_UNROLL
                ];

#pragma HLS ARRAY_PARTITION variable=accumulators complete

                for (
                    int output_lane = 0;
                    output_lane < FPGAI_CONV_OC_UNROLL;
                    ++output_lane
                ) {
#pragma HLS UNROLL

                    const int output_channel = (
                        output_channel_base
                        + output_lane
                    );

                    if (output_channel < OUT_C) {
                        accumulators[output_lane] = (
                            (ACC_T)B[output_channel]
                        );
                    } else {
                        accumulators[output_lane] = (
                            (ACC_T)0
                        );
                    }
                }

                for (
                    int input_channel_base = 0;
                    input_channel_base < IN_C;
                    input_channel_base += FPGAI_CONV_IC_UNROLL
                ) {
                    for (
                        int kernel_row = 0;
                        kernel_row < K;
                        ++kernel_row
                    ) {
                        for (
                            int kernel_column = 0;
                            kernel_column < K;
                            ++kernel_column
                        ) {
                            for (
                                int input_lane = 0;
                                input_lane < FPGAI_CONV_IC_UNROLL;
                                ++input_lane
                            ) {
#pragma HLS UNROLL

                                const int input_channel = (
                                    input_channel_base
                                    + input_lane
                                );

                                if (input_channel >= IN_C) {
                                    continue;
                                }

                                const int input_row = (
                                    output_row * STRIDE
                                    + kernel_row
                                    - PAD
                                );
                                const int input_column = (
                                    output_column * STRIDE
                                    + kernel_column
                                    - PAD
                                );

                                if (
                                    input_row < 0
                                    || input_row >= IN_H
                                    || input_column < 0
                                    || input_column >= IN_W
                                ) {
                                    continue;
                                }

                                const int input_index = (
                                    (
                                        input_row * IN_W
                                        + input_column
                                    )
                                    * IN_C
                                    + input_channel
                                );

                                const ACC_T input_value = (
                                    (ACC_T)x[input_index]
                                );

                                for (
                                    int output_lane = 0;
                                    output_lane < FPGAI_CONV_OC_UNROLL;
                                    ++output_lane
                                ) {
#pragma HLS UNROLL

                                    const int output_channel = (
                                        output_channel_base
                                        + output_lane
                                    );

                                    if (
                                        output_channel >= OUT_C
                                    ) {
                                        continue;
                                    }

                                    const int weight_index = (
                                        (
                                            (
                                                output_channel
                                                * IN_C
                                                + input_channel
                                            )
                                            * K
                                            + kernel_row
                                        )
                                        * K
                                        + kernel_column
                                    );

                                    accumulators[output_lane] += (
                                        input_value
                                        * (ACC_T)W[weight_index]
                                    );
                                }
                            }
                        }
                    }
                }

                for (
                    int output_lane = 0;
                    output_lane < FPGAI_CONV_OC_UNROLL;
                    ++output_lane
                ) {
#pragma HLS UNROLL

                    const int output_channel = (
                        output_channel_base
                        + output_lane
                    );

                    if (output_channel >= OUT_C) {
                        continue;
                    }

                    const int output_index = (
                        (
                            output_row * OUT_W
                            + output_column
                        )
                        * OUT_C
                        + output_channel
                    );

                    y[output_index] = (
                        (OUT_T)accumulators[output_lane]
                    );
                }
            }
        }
    }
}


template<
    int IN_H,
    int IN_W,
    int IN_C,
    int OUT_H,
    int OUT_W,
    int OUT_C,
    int K,
    int STRIDE,
    int PAD,
    typename GRAD_OUT_T = grad_act_t,
    typename WGT_T = wgt_t,
    typename GRAD_IN_T = grad_act_t,
    typename ACC_T = acc_t
>
void conv2d_backward_input_typed(
    const GRAD_OUT_T dY[
        OUT_H * OUT_W * OUT_C
    ],
    const WGT_T W[
        OUT_C * IN_C * K * K
    ],
    GRAD_IN_T dX[
        IN_H * IN_W * IN_C
    ]
) {
#pragma HLS INLINE off

    for (
        int index = 0;
        index < IN_H * IN_W * IN_C;
        ++index
    ) {
#pragma HLS PIPELINE II=1

        dX[index] = (GRAD_IN_T)0;
    }

    for (
        int output_row = 0;
        output_row < OUT_H;
        ++output_row
    ) {
        for (
            int output_column = 0;
            output_column < OUT_W;
            ++output_column
        ) {
            for (
                int output_channel = 0;
                output_channel < OUT_C;
                ++output_channel
            ) {
                const int output_index = (
                    (
                        output_row * OUT_W
                        + output_column
                    )
                    * OUT_C
                    + output_channel
                );

                const ACC_T output_gradient = (
                    (ACC_T)dY[output_index]
                );

                for (
                    int input_channel = 0;
                    input_channel < IN_C;
                    ++input_channel
                ) {
                    for (
                        int kernel_row = 0;
                        kernel_row < K;
                        ++kernel_row
                    ) {
                        for (
                            int kernel_column = 0;
                            kernel_column < K;
                            ++kernel_column
                        ) {
                            const int input_row = (
                                output_row * STRIDE
                                + kernel_row
                                - PAD
                            );
                            const int input_column = (
                                output_column * STRIDE
                                + kernel_column
                                - PAD
                            );

                            if (
                                input_row < 0
                                || input_row >= IN_H
                                || input_column < 0
                                || input_column >= IN_W
                            ) {
                                continue;
                            }

                            const int input_index = (
                                (
                                    input_row * IN_W
                                    + input_column
                                )
                                * IN_C
                                + input_channel
                            );

                            const int weight_index = (
                                (
                                    (
                                        output_channel
                                        * IN_C
                                        + input_channel
                                    )
                                    * K
                                    + kernel_row
                                )
                                * K
                                + kernel_column
                            );

                            const ACC_T updated = (
                                (ACC_T)dX[input_index]
                                + (
                                    output_gradient
                                    * (ACC_T)W[weight_index]
                                )
                            );

                            dX[input_index] = (
                                (GRAD_IN_T)updated
                            );
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
    int OUT_H,
    int OUT_W,
    int OUT_C,
    int K,
    int STRIDE,
    int PAD
>
void conv2d_backward_input(
    const grad_act_t dY[
        OUT_H * OUT_W * OUT_C
    ],
    const wgt_t W[
        OUT_C * IN_C * K * K
    ],
    grad_act_t dX[
        IN_H * IN_W * IN_C
    ]
) {
    conv2d_backward_input_typed<
        IN_H,
        IN_W,
        IN_C,
        OUT_H,
        OUT_W,
        OUT_C,
        K,
        STRIDE,
        PAD,
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
    int IN_H,
    int IN_W,
    int IN_C,
    int OUT_H,
    int OUT_W,
    int OUT_C,
    int K,
    int STRIDE,
    int PAD,
    typename ACT_T = act_t,
    typename GRAD_OUT_T = grad_act_t,
    typename GRAD_WGT_T = grad_wgt_t,
    typename ACC_T = acc_t
>
void conv2d_weight_grad_typed(
    const ACT_T x[
        IN_H * IN_W * IN_C
    ],
    const GRAD_OUT_T dY[
        OUT_H * OUT_W * OUT_C
    ],
    GRAD_WGT_T dW[
        OUT_C * IN_C * K * K
    ]
) {
#pragma HLS INLINE off

    for (
        int output_channel = 0;
        output_channel < OUT_C;
        ++output_channel
    ) {
        for (
            int input_channel = 0;
            input_channel < IN_C;
            ++input_channel
        ) {
            for (
                int kernel_row = 0;
                kernel_row < K;
                ++kernel_row
            ) {
                for (
                    int kernel_column = 0;
                    kernel_column < K;
                    ++kernel_column
                ) {
                    ACC_T accumulator = (ACC_T)0;

                    for (
                        int output_row = 0;
                        output_row < OUT_H;
                        ++output_row
                    ) {
                        for (
                            int output_column = 0;
                            output_column < OUT_W;
                            ++output_column
                        ) {
                            const int input_row = (
                                output_row * STRIDE
                                + kernel_row
                                - PAD
                            );
                            const int input_column = (
                                output_column * STRIDE
                                + kernel_column
                                - PAD
                            );

                            if (
                                input_row < 0
                                || input_row >= IN_H
                                || input_column < 0
                                || input_column >= IN_W
                            ) {
                                continue;
                            }

                            const int input_index = (
                                (
                                    input_row * IN_W
                                    + input_column
                                )
                                * IN_C
                                + input_channel
                            );

                            const int output_index = (
                                (
                                    output_row * OUT_W
                                    + output_column
                                )
                                * OUT_C
                                + output_channel
                            );

                            accumulator += (
                                (ACC_T)x[input_index]
                                * (ACC_T)dY[output_index]
                            );
                        }
                    }

                    const int weight_index = (
                        (
                            (
                                output_channel
                                * IN_C
                                + input_channel
                            )
                            * K
                            + kernel_row
                        )
                        * K
                        + kernel_column
                    );

                    dW[weight_index] = (
                        (GRAD_WGT_T)accumulator
                    );
                }
            }
        }
    }
}


template<
    int IN_H,
    int IN_W,
    int IN_C,
    int OUT_H,
    int OUT_W,
    int OUT_C,
    int K,
    int STRIDE,
    int PAD
>
void conv2d_weight_grad(
    const act_t x[
        IN_H * IN_W * IN_C
    ],
    const grad_act_t dY[
        OUT_H * OUT_W * OUT_C
    ],
    grad_wgt_t dW[
        OUT_C * IN_C * K * K
    ]
) {
    conv2d_weight_grad_typed<
        IN_H,
        IN_W,
        IN_C,
        OUT_H,
        OUT_W,
        OUT_C,
        K,
        STRIDE,
        PAD,
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
    int OUT_C,
    int OUT_H,
    int OUT_W,
    typename GRAD_OUT_T = grad_act_t,
    typename GRAD_BIAS_T = grad_bias_t,
    typename ACC_T = acc_t
>
void conv2d_bias_grad_typed(
    const GRAD_OUT_T dY[
        OUT_H * OUT_W * OUT_C
    ],
    GRAD_BIAS_T dB[OUT_C]
) {
#pragma HLS INLINE off

    for (
        int output_channel = 0;
        output_channel < OUT_C;
        ++output_channel
    ) {
        ACC_T accumulator = (ACC_T)0;

        for (
            int output_row = 0;
            output_row < OUT_H;
            ++output_row
        ) {
            for (
                int output_column = 0;
                output_column < OUT_W;
                ++output_column
            ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

                const int output_index = (
                    (
                        output_row * OUT_W
                        + output_column
                    )
                    * OUT_C
                    + output_channel
                );

                accumulator += (
                    (ACC_T)dY[output_index]
                );
            }
        }

        dB[output_channel] = (
            (GRAD_BIAS_T)accumulator
        );
    }
}


template<
    int OUT_C,
    int OUT_H,
    int OUT_W
>
void conv2d_bias_grad(
    const grad_act_t dY[
        OUT_H * OUT_W * OUT_C
    ],
    grad_bias_t dB[OUT_C]
) {
    conv2d_bias_grad_typed<
        OUT_C,
        OUT_H,
        OUT_W,
        grad_act_t,
        grad_bias_t,
        acc_t
    >(
        dY,
        dB
    );
}

} // namespace fpgai
'''


def emit_conv_cpp() -> str:
    return '#include "layers/conv.h"\n'