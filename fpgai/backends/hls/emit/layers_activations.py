from __future__ import annotations


def emit_activations_h() -> str:
    return r'''#pragma once

#include "fpgai_types.h"

#ifndef FPGAI_PIPELINE_II
#define FPGAI_PIPELINE_II 1
#endif

#ifndef FPGAI_ACT_UNROLL
#define FPGAI_ACT_UNROLL 1
#endif

namespace fpgai {

template<
    typename OUT_T,
    typename IN_T,
    typename ACC_T
>
static inline OUT_T sigmoid_approx_scalar_typed(
    IN_T input
) {
    const ACC_T value = (ACC_T)input;

    if (value <= (ACC_T)-4) {
        return (OUT_T)0;
    }

    if (value >= (ACC_T)4) {
        return (OUT_T)1;
    }

    return (OUT_T)(
        (ACC_T)0.5
        + value / (ACC_T)8
    );
}


static inline act_t sigmoid_approx_scalar(
    act_t input
) {
    return sigmoid_approx_scalar_typed<
        act_t,
        act_t,
        acc_t
    >(input);
}


template<
    typename OUT_T,
    typename IN_T,
    typename ACC_T
>
static inline OUT_T exp_approx_neg_scalar_typed(
    IN_T input
) {
    const ACC_T value = (ACC_T)input;

    if (value <= (ACC_T)-8) {
        return (OUT_T)0;
    }

    if (value >= (ACC_T)0) {
        return (OUT_T)1;
    }

    const ACC_T squared = value * value;

    ACC_T result = (
        (ACC_T)1
        + value
        + squared * (ACC_T)0.5
    );

    if (result < (ACC_T)0) {
        result = (ACC_T)0;
    }

    return (OUT_T)result;
}


static inline act_t exp_approx_neg_scalar(
    act_t input
) {
    return exp_approx_neg_scalar_typed<
        act_t,
        act_t,
        acc_t
    >(input);
}


template<
    int N,
    typename IN_T = act_t,
    typename OUT_T = act_t
>
void relu_typed(
    const IN_T x[N],
    OUT_T y[N]
) {
#pragma HLS INLINE off

    for (
        int base = 0;
        base < N;
        base += FPGAI_ACT_UNROLL
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        for (
            int lane = 0;
            lane < FPGAI_ACT_UNROLL;
            ++lane
        ) {
#pragma HLS UNROLL

            const int index = base + lane;

            if (index < N) {
                y[index] = (
                    x[index] > (IN_T)0
                    ? (OUT_T)x[index]
                    : (OUT_T)0
                );
            }
        }
    }
}


template<int N>
void relu(
    const act_t* x,
    act_t* y
) {
    relu_typed<
        N,
        act_t,
        act_t
    >(x, y);
}


template<
    int N,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t
>
void leaky_relu_typed(
    const IN_T x[N],
    OUT_T y[N],
    ACC_T alpha = (ACC_T)0.1
) {
#pragma HLS INLINE off

    for (
        int base = 0;
        base < N;
        base += FPGAI_ACT_UNROLL
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        for (
            int lane = 0;
            lane < FPGAI_ACT_UNROLL;
            ++lane
        ) {
#pragma HLS UNROLL

            const int index = base + lane;

            if (index < N) {
                const ACC_T value = (
                    (ACC_T)x[index]
                );

                y[index] = (
                    value > (ACC_T)0
                    ? (OUT_T)value
                    : (OUT_T)(alpha * value)
                );
            }
        }
    }
}


template<int N>
void leaky_relu(
    const act_t* x,
    act_t* y,
    act_t alpha = (act_t)0.1
) {
    leaky_relu_typed<
        N,
        act_t,
        act_t,
        acc_t
    >(
        x,
        y,
        (acc_t)alpha
    );
}


template<
    int N,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t
>
void sigmoid_typed(
    const IN_T x[N],
    OUT_T y[N]
) {
#pragma HLS INLINE off

    for (
        int base = 0;
        base < N;
        base += FPGAI_ACT_UNROLL
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        for (
            int lane = 0;
            lane < FPGAI_ACT_UNROLL;
            ++lane
        ) {
#pragma HLS UNROLL

            const int index = base + lane;

            if (index < N) {
                y[index] = (
                    sigmoid_approx_scalar_typed<
                        OUT_T,
                        IN_T,
                        ACC_T
                    >(x[index])
                );
            }
        }
    }
}


template<int N>
void sigmoid(
    const act_t* x,
    act_t* y
) {
    sigmoid_typed<
        N,
        act_t,
        act_t,
        acc_t
    >(x, y);
}


template<
    int N,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t
>
void softmax_typed(
    const IN_T x[N],
    OUT_T y[N]
) {
#pragma HLS INLINE off

    ACC_T maximum = (ACC_T)x[0];

    for (
        int index = 1;
        index < N;
        ++index
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        const ACC_T value = (
            (ACC_T)x[index]
        );

        if (value > maximum) {
            maximum = value;
        }
    }

    OUT_T temporary[N];

#pragma HLS ARRAY_PARTITION variable=temporary cyclic factor=FPGAI_ACT_UNROLL

    ACC_T sum = (ACC_T)0;

    for (
        int index = 0;
        index < N;
        ++index
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        const ACC_T shifted = (
            (ACC_T)x[index]
            - maximum
        );

        const OUT_T exponential = (
            exp_approx_neg_scalar_typed<
                OUT_T,
                ACC_T,
                ACC_T
            >(shifted)
        );

        temporary[index] = exponential;
        sum += (ACC_T)exponential;
    }

    if (sum <= (ACC_T)0) {
        sum = (ACC_T)1;
    }

    for (
        int index = 0;
        index < N;
        ++index
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        y[index] = (
            (OUT_T)(
                (ACC_T)temporary[index]
                / sum
            )
        );
    }
}


template<int N>
void softmax(
    const act_t* x,
    act_t* y
) {
    softmax_typed<
        N,
        act_t,
        act_t,
        acc_t
    >(x, y);
}


template<
    int N,
    typename IN_T = act_t,
    typename OUT_T = act_t
>
void reshape_copy_typed(
    const IN_T x[N],
    OUT_T y[N]
) {
#pragma HLS INLINE off

    for (
        int index = 0;
        index < N;
        ++index
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        y[index] = (OUT_T)x[index];
    }
}


template<int N>
void reshape_copy(
    const act_t* x,
    act_t* y
) {
    reshape_copy_typed<
        N,
        act_t,
        act_t
    >(x, y);
}


template<
    int N,
    typename LEFT_T = act_t,
    typename RIGHT_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t
>
void add_vec_typed(
    const LEFT_T left[N],
    const RIGHT_T right[N],
    OUT_T output[N]
) {
#pragma HLS INLINE off

    for (
        int index = 0;
        index < N;
        ++index
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        output[index] = (
            (OUT_T)(
                (ACC_T)left[index]
                + (ACC_T)right[index]
            )
        );
    }
}


template<int N>
void add_vec(
    const act_t* left,
    const act_t* right,
    act_t* output
) {
    add_vec_typed<
        N,
        act_t,
        act_t,
        act_t,
        acc_t
    >(
        left,
        right,
        output
    );
}


template<
    int N,
    typename ACT_T = act_t,
    typename GRAD_OUT_T = grad_act_t,
    typename GRAD_IN_T = grad_act_t
>
void relu_backward_from_output_typed(
    const ACT_T y[N],
    const GRAD_OUT_T dY[N],
    GRAD_IN_T dX[N]
) {
#pragma HLS INLINE off

    for (
        int base = 0;
        base < N;
        base += FPGAI_ACT_UNROLL
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        for (
            int lane = 0;
            lane < FPGAI_ACT_UNROLL;
            ++lane
        ) {
#pragma HLS UNROLL

            const int index = base + lane;

            if (index < N) {
                dX[index] = (
                    y[index] > (ACT_T)0
                    ? (GRAD_IN_T)dY[index]
                    : (GRAD_IN_T)0
                );
            }
        }
    }
}


template<int N>
void relu_backward_from_output(
    const act_t* y,
    const grad_act_t* dY,
    grad_act_t* dX
) {
    relu_backward_from_output_typed<
        N,
        act_t,
        grad_act_t,
        grad_act_t
    >(
        y,
        dY,
        dX
    );
}


template<
    int N,
    typename ACT_T = act_t,
    typename GRAD_OUT_T = grad_act_t,
    typename GRAD_IN_T = grad_act_t,
    typename ACC_T = acc_t
>
void leaky_relu_backward_from_input_typed(
    const ACT_T x[N],
    const GRAD_OUT_T dY[N],
    GRAD_IN_T dX[N],
    ACC_T alpha = (ACC_T)0.1
) {
#pragma HLS INLINE off

    for (
        int base = 0;
        base < N;
        base += FPGAI_ACT_UNROLL
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        for (
            int lane = 0;
            lane < FPGAI_ACT_UNROLL;
            ++lane
        ) {
#pragma HLS UNROLL

            const int index = base + lane;

            if (index < N) {
                dX[index] = (
                    x[index] > (ACT_T)0
                    ? (GRAD_IN_T)dY[index]
                    : (
                        (GRAD_IN_T)(
                            alpha
                            * (ACC_T)dY[index]
                        )
                    )
                );
            }
        }
    }
}


template<int N>
void leaky_relu_backward_from_input(
    const act_t* x,
    const grad_act_t* dY,
    grad_act_t* dX,
    act_t alpha = (act_t)0.1
) {
    leaky_relu_backward_from_input_typed<
        N,
        act_t,
        grad_act_t,
        grad_act_t,
        acc_t
    >(
        x,
        dY,
        dX,
        (acc_t)alpha
    );
}


template<
    int N,
    typename ACT_T = act_t,
    typename GRAD_OUT_T = grad_act_t,
    typename GRAD_IN_T = grad_act_t,
    typename ACC_T = acc_t
>
void sigmoid_backward_from_output_typed(
    const ACT_T y[N],
    const GRAD_OUT_T dY[N],
    GRAD_IN_T dX[N]
) {
#pragma HLS INLINE off

    for (
        int base = 0;
        base < N;
        base += FPGAI_ACT_UNROLL
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        for (
            int lane = 0;
            lane < FPGAI_ACT_UNROLL;
            ++lane
        ) {
#pragma HLS UNROLL

            const int index = base + lane;

            if (index < N) {
                const ACC_T output = (
                    (ACC_T)y[index]
                );

                dX[index] = (
                    (GRAD_IN_T)(
                        (ACC_T)dY[index]
                        * output
                        * (
                            (ACC_T)1
                            - output
                        )
                    )
                );
            }
        }
    }
}


template<int N>
void sigmoid_backward_from_output(
    const act_t* y,
    const grad_act_t* dY,
    grad_act_t* dX
) {
    sigmoid_backward_from_output_typed<
        N,
        act_t,
        grad_act_t,
        grad_act_t,
        acc_t
    >(
        y,
        dY,
        dX
    );
}


template<
    int N,
    typename ACT_T = act_t,
    typename GRAD_OUT_T = grad_act_t,
    typename GRAD_IN_T = grad_act_t,
    typename ACC_T = acc_t
>
void softmax_backward_typed(
    const ACT_T y[N],
    const GRAD_OUT_T dY[N],
    GRAD_IN_T dX[N]
) {
#pragma HLS INLINE off

    for (
        int output_index = 0;
        output_index < N;
        ++output_index
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        ACC_T accumulator = (ACC_T)0;

        for (
            int input_index = 0;
            input_index < N;
            ++input_index
        ) {
            const ACC_T jacobian = (
                output_index == input_index
                ? (
                    (ACC_T)y[output_index]
                    * (
                        (ACC_T)1
                        - (ACC_T)y[output_index]
                    )
                )
                : (
                    -(ACC_T)y[output_index]
                    * (ACC_T)y[input_index]
                )
            );

            accumulator += (
                jacobian
                * (ACC_T)dY[input_index]
            );
        }

        dX[output_index] = (
            (GRAD_IN_T)accumulator
        );
    }
}


template<int N>
void softmax_backward(
    const act_t* y,
    const grad_act_t* dY,
    grad_act_t* dX
) {
    softmax_backward_typed<
        N,
        act_t,
        grad_act_t,
        grad_act_t,
        acc_t
    >(
        y,
        dY,
        dX
    );
}


template<
    int N,
    typename GRAD_OUT_T = grad_act_t,
    typename GRAD_LEFT_T = grad_act_t,
    typename GRAD_RIGHT_T = grad_act_t,
    typename ACC_T = acc_t
>
void add_backward_typed(
    const GRAD_OUT_T dY[N],
    GRAD_LEFT_T dLeft[N],
    GRAD_RIGHT_T dRight[N]
) {
#pragma HLS INLINE off

    for (
        int index = 0;
        index < N;
        ++index
    ) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II

        dLeft[index] = (
            (GRAD_LEFT_T)(
                (ACC_T)dLeft[index]
                + (ACC_T)dY[index]
            )
        );

        dRight[index] = (
            (GRAD_RIGHT_T)(
                (ACC_T)dRight[index]
                + (ACC_T)dY[index]
            )
        );
    }
}


template<int N>
void add_backward(
    const grad_act_t* dY,
    grad_act_t* dLeft,
    grad_act_t* dRight
) {
    add_backward_typed<
        N,
        grad_act_t,
        grad_act_t,
        grad_act_t,
        acc_t
    >(
        dY,
        dLeft,
        dRight
    );
}

} // namespace fpgai
'''


def emit_activations_cpp() -> str:
    return '#include "layers/activations.h"\n'