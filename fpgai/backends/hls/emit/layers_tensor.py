from __future__ import annotations


def emit_tensor_h() -> str:
    return r'''#pragma once
#include <hls_math.h>

namespace fpgai {

template<int OUTER, int LEFT_AXIS, int RIGHT_AXIS, int INNER, typename LEFT_T=act_t, typename RIGHT_T=act_t, typename OUT_T=act_t>
void concat_axis(const LEFT_T left[OUTER * LEFT_AXIS * INNER], const RIGHT_T right[OUTER * RIGHT_AXIS * INNER], OUT_T output[OUTER * (LEFT_AXIS + RIGHT_AXIS) * INNER]) {
#pragma HLS INLINE off
    for (int outer = 0; outer < OUTER; ++outer) {
        for (int a = 0; a < LEFT_AXIS; ++a) {
            for (int inner = 0; inner < INNER; ++inner) {
#pragma HLS PIPELINE II=1
                output[(outer * (LEFT_AXIS + RIGHT_AXIS) + a) * INNER + inner] = (OUT_T)left[(outer * LEFT_AXIS + a) * INNER + inner];
            }
        }
        for (int b = 0; b < RIGHT_AXIS; ++b) {
            for (int inner = 0; inner < INNER; ++inner) {
#pragma HLS PIPELINE II=1
                output[(outer * (LEFT_AXIS + RIGHT_AXIS) + LEFT_AXIS + b) * INNER + inner] = (OUT_T)right[(outer * RIGHT_AXIS + b) * INNER + inner];
            }
        }
    }
}

template<int OUTER, int LEFT_AXIS, int RIGHT_AXIS, int INNER, typename GRAD_T=grad_act_t>
void concat_axis_backward(const GRAD_T output_gradient[OUTER * (LEFT_AXIS + RIGHT_AXIS) * INNER], GRAD_T left_gradient[OUTER * LEFT_AXIS * INNER], GRAD_T right_gradient[OUTER * RIGHT_AXIS * INNER]) {
#pragma HLS INLINE off
    for (int outer = 0; outer < OUTER; ++outer) {
        for (int a = 0; a < LEFT_AXIS; ++a) {
            for (int inner = 0; inner < INNER; ++inner) {
#pragma HLS PIPELINE II=1
                left_gradient[(outer * LEFT_AXIS + a) * INNER + inner] += output_gradient[(outer * (LEFT_AXIS + RIGHT_AXIS) + a) * INNER + inner];
            }
        }
        for (int b = 0; b < RIGHT_AXIS; ++b) {
            for (int inner = 0; inner < INNER; ++inner) {
#pragma HLS PIPELINE II=1
                right_gradient[(outer * RIGHT_AXIS + b) * INNER + inner] += output_gradient[(outer * (LEFT_AXIS + RIGHT_AXIS) + LEFT_AXIS + b) * INNER + inner];
            }
        }
    }
}



template<int OUTER, int OUT_AXIS, int IN_AXIS, int OFFSET, int INNER, typename IN_T=act_t, typename OUT_T=act_t>
void concat_axis_segment(const IN_T input[OUTER * IN_AXIS * INNER], OUT_T output[OUTER * OUT_AXIS * INNER]) {
#pragma HLS INLINE off
    for (int outer = 0; outer < OUTER; ++outer) {
        for (int a = 0; a < IN_AXIS; ++a) {
            for (int inner = 0; inner < INNER; ++inner) {
#pragma HLS PIPELINE II=1
                output[(outer * OUT_AXIS + OFFSET + a) * INNER + inner] =
                    (OUT_T)input[(outer * IN_AXIS + a) * INNER + inner];
            }
        }
    }
}

template<int OUTER, int OUT_AXIS, int IN_AXIS, int OFFSET, int INNER, typename GRAD_T=grad_act_t>
void concat_axis_backward_segment(const GRAD_T output_gradient[OUTER * OUT_AXIS * INNER], GRAD_T input_gradient[OUTER * IN_AXIS * INNER]) {
#pragma HLS INLINE off
    for (int outer = 0; outer < OUTER; ++outer) {
        for (int a = 0; a < IN_AXIS; ++a) {
            for (int inner = 0; inner < INNER; ++inner) {
#pragma HLS PIPELINE II=1
                input_gradient[(outer * IN_AXIS + a) * INNER + inner] +=
                    output_gradient[(outer * OUT_AXIS + OFFSET + a) * INNER + inner];
            }
        }
    }
}

template<int IN_SIZE, int OUT_SIZE, int COORD_MODE, int NEAREST_MODE>
int resize_nearest_source_index(int out_index) {
#pragma HLS INLINE
    long long numerator = 0;
    long long denominator = 1;
    if (COORD_MODE == 0) { // asymmetric
        numerator = (long long)out_index * IN_SIZE;
        denominator = OUT_SIZE;
    } else if (COORD_MODE == 1) { // half_pixel
        numerator = ((long long)2 * out_index + 1) * IN_SIZE - OUT_SIZE;
        denominator = (long long)2 * OUT_SIZE;
    } else { // align_corners
        if (OUT_SIZE <= 1) return 0;
        numerator = (long long)out_index * (IN_SIZE - 1);
        denominator = OUT_SIZE - 1;
    }

    long long q = numerator / denominator;
    long long r = numerator % denominator;
    if (r < 0) {
        q -= 1;
        r += denominator;
    }

    long long index = q;
    if (NEAREST_MODE == 1) { // round_prefer_floor
        if ((2 * r) > denominator) index = q + 1;
    } else if (NEAREST_MODE == 2) { // ceil
        if (r != 0) index = q + 1;
    } else if (NEAREST_MODE == 3) { // round_prefer_ceil
        if ((2 * r) >= denominator) index = q + 1;
    }
    if (index < 0) index = 0;
    if (index >= IN_SIZE) index = IN_SIZE - 1;
    return (int)index;
}

template<int OUTER, int IN_AXIS, int START, int OUT_AXIS, int INNER, typename IN_T=act_t, typename OUT_T=act_t>
void slice_axis(const IN_T input[OUTER * IN_AXIS * INNER], OUT_T output[OUTER * OUT_AXIS * INNER]) {
#pragma HLS INLINE off
    for (int outer = 0; outer < OUTER; ++outer) {
        for (int a = 0; a < OUT_AXIS; ++a) {
            for (int inner = 0; inner < INNER; ++inner) {
#pragma HLS PIPELINE II=1
                output[(outer * OUT_AXIS + a) * INNER + inner] = (OUT_T)input[(outer * IN_AXIS + START + a) * INNER + inner];
            }
        }
    }
}

template<int OUTER, int IN_AXIS, int START, int OUT_AXIS, int INNER, typename GRAD_T=grad_act_t>
void slice_axis_backward(const GRAD_T output_gradient[OUTER * OUT_AXIS * INNER], GRAD_T input_gradient[OUTER * IN_AXIS * INNER]) {
#pragma HLS INLINE off
    for (int outer = 0; outer < OUTER; ++outer) {
        for (int a = 0; a < OUT_AXIS; ++a) {
            for (int inner = 0; inner < INNER; ++inner) {
#pragma HLS PIPELINE II=1
                input_gradient[(outer * IN_AXIS + START + a) * INNER + inner] += output_gradient[(outer * OUT_AXIS + a) * INNER + inner];
            }
        }
    }
}

template<int BATCH, int CHANNELS, int IN_H, int IN_W, int OUT_H, int OUT_W, int COORD_MODE=0, int NEAREST_MODE=0, typename IN_T=act_t, typename OUT_T=act_t>
void resize_nearest_nchw(const IN_T input[BATCH * CHANNELS * IN_H * IN_W], OUT_T output[BATCH * CHANNELS * OUT_H * OUT_W]) {
#pragma HLS INLINE off
    for (int b = 0; b < BATCH; ++b) {
        for (int c = 0; c < CHANNELS; ++c) {
            for (int oh = 0; oh < OUT_H; ++oh) {
                const int ih = resize_nearest_source_index<IN_H, OUT_H, COORD_MODE, NEAREST_MODE>(oh);
                for (int ow = 0; ow < OUT_W; ++ow) {
#pragma HLS PIPELINE II=1
                    const int iw = resize_nearest_source_index<IN_W, OUT_W, COORD_MODE, NEAREST_MODE>(ow);
                    output[((b * CHANNELS + c) * OUT_H + oh) * OUT_W + ow] = (OUT_T)input[((b * CHANNELS + c) * IN_H + ih) * IN_W + iw];
                }
            }
        }
    }
}

template<int BATCH, int CHANNELS, int IN_H, int IN_W, int OUT_H, int OUT_W, int COORD_MODE=0, int NEAREST_MODE=0, typename GRAD_T=grad_act_t>
void resize_nearest_nchw_backward(const GRAD_T output_gradient[BATCH * CHANNELS * OUT_H * OUT_W], GRAD_T input_gradient[BATCH * CHANNELS * IN_H * IN_W]) {
#pragma HLS INLINE off
    for (int b = 0; b < BATCH; ++b) {
        for (int c = 0; c < CHANNELS; ++c) {
            for (int oh = 0; oh < OUT_H; ++oh) {
                const int ih = resize_nearest_source_index<IN_H, OUT_H, COORD_MODE, NEAREST_MODE>(oh);
                for (int ow = 0; ow < OUT_W; ++ow) {
#pragma HLS PIPELINE II=1
                    const int iw = resize_nearest_source_index<IN_W, OUT_W, COORD_MODE, NEAREST_MODE>(ow);
                    input_gradient[((b * CHANNELS + c) * IN_H + ih) * IN_W + iw] += output_gradient[((b * CHANNELS + c) * OUT_H + oh) * OUT_W + ow];
                }
            }
        }
    }
}

template<int ROWS, int WIDTH, int INDEX_COUNT, typename DATA_T=act_t, typename INDEX_T=act_t, typename OUT_T=act_t>
void gather_rows(const DATA_T data[ROWS * WIDTH], const INDEX_T indices[INDEX_COUNT], OUT_T output[INDEX_COUNT * WIDTH]) {
#pragma HLS INLINE off
    for (int i = 0; i < INDEX_COUNT; ++i) {
        int row = (int)indices[i];
        if (row < 0) row += ROWS;
        if (row < 0) row = 0;
        if (row >= ROWS) row = ROWS - 1;
        for (int j = 0; j < WIDTH; ++j) {
#pragma HLS PIPELINE II=1
            output[i * WIDTH + j] = (OUT_T)data[row * WIDTH + j];
        }
    }
}

template<int ROWS, int WIDTH, int INDEX_COUNT, typename INDEX_T=act_t, typename GRAD_T=grad_act_t>
void gather_rows_backward(const INDEX_T indices[INDEX_COUNT], const GRAD_T output_gradient[INDEX_COUNT * WIDTH], GRAD_T data_gradient[ROWS * WIDTH]) {
#pragma HLS INLINE off
    for (int i = 0; i < INDEX_COUNT; ++i) {
        int row = (int)indices[i];
        if (row < 0) row += ROWS;
        if (row < 0) row = 0;
        if (row >= ROWS) row = ROWS - 1;
        for (int j = 0; j < WIDTH; ++j) {
#pragma HLS PIPELINE II=1
            data_gradient[row * WIDTH + j] += output_gradient[i * WIDTH + j];
        }
    }
}


template<int N, typename LEFT_T=act_t, typename RIGHT_T=act_t, typename OUT_T=act_t, typename ACC_T=acc_t>
void sub_vec_typed(const LEFT_T left[N], const RIGHT_T right[N], OUT_T output[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = (OUT_T)((ACC_T)left[i] - (ACC_T)right[i]);
    }
}

template<int N, typename LEFT_T=act_t, typename RIGHT_T=act_t, typename OUT_T=act_t, typename ACC_T=acc_t>
void div_vec_typed(const LEFT_T left[N], const RIGHT_T right[N], OUT_T output[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = (OUT_T)((ACC_T)left[i] / (ACC_T)right[i]);
    }
}

template<int N, typename IN_T=act_t, typename OUT_T=act_t, typename ACC_T=acc_t>
void div_scalar_typed(const IN_T input[N], ACC_T scalar, OUT_T output[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = (OUT_T)((ACC_T)input[i] / scalar);
    }
}

template<int N, typename IN_T=act_t, typename OUT_T=act_t, typename ACC_T=acc_t>
void sub_scalar_right_typed(const IN_T input[N], ACC_T scalar, OUT_T output[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = (OUT_T)((ACC_T)input[i] - scalar);
    }
}

template<int N, typename IN_T=act_t, typename OUT_T=act_t, typename ACC_T=acc_t>
void sqrt_vec_typed(const IN_T input[N], OUT_T output[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        ACC_T value = (ACC_T)input[i];
        output[i] = (OUT_T)hls::sqrt(value);
    }
}

template<int N, typename IN_T=act_t, typename OUT_T=act_t, typename ACC_T=acc_t>
void square_vec_typed(const IN_T input[N], OUT_T output[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        ACC_T value = (ACC_T)input[i];
        output[i] = (OUT_T)(value * value);
    }
}

template<int ROWS, int COLS, typename IN_T=act_t, typename OUT_T=act_t, typename ACC_T=acc_t>
void reduce_mean_last_axis(const IN_T input[ROWS * COLS], OUT_T output[ROWS]) {
#pragma HLS INLINE off
    for (int row = 0; row < ROWS; ++row) {
        ACC_T sum = (ACC_T)0;
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            sum += (ACC_T)input[row * COLS + col];
        }
        output[row] = (OUT_T)(sum / (ACC_T)COLS);
    }
}

template<int OUTER, int AXIS, int INNER, typename IN_T=act_t, typename OUT_T=act_t, typename ACC_T=acc_t>
void reduce_sum_axis_typed(const IN_T input[OUTER * AXIS * INNER], OUT_T output[OUTER * INNER]) {
#pragma HLS INLINE off
    for (int outer = 0; outer < OUTER; ++outer) {
        for (int inner = 0; inner < INNER; ++inner) {
            ACC_T sum = (ACC_T)0;
            for (int a = 0; a < AXIS; ++a) {
#pragma HLS PIPELINE II=1
                sum += (ACC_T)input[(outer * AXIS + a) * INNER + inner];
            }
            output[outer * INNER + inner] = (OUT_T)sum;
        }
    }
}

template<int OUTER, int AXIS, int INNER, typename OUT_GRAD_T=grad_act_t, typename IN_GRAD_T=grad_act_t>
void reduce_sum_axis_backward_typed(const OUT_GRAD_T output_gradient[OUTER * INNER], IN_GRAD_T input_gradient[OUTER * AXIS * INNER]) {
#pragma HLS INLINE off
    for (int outer = 0; outer < OUTER; ++outer) {
        for (int a = 0; a < AXIS; ++a) {
            for (int inner = 0; inner < INNER; ++inner) {
#pragma HLS PIPELINE II=1
                input_gradient[(outer * AXIS + a) * INNER + inner] += (IN_GRAD_T)output_gradient[outer * INNER + inner];
            }
        }
    }
}


template<int N, typename IN_T=act_t, typename OUT_T=act_t, typename ACC_T=acc_t>
void add_scalar_typed(const IN_T input[N], ACC_T scalar, OUT_T output[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = (OUT_T)((ACC_T)input[i] + scalar);
    }
}

template<int ROWS, int COLS, typename LEFT_T=act_t, typename RIGHT_T=act_t, typename OUT_T=act_t, typename ACC_T=acc_t>
void div_rows_by_scalar_vector(const LEFT_T left[ROWS * COLS], const RIGHT_T right[ROWS], OUT_T output[ROWS * COLS]) {
#pragma HLS INLINE off
    for (int row = 0; row < ROWS; ++row) {
        ACC_T denom = (ACC_T)right[row];
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            output[row * COLS + col] = (OUT_T)((ACC_T)left[row * COLS + col] / denom);
        }
    }
}

template<int ROWS, int COLS, typename IN_T=act_t, typename SCALE_T=act_t, typename OUT_T=act_t, typename ACC_T=acc_t>
void mul_rows_by_col_vector(const IN_T input[ROWS * COLS], const SCALE_T scale[COLS], OUT_T output[ROWS * COLS]) {
#pragma HLS INLINE off
    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            output[row * COLS + col] = (OUT_T)((ACC_T)input[row * COLS + col] * (ACC_T)scale[col]);
        }
    }
}


template<int N, typename IN_T=act_t, typename OUT_T=act_t>
void copy_vector(const IN_T input[N], OUT_T output[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = (OUT_T)input[i];
    }
}

template<int OUTER, int CAPACITY, int UPDATE_AXIS, int INNER, typename STATE_T=act_t, typename UPDATE_T=act_t>
void persistent_state_append_axis(
    STATE_T state[OUTER * CAPACITY * INNER],
    const UPDATE_T update[OUTER * UPDATE_AXIS * INNER],
    int &cursor
) {
#pragma HLS INLINE off
    int start = cursor;
    if (start < 0) start = 0;
    if (start > CAPACITY) start = CAPACITY;
    int writable = UPDATE_AXIS;
    if (start + writable > CAPACITY) writable = CAPACITY - start;
    for (int outer = 0; outer < OUTER; ++outer) {
        for (int step = 0; step < UPDATE_AXIS; ++step) {
            for (int inner = 0; inner < INNER; ++inner) {
#pragma HLS PIPELINE II=1
                if (step < writable) {
                    state[(outer * CAPACITY + start + step) * INNER + inner] =
                        (STATE_T)update[(outer * UPDATE_AXIS + step) * INNER + inner];
                }
            }
        }
    }
    cursor = start + writable;
}

template<int N, typename STATE_T=act_t, typename OUT_T=act_t>
void persistent_state_snapshot(const STATE_T state[N], OUT_T output[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = (OUT_T)state[i];
    }
}


template<int N, typename STATE_T=act_t, typename FLAG_T=ap_uint<1>>
void persistent_state_reset_if(STATE_T state[N], const FLAG_T reset_flag[1], int &cursor) {
#pragma HLS INLINE off
    if ((int)reset_flag[0] != 0) {
        for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            state[i] = (STATE_T)0;
        }
        cursor = 0;
    }
}

template<typename OUT_T=ap_int<32>>
void persistent_state_length(int cursor, OUT_T output[1]) {
#pragma HLS INLINE off
    output[0] = (OUT_T)cursor;
}

} // namespace fpgai
'''
