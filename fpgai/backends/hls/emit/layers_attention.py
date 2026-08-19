from __future__ import annotations


def emit_attention_h() -> str:
    return r'''#pragma once

#include "fpgai_types.h"
#include <hls_math.h>

namespace fpgai {

template<
    int M,
    int K,
    int N,
    typename LEFT_T = act_t,
    typename RIGHT_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t,
    int TILE_M = 1,
    int TILE_N = 1,
    int TILE_K = 1,
    int PIPELINE_II = 1,
    int M_UNROLL = 1,
    int N_UNROLL = 1,
    int K_UNROLL = 1,
    int INPUT_PARTITION = 1,
    int OUTPUT_PARTITION = 1,
    int WEIGHT_PARTITION = 1
>
void matmul_tiled(
    const LEFT_T left[M * K],
    const RIGHT_T right[K * N],
    OUT_T output[M * N]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=left cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=right cyclic factor=WEIGHT_PARTITION
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_PARTITION
    for (int m0 = 0; m0 < M; m0 += TILE_M) {
        for (int n0 = 0; n0 < N; n0 += TILE_N) {
            ACC_T acc[TILE_M][TILE_N];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
            for (int mi = 0; mi < TILE_M; ++mi) {
#pragma HLS UNROLL factor=M_UNROLL
                for (int ni = 0; ni < TILE_N; ++ni) {
#pragma HLS UNROLL factor=N_UNROLL
                    acc[mi][ni] = (ACC_T)0;
                }
            }
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
#pragma HLS PIPELINE II=PIPELINE_II
                for (int mi = 0; mi < TILE_M; ++mi) {
#pragma HLS UNROLL factor=M_UNROLL
                    for (int ni = 0; ni < TILE_N; ++ni) {
#pragma HLS UNROLL factor=N_UNROLL
                        const int m = m0 + mi;
                        const int n = n0 + ni;
                        for (int ki = 0; ki < TILE_K; ++ki) {
#pragma HLS UNROLL factor=K_UNROLL
                            const int k = k0 + ki;
                            if (m < M && n < N && k < K) {
                                acc[mi][ni] += (ACC_T)left[m * K + k] * (ACC_T)right[k * N + n];
                            }
                        }
                    }
                }
            }
            for (int mi = 0; mi < TILE_M; ++mi) {
#pragma HLS UNROLL
                for (int ni = 0; ni < TILE_N; ++ni) {
#pragma HLS UNROLL
                    const int m = m0 + mi;
                    const int n = n0 + ni;
                    if (m < M && n < N) {
                        output[m * N + n] = (OUT_T)acc[mi][ni];
                    }
                }
            }
        }
    }
}


template<
    int M,
    int K,
    int N,
    typename LEFT_T = act_t,
    typename RIGHT_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t,
    int TILE_M = 1,
    int TILE_N = 1,
    int TILE_K = 1
>
void matmul_tiled_right_transposed(
    const LEFT_T left[M * K],
    const RIGHT_T right_transposed[N * K],
    OUT_T output[M * N]
) {
#pragma HLS INLINE off
    for (int m0 = 0; m0 < M; m0 += TILE_M) {
        for (int n0 = 0; n0 < N; n0 += TILE_N) {
            ACC_T acc[TILE_M][TILE_N];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
            for (int mi = 0; mi < TILE_M; ++mi) {
                for (int ni = 0; ni < TILE_N; ++ni) acc[mi][ni] = (ACC_T)0;
            }
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
#pragma HLS PIPELINE II=1
                for (int mi = 0; mi < TILE_M; ++mi) {
                    for (int ni = 0; ni < TILE_N; ++ni) {
                        const int m = m0 + mi;
                        const int n = n0 + ni;
                        for (int ki = 0; ki < TILE_K; ++ki) {
                            const int k = k0 + ki;
                            if (m < M && n < N && k < K) {
                                acc[mi][ni] += (ACC_T)left[m * K + k] * (ACC_T)right_transposed[n * K + k];
                            }
                        }
                    }
                }
            }
            for (int mi = 0; mi < TILE_M; ++mi) {
                for (int ni = 0; ni < TILE_N; ++ni) {
                    const int m = m0 + mi;
                    const int n = n0 + ni;
                    if (m < M && n < N) output[m * N + n] = (OUT_T)acc[mi][ni];
                }
            }
        }
    }
}

template<
    int ROWS,
    int COLS,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t
>
void softmax_rows(
    const IN_T input[ROWS * COLS],
    OUT_T output[ROWS * COLS]
) {
#pragma HLS INLINE off
    for (int row = 0; row < ROWS; ++row) {
        ACC_T maximum = (ACC_T)input[row * COLS];
        for (int col = 1; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            const ACC_T value = (ACC_T)input[row * COLS + col];
            if (value > maximum) maximum = value;
        }
        ACC_T denominator = (ACC_T)0;
        ACC_T exponentials[COLS];
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            const float shifted = (float)((ACC_T)input[row * COLS + col] - maximum);
            const ACC_T exp_value = (ACC_T)hls::expf(shifted);
            exponentials[col] = exp_value;
            denominator += exp_value;
        }
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            output[row * COLS + col] = (OUT_T)(exponentials[col] / denominator);
        }
    }
}

template<
    int ROWS,
    int COLS,
    typename IN_T = act_t,
    typename OUT_T = act_t
>
void transpose_2d(
    const IN_T input[ROWS * COLS],
    OUT_T output[ROWS * COLS]
) {
#pragma HLS INLINE off
    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            output[col * ROWS + row] = (OUT_T)input[row * COLS + col];
        }
    }
}

template<
    int ROWS,
    int COLS,
    typename IN_T = act_t,
    typename SCALE_T = act_t,
    typename BIAS_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t
>
void layer_norm_rows(
    const IN_T input[ROWS * COLS],
    const SCALE_T scale[COLS],
    const BIAS_T bias[COLS],
    OUT_T output[ROWS * COLS],
    ACC_T epsilon
) {
#pragma HLS INLINE off
    for (int row = 0; row < ROWS; ++row) {
        ACC_T mean = (ACC_T)0;
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            mean += (ACC_T)input[row * COLS + col];
        }
        mean /= (ACC_T)COLS;
        ACC_T variance = (ACC_T)0;
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            const ACC_T centered = (ACC_T)input[row * COLS + col] - mean;
            variance += centered * centered;
        }
        variance /= (ACC_T)COLS;
        const ACC_T inv_std = (ACC_T)(1.0f / hls::sqrtf((float)(variance + epsilon)));
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            const ACC_T normalized = ((ACC_T)input[row * COLS + col] - mean) * inv_std;
            output[row * COLS + col] = (OUT_T)(normalized * (ACC_T)scale[col] + (ACC_T)bias[col]);
        }
    }
}

template<
    int ROWS,
    int COLS,
    typename IN_T = act_t,
    typename SCALE_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t,
    int REDUCE_II = 1,
    int NORMALIZE_II = 1,
    int COL_UNROLL = 1,
    int INPUT_PARTITION = 1,
    int OUTPUT_PARTITION = 1,
    int WEIGHT_PARTITION = 1
>
void rms_norm_rows(
    const IN_T input[ROWS * COLS],
    const SCALE_T scale[COLS],
    OUT_T output[ROWS * COLS],
    ACC_T epsilon
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=scale cyclic factor=WEIGHT_PARTITION
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_PARTITION
    for (int row = 0; row < ROWS; ++row) {
        ACC_T mean_square = (ACC_T)0;
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=REDUCE_II
#pragma HLS UNROLL factor=COL_UNROLL
            const ACC_T value = (ACC_T)input[row * COLS + col];
            mean_square += value * value;
        }
        mean_square /= (ACC_T)COLS;
        const ACC_T inv_rms = (ACC_T)(1.0f / hls::sqrtf((float)(mean_square + epsilon)));
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=NORMALIZE_II
#pragma HLS UNROLL factor=COL_UNROLL
            output[row * COLS + col] = (OUT_T)((ACC_T)input[row * COLS + col] * inv_rms * (ACC_T)scale[col]);
        }
    }
}

template<
    int ROWS,
    int COLS,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t
>
void causal_mask_rows(
    const IN_T input[ROWS * COLS],
    OUT_T output[ROWS * COLS],
    int diagonal,
    ACC_T masked_value
) {
#pragma HLS INLINE off
    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            output[row * COLS + col] = (col > row + diagonal) ? (OUT_T)masked_value : (OUT_T)input[row * COLS + col];
        }
    }
}

template<
    int N,
    typename LEFT_T = act_t,
    typename RIGHT_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t,
    int PIPELINE_II = 1,
    int ELEMENT_UNROLL = 1,
    int INPUT_PARTITION = 1,
    int OUTPUT_PARTITION = 1
>
void mul_vectors(
    const LEFT_T left[N],
    const RIGHT_T right[N],
    OUT_T output[N]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=left cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=right cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_PARTITION
    for (int base = 0; base < N; base += ELEMENT_UNROLL) {
#pragma HLS PIPELINE II=PIPELINE_II
        for (int lane = 0; lane < ELEMENT_UNROLL; ++lane) {
#pragma HLS UNROLL
            const int i = base + lane;
            if (i < N) output[i] = (OUT_T)((ACC_T)left[i] * (ACC_T)right[i]);
        }
    }
}

template<
    int N,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t,
    int PIPELINE_II = 1,
    int ELEMENT_UNROLL = 1,
    int INPUT_PARTITION = 1,
    int OUTPUT_PARTITION = 1
>
void silu_vector(
    const IN_T input[N],
    OUT_T output[N]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_PARTITION
    for (int base = 0; base < N; base += ELEMENT_UNROLL) {
#pragma HLS PIPELINE II=PIPELINE_II
        for (int lane = 0; lane < ELEMENT_UNROLL; ++lane) {
#pragma HLS UNROLL
            const int i = base + lane;
            if (i < N) {
                const ACC_T x = (ACC_T)input[i];
                const ACC_T sigmoid = (ACC_T)(1.0f / (1.0f + hls::expf((float)(-x))));
                output[i] = (OUT_T)(x * sigmoid);
            }
        }
    }
}

template<
    int N,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename SCALE_T = acc_t
>
void scale_vector(
    const IN_T input[N],
    OUT_T output[N],
    SCALE_T scale
) {
#pragma HLS INLINE off
    for (int index = 0; index < N; ++index) {
#pragma HLS PIPELINE II=1
        output[index] = (OUT_T)((SCALE_T)input[index] * scale);
    }
}


template<
    int ROWS,
    int COLS,
    typename IN_T = act_t,
    typename TABLE_T = acc_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t,
    int PIPELINE_II = 1,
    int PAIR_UNROLL = 1,
    int INPUT_PARTITION = 1,
    int OUTPUT_PARTITION = 1
>
void rotary_embedding_pairs(
    const IN_T input[ROWS * COLS],
    const TABLE_T cos_table[],
    const TABLE_T sin_table[],
    OUT_T output[ROWS * COLS],
    int position_offset
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_PARTITION
    static_assert((COLS % 2) == 0, "RoPE requires an even rotary dimension");
    for (int row = 0; row < ROWS; ++row) {
        for (int pair = 0; pair < COLS / 2; ++pair) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=PAIR_UNROLL
            const int i0 = row * COLS + pair * 2;
            const int i1 = i0 + 1;
            const int ti = (row + position_offset) * (COLS / 2) + pair;
            const ACC_T x0 = (ACC_T)input[i0];
            const ACC_T x1 = (ACC_T)input[i1];
            const ACC_T c = (ACC_T)cos_table[ti];
            const ACC_T s = (ACC_T)sin_table[ti];
            output[i0] = (OUT_T)(x0 * c - x1 * s);
            output[i1] = (OUT_T)(x0 * s + x1 * c);
        }
    }
}


template<
    int ROWS,
    int MODEL,
    int HEADS,
    int ROTARY_DIM,
    bool INTERLEAVED = false,
    typename IN_T = act_t,
    typename TABLE_T = acc_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t,
    int PIPELINE_II = 1,
    int PAIR_UNROLL = 1
>
void rotary_embedding_heads(
    const IN_T input[ROWS * MODEL],
    const TABLE_T cos_table[],
    const TABLE_T sin_table[],
    OUT_T output[ROWS * MODEL],
    int position_offset
) {
#pragma HLS INLINE off
    static_assert((MODEL % HEADS) == 0, "MODEL must be divisible by HEADS");
    static_assert((ROTARY_DIM % 2) == 0, "RoPE rotary dimension must be even");
    const int HEAD_DIM = MODEL / HEADS;
    static_assert(ROTARY_DIM <= HEAD_DIM, "RoPE rotary dimension cannot exceed head dimension");

    for (int row = 0; row < ROWS; ++row) {
        for (int head = 0; head < HEADS; ++head) {
            const int head_base = row * MODEL + head * HEAD_DIM;
            // Preserve dimensions outside the rotary portion of every head.
            for (int d = ROTARY_DIM; d < HEAD_DIM; ++d) {
#pragma HLS PIPELINE II=PIPELINE_II
                output[head_base + d] = (OUT_T)input[head_base + d];
            }
            for (int pair = 0; pair < ROTARY_DIM / 2; ++pair) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=PAIR_UNROLL
                const int d0 = INTERLEAVED ? pair * 2 : pair;
                const int d1 = INTERLEAVED ? (pair * 2 + 1) : (pair + ROTARY_DIM / 2);
                const int i0 = head_base + d0;
                const int i1 = head_base + d1;
                const int ti = (row + position_offset) * (ROTARY_DIM / 2) + pair;
                const ACC_T x0 = (ACC_T)input[i0];
                const ACC_T x1 = (ACC_T)input[i1];
                const ACC_T c = (ACC_T)cos_table[ti];
                const ACC_T ss = (ACC_T)sin_table[ti];
                output[i0] = (OUT_T)(x0 * c - x1 * ss);
                output[i1] = (OUT_T)(x0 * ss + x1 * c);
            }
        }
    }
}

template<
    int SEQ,
    int MODEL,
    int HEADS,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename ACC_T = acc_t,
    int SCORE_II = 1,
    int SOFTMAX_MAX_II = 1,
    int SOFTMAX_EXP_II = 1,
    int SOFTMAX_NORM_II = 1,
    int VALUE_II = 1,
    int HEAD_UNROLL = 1,
    int ROW_UNROLL = 1,
    int COL_UNROLL = 1,
    int D_UNROLL = 1,
    int INPUT_PARTITION = 1,
    int OUTPUT_PARTITION = 1
>
void multi_head_attention_serialized(
    const IN_T q[SEQ * MODEL],
    const IN_T k[SEQ * MODEL],
    const IN_T v[SEQ * MODEL],
    OUT_T output[SEQ * MODEL],
    ACC_T scale,
    bool causal,
    ACC_T masked_value
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=k cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=v cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_PARTITION
    static_assert((MODEL % HEADS) == 0, "MODEL must be divisible by HEADS");
    const int HEAD_DIM = MODEL / HEADS;
    ACC_T scores[SEQ * SEQ];
    ACC_T probs[SEQ * SEQ];

    for (int i = 0; i < SEQ * MODEL; ++i) {
#pragma HLS PIPELINE II=VALUE_II
        output[i] = (OUT_T)0;
    }

    // One score/softmax/value engine is deliberately reused across heads.
    for (int head = 0; head < HEADS; ++head) {
#pragma HLS UNROLL factor=HEAD_UNROLL
        for (int row = 0; row < SEQ; ++row) {
#pragma HLS UNROLL factor=ROW_UNROLL
            for (int col = 0; col < SEQ; ++col) {
#pragma HLS UNROLL factor=COL_UNROLL
#pragma HLS PIPELINE II=SCORE_II
                ACC_T acc = (ACC_T)0;
                for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL factor=D_UNROLL
                    const int qi = row * MODEL + head * HEAD_DIM + d;
                    const int ki = col * MODEL + head * HEAD_DIM + d;
                    acc += (ACC_T)q[qi] * (ACC_T)k[ki];
                }
                ACC_T score = acc * scale;
                if (causal && col > row) score = masked_value;
                scores[row * SEQ + col] = score;
            }
        }

        for (int row = 0; row < SEQ; ++row) {
#pragma HLS UNROLL factor=ROW_UNROLL
            ACC_T maximum = scores[row * SEQ];
            for (int col = 1; col < SEQ; ++col) {
#pragma HLS PIPELINE II=SOFTMAX_MAX_II
                if (scores[row * SEQ + col] > maximum) maximum = scores[row * SEQ + col];
            }
            ACC_T denominator = (ACC_T)0;
            for (int col = 0; col < SEQ; ++col) {
#pragma HLS UNROLL factor=COL_UNROLL
#pragma HLS PIPELINE II=SOFTMAX_EXP_II
                const ACC_T e = (ACC_T)hls::expf((float)(scores[row * SEQ + col] - maximum));
                probs[row * SEQ + col] = e;
                denominator += e;
            }
            for (int col = 0; col < SEQ; ++col) {
#pragma HLS UNROLL factor=COL_UNROLL
#pragma HLS PIPELINE II=SOFTMAX_NORM_II
                probs[row * SEQ + col] = probs[row * SEQ + col] / denominator;
            }
        }

        for (int row = 0; row < SEQ; ++row) {
#pragma HLS UNROLL factor=ROW_UNROLL
            for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL factor=D_UNROLL
#pragma HLS PIPELINE II=VALUE_II
                ACC_T acc = (ACC_T)0;
                for (int col = 0; col < SEQ; ++col) {
#pragma HLS UNROLL factor=COL_UNROLL
                    const int vi = col * MODEL + head * HEAD_DIM + d;
                    acc += probs[row * SEQ + col] * (ACC_T)v[vi];
                }
                output[row * MODEL + head * HEAD_DIM + d] = (OUT_T)acc;
            }
        }
    }
}



template<
    int QSEQ,
    int KV_CAP,
    int MODEL,
    int HEADS,
    int KV_HEADS,
    typename IN_T = act_t,
    typename OUT_T = act_t,
    typename LEN_T = ap_int<32>,
    typename ACC_T = acc_t,
    int SCORE_II = 1,
    int SOFTMAX_MAX_II = 1,
    int SOFTMAX_EXP_II = 1,
    int SOFTMAX_NORM_II = 1,
    int VALUE_II = 1,
    int HEAD_UNROLL = 1,
    int ROW_UNROLL = 1,
    int COL_UNROLL = 1,
    int D_UNROLL = 1,
    int INPUT_PARTITION = 1,
    int OUTPUT_PARTITION = 1
>
void multi_head_attention_cached_serialized(
    const IN_T q[QSEQ * MODEL],
    const IN_T k[KV_CAP * (MODEL / HEADS) * KV_HEADS],
    const IN_T v[KV_CAP * (MODEL / HEADS) * KV_HEADS],
    const LEN_T valid_length_in[1],
    OUT_T output[QSEQ * MODEL],
    ACC_T scale,
    bool causal,
    ACC_T masked_value
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=k cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=v cyclic factor=INPUT_PARTITION
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_PARTITION
    static_assert((MODEL % HEADS) == 0, "MODEL must be divisible by HEADS");
    static_assert(KV_HEADS > 0 && HEADS % KV_HEADS == 0, "HEADS must be divisible by KV_HEADS");
    const int HEAD_DIM = MODEL / HEADS;
    const int KV_MODEL = HEAD_DIM * KV_HEADS;
    const int Q_PER_KV = HEADS / KV_HEADS;
    int valid_length = (int)valid_length_in[0];
    if (valid_length < 0) valid_length = 0;
    if (valid_length > KV_CAP) valid_length = KV_CAP;
    const int query_base = valid_length > QSEQ ? valid_length - QSEQ : 0;
    ACC_T scores[QSEQ * KV_CAP];
    ACC_T probs[QSEQ * KV_CAP];

    for (int i = 0; i < QSEQ * MODEL; ++i) {
#pragma HLS PIPELINE II=VALUE_II
        output[i] = (OUT_T)0;
    }

    for (int head = 0; head < HEADS; ++head) {
#pragma HLS UNROLL factor=HEAD_UNROLL
        for (int row = 0; row < QSEQ; ++row) {
#pragma HLS UNROLL factor=ROW_UNROLL
            for (int col = 0; col < KV_CAP; ++col) {
#pragma HLS UNROLL factor=COL_UNROLL
#pragma HLS PIPELINE II=SCORE_II
                const bool valid_col = col < valid_length;
                const bool causal_ok = !causal || col <= (query_base + row);
                ACC_T score = masked_value;
                if (valid_col && causal_ok) {
                    ACC_T acc = (ACC_T)0;
                    for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL factor=D_UNROLL
                        const int qi = row * MODEL + head * HEAD_DIM + d;
                        const int kv_head = head / Q_PER_KV;
                        const int ki = col * KV_MODEL + kv_head * HEAD_DIM + d;
                        acc += (ACC_T)q[qi] * (ACC_T)k[ki];
                    }
                    score = acc * scale;
                }
                scores[row * KV_CAP + col] = score;
            }
        }

        for (int row = 0; row < QSEQ; ++row) {
#pragma HLS UNROLL factor=ROW_UNROLL
            ACC_T maximum = masked_value;
            bool have_value = false;
            for (int col = 0; col < KV_CAP; ++col) {
#pragma HLS PIPELINE II=SOFTMAX_MAX_II
                const bool allowed = col < valid_length && (!causal || col <= (query_base + row));
                if (allowed && (!have_value || scores[row * KV_CAP + col] > maximum)) {
                    maximum = scores[row * KV_CAP + col];
                    have_value = true;
                }
            }
            ACC_T denominator = (ACC_T)0;
            for (int col = 0; col < KV_CAP; ++col) {
#pragma HLS UNROLL factor=COL_UNROLL
#pragma HLS PIPELINE II=SOFTMAX_EXP_II
                const bool allowed = col < valid_length && (!causal || col <= (query_base + row));
                const ACC_T e = (allowed && have_value)
                    ? (ACC_T)hls::expf((float)(scores[row * KV_CAP + col] - maximum))
                    : (ACC_T)0;
                probs[row * KV_CAP + col] = e;
                denominator += e;
            }
            if (denominator <= (ACC_T)0) denominator = (ACC_T)1;
            for (int col = 0; col < KV_CAP; ++col) {
#pragma HLS UNROLL factor=COL_UNROLL
#pragma HLS PIPELINE II=SOFTMAX_NORM_II
                probs[row * KV_CAP + col] = probs[row * KV_CAP + col] / denominator;
            }
        }

        for (int row = 0; row < QSEQ; ++row) {
#pragma HLS UNROLL factor=ROW_UNROLL
            for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL factor=D_UNROLL
#pragma HLS PIPELINE II=VALUE_II
                ACC_T acc = (ACC_T)0;
                for (int col = 0; col < KV_CAP; ++col) {
#pragma HLS UNROLL factor=COL_UNROLL
                    if (col < valid_length) {
                        const int kv_head = head / Q_PER_KV;
                        const int vi = col * KV_MODEL + kv_head * HEAD_DIM + d;
                        acc += probs[row * KV_CAP + col] * (ACC_T)v[vi];
                    }
                }
                output[row * MODEL + head * HEAD_DIM + d] = (OUT_T)acc;
            }
        }
    }
}



template<int M, int K, int N, typename DY_T = grad_act_t, typename RIGHT_T = wgt_t, typename DX_T = grad_act_t, typename ACC_T = acc_t, int PIPELINE_II = 1, int M_UNROLL = 1, int K_UNROLL = 1, int N_UNROLL = 1, int GRAD_PARTITION = 1, int WEIGHT_PARTITION = 1>
void matmul_backward_left_accumulate(const DY_T dy[M * N], const RIGHT_T right[K * N], DX_T dx[M * K]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=dy cyclic factor=GRAD_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=right cyclic factor=WEIGHT_PARTITION dim=1
    for (int m = 0; m < M; ++m) {
#pragma HLS UNROLL factor=M_UNROLL
        for (int k = 0; k < K; ++k) {
#pragma HLS UNROLL factor=K_UNROLL
#pragma HLS PIPELINE II=PIPELINE_II
        ACC_T acc = (ACC_T)0;
        for (int n = 0; n < N; ++n) {
#pragma HLS UNROLL factor=N_UNROLL
            acc += (ACC_T)dy[m * N + n] * (ACC_T)right[k * N + n];
        }
        dx[m * K + k] += (DX_T)acc;
        }
    }
}

template<int M, int K, int N, typename LEFT_T = act_t, typename DY_T = grad_act_t, typename DW_T = grad_wgt_t, typename ACC_T = acc_t, int PIPELINE_II = 1, int M_UNROLL = 1, int K_UNROLL = 1, int N_UNROLL = 1, int INPUT_PARTITION = 1, int GRAD_PARTITION = 1>
void matmul_weight_grad(const LEFT_T left[M * K], const DY_T dy[M * N], DW_T dw[K * N]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=left cyclic factor=INPUT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=dy cyclic factor=GRAD_PARTITION dim=1
    for (int k = 0; k < K; ++k) {
#pragma HLS UNROLL factor=K_UNROLL
        for (int n = 0; n < N; ++n) {
#pragma HLS UNROLL factor=N_UNROLL
#pragma HLS PIPELINE II=PIPELINE_II
        ACC_T acc = (ACC_T)0;
        for (int m = 0; m < M; ++m) {
#pragma HLS UNROLL factor=M_UNROLL
            acc += (ACC_T)left[m * K + k] * (ACC_T)dy[m * N + n];
        }
        dw[k * N + n] = (DW_T)acc;
        }
    }
}

template<int M, int K, int N, typename LEFT_T = act_t, typename DY_T = grad_act_t, typename DR_T = grad_act_t, typename ACC_T = acc_t, int PIPELINE_II = 1, int M_UNROLL = 1, int K_UNROLL = 1, int N_UNROLL = 1, int INPUT_PARTITION = 1, int GRAD_PARTITION = 1>
void matmul_backward_right_accumulate(const LEFT_T left[M * K], const DY_T dy[M * N], DR_T dr[K * N]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=left cyclic factor=INPUT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=dy cyclic factor=GRAD_PARTITION dim=1
    for (int k = 0; k < K; ++k) {
#pragma HLS UNROLL factor=K_UNROLL
        for (int n = 0; n < N; ++n) {
#pragma HLS UNROLL factor=N_UNROLL
#pragma HLS PIPELINE II=PIPELINE_II
        ACC_T acc = (ACC_T)0;
        for (int m = 0; m < M; ++m) {
#pragma HLS UNROLL factor=M_UNROLL
            acc += (ACC_T)left[m * K + k] * (ACC_T)dy[m * N + n];
        }
        dr[k * N + n] += (DR_T)acc;
        }
    }
}

template<int N, typename LEFT_T = act_t, typename RIGHT_T = act_t, typename DY_T = grad_act_t, typename DX_T = grad_act_t, typename ACC_T = acc_t, int PIPELINE_II = 1, int ELEMENT_UNROLL = 1>
void mul_backward_accumulate(const LEFT_T left[N], const RIGHT_T right[N], const DY_T dy[N], DX_T dleft[N], DX_T dright[N]) {
#pragma HLS INLINE off
    for (int base = 0; base < N; base += ELEMENT_UNROLL) {
#pragma HLS PIPELINE II=PIPELINE_II
        for (int lane = 0; lane < ELEMENT_UNROLL; ++lane) {
#pragma HLS UNROLL
            const int i = base + lane;
            if (i < N) {
                const ACC_T g = (ACC_T)dy[i];
                dleft[i] += (DX_T)(g * (ACC_T)right[i]);
                dright[i] += (DX_T)(g * (ACC_T)left[i]);
            }
        }
    }
}

template<int N, typename IN_T = act_t, typename DY_T = grad_act_t, typename DX_T = grad_act_t, typename ACC_T = acc_t, int PIPELINE_II = 1, int ELEMENT_UNROLL = 1>
void silu_backward_accumulate(const IN_T input[N], const DY_T dy[N], DX_T dx[N]) {
#pragma HLS INLINE off
    for (int base = 0; base < N; base += ELEMENT_UNROLL) {
#pragma HLS PIPELINE II=PIPELINE_II
        for (int lane = 0; lane < ELEMENT_UNROLL; ++lane) {
#pragma HLS UNROLL
            const int i = base + lane;
            if (i < N) {
                const ACC_T x = (ACC_T)input[i];
                const ACC_T sig = (ACC_T)(1.0f / (1.0f + hls::expf((float)(-x))));
                dx[i] += (DX_T)((ACC_T)dy[i] * (sig + x * sig * ((ACC_T)1 - sig)));
            }
        }
    }
}

template<int ROWS, int COLS, typename DY_T = grad_act_t, typename DX_T = grad_act_t>
void transpose_backward_accumulate(const DY_T dy[ROWS * COLS], DX_T dx[ROWS * COLS]) {
#pragma HLS INLINE off
    for (int row = 0; row < ROWS; ++row) for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
        dx[row * COLS + col] += (DX_T)dy[col * ROWS + row];
    }
}

template<int ROWS, int COLS, typename IN_T = act_t, typename SCALE_T = wgt_t, typename DY_T = grad_act_t, typename DX_T = grad_act_t, typename DG_T = grad_wgt_t, typename ACC_T = acc_t, int REDUCE_II = 1, int NORMALIZE_II = 1, int COL_UNROLL = 1>
void rms_norm_backward_rows(const IN_T input[ROWS * COLS], const SCALE_T scale[COLS], const DY_T dy[ROWS * COLS], DX_T dx[ROWS * COLS], DG_T dscale[COLS], ACC_T epsilon) {
#pragma HLS INLINE off
    for (int col = 0; col < COLS; ++col) dscale[col] = (DG_T)0;
    for (int row = 0; row < ROWS; ++row) {
        ACC_T mean_square = (ACC_T)0;
        for (int col = 0; col < COLS; ++col) { const ACC_T x = (ACC_T)input[row * COLS + col]; mean_square += x * x; }
        mean_square /= (ACC_T)COLS;
        const ACC_T inv = (ACC_T)(1.0f / hls::sqrtf((float)(mean_square + epsilon)));
        ACC_T mean_gx = (ACC_T)0;
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=REDUCE_II
#pragma HLS UNROLL factor=COL_UNROLL
            const ACC_T x = (ACC_T)input[row * COLS + col];
            const ACC_T g = (ACC_T)dy[row * COLS + col] * (ACC_T)scale[col];
            mean_gx += g * x;
            dscale[col] += (DG_T)((ACC_T)dy[row * COLS + col] * x * inv);
        }
        mean_gx /= (ACC_T)COLS;
        const ACC_T inv3 = inv * inv * inv;
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=NORMALIZE_II
#pragma HLS UNROLL factor=COL_UNROLL
            const ACC_T x = (ACC_T)input[row * COLS + col];
            const ACC_T g = (ACC_T)dy[row * COLS + col] * (ACC_T)scale[col];
            dx[row * COLS + col] += (DX_T)(g * inv - x * inv3 * mean_gx);
        }
    }
}

template<int ROWS, int COLS, typename IN_T = act_t, typename SCALE_T = wgt_t, typename DY_T = grad_act_t, typename DX_T = grad_act_t, typename DG_T = grad_wgt_t, typename DB_T = grad_bias_t, typename ACC_T = acc_t>
void layer_norm_backward_rows(const IN_T input[ROWS * COLS], const SCALE_T scale[COLS], const DY_T dy[ROWS * COLS], DX_T dx[ROWS * COLS], DG_T dscale[COLS], DB_T dbias[COLS], ACC_T epsilon) {
#pragma HLS INLINE off
    for (int col = 0; col < COLS; ++col) { dscale[col] = (DG_T)0; dbias[col] = (DB_T)0; }
    for (int row = 0; row < ROWS; ++row) {
        ACC_T mean = (ACC_T)0;
        for (int col = 0; col < COLS; ++col) mean += (ACC_T)input[row * COLS + col];
        mean /= (ACC_T)COLS;
        ACC_T variance = (ACC_T)0;
        for (int col = 0; col < COLS; ++col) { const ACC_T c = (ACC_T)input[row * COLS + col] - mean; variance += c * c; }
        variance /= (ACC_T)COLS;
        const ACC_T inv = (ACC_T)(1.0f / hls::sqrtf((float)(variance + epsilon)));
        ACC_T sum_g = (ACC_T)0, sum_gn = (ACC_T)0;
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            const ACC_T normalized = ((ACC_T)input[row * COLS + col] - mean) * inv;
            const ACC_T g = (ACC_T)dy[row * COLS + col] * (ACC_T)scale[col];
            sum_g += g; sum_gn += g * normalized;
            dscale[col] += (DG_T)((ACC_T)dy[row * COLS + col] * normalized);
            dbias[col] += (DB_T)dy[row * COLS + col];
        }
        for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
            const ACC_T normalized = ((ACC_T)input[row * COLS + col] - mean) * inv;
            const ACC_T g = (ACC_T)dy[row * COLS + col] * (ACC_T)scale[col];
            dx[row * COLS + col] += (DX_T)((inv / (ACC_T)COLS) * ((ACC_T)COLS * g - sum_g - normalized * sum_gn));
        }
    }
}

template<int ROWS, int COLS, typename DY_T = grad_act_t, typename DX_T = grad_act_t>
void causal_mask_backward_rows(const DY_T dy[ROWS * COLS], DX_T dx[ROWS * COLS], int diagonal) {
#pragma HLS INLINE off
    for (int row = 0; row < ROWS; ++row) for (int col = 0; col < COLS; ++col) {
#pragma HLS PIPELINE II=1
        if (col <= row + diagonal) dx[row * COLS + col] += (DX_T)dy[row * COLS + col];
    }
}

template<int ROWS, int COLS, typename DY_T = grad_act_t, typename TABLE_T = acc_t, typename DX_T = grad_act_t, typename ACC_T = acc_t, int PIPELINE_II = 1, int PAIR_UNROLL = 1>
void rotary_embedding_backward_pairs(const DY_T dy[ROWS * COLS], const TABLE_T cos_table[], const TABLE_T sin_table[], DX_T dx[ROWS * COLS], int position_offset) {
#pragma HLS INLINE off
    for (int row = 0; row < ROWS; ++row) for (int pair = 0; pair < COLS / 2; ++pair) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=PAIR_UNROLL
        const int i0 = row * COLS + pair * 2, i1 = i0 + 1;
        const int ti = (row + position_offset) * (COLS / 2) + pair;
        const ACC_T g0 = (ACC_T)dy[i0], g1 = (ACC_T)dy[i1], c = (ACC_T)cos_table[ti], ss = (ACC_T)sin_table[ti];
        dx[i0] += (DX_T)(g0 * c + g1 * ss);
        dx[i1] += (DX_T)(-g0 * ss + g1 * c);
    }
}

template<int SEQ, int MODEL, int HEADS, typename IN_T = act_t, typename DY_T = grad_act_t, typename DX_T = grad_act_t, typename ACC_T = acc_t, int SCORE_II = 1, int SOFTMAX_EXP_II = 1, int SOFTMAX_NORM_II = 1, int DQ_DV_II = 1, int DK_II = 1, int HEAD_UNROLL = 1, int ROW_UNROLL = 1, int COL_UNROLL = 1, int D_UNROLL = 1>
void multi_head_attention_backward_serialized(const IN_T q[SEQ * MODEL], const IN_T k[SEQ * MODEL], const IN_T v[SEQ * MODEL], const DY_T dy[SEQ * MODEL], DX_T dq[SEQ * MODEL], DX_T dk[SEQ * MODEL], DX_T dv[SEQ * MODEL], ACC_T scale, bool causal, ACC_T masked_value) {
#pragma HLS INLINE off
    const int HEAD_DIM = MODEL / HEADS;
    ACC_T scores[SEQ * SEQ], probs[SEQ * SEQ], dprobs[SEQ * SEQ], dscores[SEQ * SEQ];
    for (int head = 0; head < HEADS; ++head) {
#pragma HLS UNROLL factor=HEAD_UNROLL
        for (int row = 0; row < SEQ; ++row) {
#pragma HLS UNROLL factor=ROW_UNROLL
        for (int col = 0; col < SEQ; ++col) {
#pragma HLS UNROLL factor=COL_UNROLL
#pragma HLS PIPELINE II=SCORE_II
            ACC_T acc = (ACC_T)0;
            for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL factor=D_UNROLL
                acc += (ACC_T)q[row * MODEL + head * HEAD_DIM + d] * (ACC_T)k[col * MODEL + head * HEAD_DIM + d];
            }
            ACC_T score = acc * scale;
            if (causal && col > row) score = masked_value;
            scores[row * SEQ + col] = score;
        }
        }
        for (int row = 0; row < SEQ; ++row) {
#pragma HLS UNROLL factor=ROW_UNROLL
            ACC_T maximum = scores[row * SEQ];
            for (int col = 1; col < SEQ; ++col) if (scores[row * SEQ + col] > maximum) maximum = scores[row * SEQ + col];
            ACC_T denom = (ACC_T)0;
            for (int col = 0; col < SEQ; ++col) {
#pragma HLS PIPELINE II=SOFTMAX_EXP_II
                probs[row * SEQ + col] = (ACC_T)hls::expf((float)(scores[row * SEQ + col] - maximum)); denom += probs[row * SEQ + col]; }
            for (int col = 0; col < SEQ; ++col) {
#pragma HLS PIPELINE II=SOFTMAX_NORM_II
                probs[row * SEQ + col] /= denom;
            }
            for (int col = 0; col < SEQ; ++col) {
                ACC_T acc = (ACC_T)0;
                for (int d = 0; d < HEAD_DIM; ++d) acc += (ACC_T)dy[row * MODEL + head * HEAD_DIM + d] * (ACC_T)v[col * MODEL + head * HEAD_DIM + d];
                dprobs[row * SEQ + col] = acc;
            }
            ACC_T dot = (ACC_T)0;
            for (int col = 0; col < SEQ; ++col) dot += dprobs[row * SEQ + col] * probs[row * SEQ + col];
            for (int col = 0; col < SEQ; ++col) { ACC_T ds = probs[row * SEQ + col] * (dprobs[row * SEQ + col] - dot); if (causal && col > row) ds = (ACC_T)0; dscores[row * SEQ + col] = ds; }
        }
        for (int row = 0; row < SEQ; ++row) {
#pragma HLS UNROLL factor=ROW_UNROLL
        for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL factor=D_UNROLL
#pragma HLS PIPELINE II=DQ_DV_II
            ACC_T aq = (ACC_T)0, av = (ACC_T)0;
            for (int col = 0; col < SEQ; ++col) {
                aq += dscores[row * SEQ + col] * (ACC_T)k[col * MODEL + head * HEAD_DIM + d] * scale;
                av += probs[col * SEQ + row] * (ACC_T)dy[col * MODEL + head * HEAD_DIM + d];
            }
            dq[row * MODEL + head * HEAD_DIM + d] += (DX_T)aq;
            dv[row * MODEL + head * HEAD_DIM + d] += (DX_T)av;
        }
        }
        for (int col = 0; col < SEQ; ++col) {
#pragma HLS UNROLL factor=COL_UNROLL
        for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL factor=D_UNROLL
#pragma HLS PIPELINE II=DK_II
            ACC_T ak = (ACC_T)0;
            for (int row = 0; row < SEQ; ++row) ak += dscores[row * SEQ + col] * (ACC_T)q[row * MODEL + head * HEAD_DIM + d] * scale;
            dk[col * MODEL + head * HEAD_DIM + d] += (DX_T)ak;
        }
        }
    }
}


// Cross-specialization phase-shared GEMM engines.
// These engines intentionally use bounded compile-time maxima with runtime
// active dimensions so shape-compatible MatMul calls can share one RTL module.
// They are emitted only when the network phase-sharing transform rewrites calls.
template<
    int MAX_M, int MAX_K, int MAX_N,
    typename LEFT_T = act_t, typename RIGHT_T = wgt_t,
    typename OUT_T = act_t, typename ACC_T = acc_t,
    int TILE_M = 1, int TILE_N = 1, int TILE_K = 1,
    int PIPELINE_II = 1, int M_UNROLL = 1, int N_UNROLL = 1, int K_UNROLL = 1,
    int INPUT_PARTITION = 1, int OUTPUT_PARTITION = 1, int WEIGHT_PARTITION = 1
>
void phase_shared_matmul_forward(
    const LEFT_T left[MAX_M * MAX_K],
    const RIGHT_T right[MAX_K * MAX_N],
    OUT_T output[MAX_M * MAX_N],
    int active_m, int active_k, int active_n
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=left cyclic factor=INPUT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=right cyclic factor=WEIGHT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_PARTITION dim=1
    for (int m0 = 0; m0 < MAX_M; m0 += TILE_M) {
        for (int n0 = 0; n0 < MAX_N; n0 += TILE_N) {
            ACC_T acc[TILE_M][TILE_N];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
            for (int mi = 0; mi < TILE_M; ++mi) {
#pragma HLS UNROLL factor=M_UNROLL
                for (int ni = 0; ni < TILE_N; ++ni) {
#pragma HLS UNROLL factor=N_UNROLL
                    acc[mi][ni] = (ACC_T)0;
                }
            }
            for (int k0 = 0; k0 < MAX_K; k0 += TILE_K) {
#pragma HLS PIPELINE II=PIPELINE_II
                for (int mi = 0; mi < TILE_M; ++mi) {
#pragma HLS UNROLL factor=M_UNROLL
                    for (int ni = 0; ni < TILE_N; ++ni) {
#pragma HLS UNROLL factor=N_UNROLL
                        const int m = m0 + mi;
                        const int n = n0 + ni;
                        for (int ki = 0; ki < TILE_K; ++ki) {
#pragma HLS UNROLL factor=K_UNROLL
                            const int k = k0 + ki;
                            if (m < active_m && n < active_n && k < active_k) {
                                acc[mi][ni] += (ACC_T)left[m * active_k + k] *
                                               (ACC_T)right[k * active_n + n];
                            }
                        }
                    }
                }
            }
            for (int mi = 0; mi < TILE_M; ++mi) {
#pragma HLS UNROLL factor=M_UNROLL
                for (int ni = 0; ni < TILE_N; ++ni) {
#pragma HLS UNROLL factor=N_UNROLL
                    const int m = m0 + mi;
                    const int n = n0 + ni;
                    if (m < active_m && n < active_n) {
                        output[m * active_n + n] = (OUT_T)acc[mi][ni];
                    }
                }
            }
        }
    }
}

template<
    int MAX_M, int MAX_K, int MAX_N,
    typename DY_T = grad_act_t, typename RIGHT_T = wgt_t,
    typename DX_T = grad_act_t, typename ACC_T = acc_t,
    int PIPELINE_II = 1, int M_UNROLL = 1, int K_UNROLL = 1, int N_UNROLL = 1,
    int GRAD_PARTITION = 1, int WEIGHT_PARTITION = 1
>
void phase_shared_matmul_backward_left(
    const DY_T dy[MAX_M * MAX_N],
    const RIGHT_T right[MAX_K * MAX_N],
    DX_T dx[MAX_M * MAX_K],
    int active_m, int active_k, int active_n
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=dy cyclic factor=GRAD_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=right cyclic factor=WEIGHT_PARTITION dim=1
    for (int m = 0; m < MAX_M; ++m) {
#pragma HLS UNROLL factor=M_UNROLL
        for (int k = 0; k < MAX_K; ++k) {
#pragma HLS UNROLL factor=K_UNROLL
#pragma HLS PIPELINE II=PIPELINE_II
            if (m < active_m && k < active_k) {
                ACC_T acc = (ACC_T)0;
                for (int n = 0; n < MAX_N; ++n) {
#pragma HLS UNROLL factor=N_UNROLL
                    if (n < active_n) {
                        acc += (ACC_T)dy[m * active_n + n] *
                               (ACC_T)right[k * active_n + n];
                    }
                }
                dx[m * active_k + k] += (DX_T)acc;
            }
        }
    }
}

template<
    int MAX_M, int MAX_K, int MAX_N,
    typename LEFT_T = act_t, typename DY_T = grad_act_t,
    typename DW_T = grad_wgt_t, typename ACC_T = acc_t,
    int PIPELINE_II = 1, int M_UNROLL = 1, int K_UNROLL = 1, int N_UNROLL = 1,
    int INPUT_PARTITION = 1, int GRAD_PARTITION = 1
>
void phase_shared_matmul_weight_grad(
    const LEFT_T left[MAX_M * MAX_K],
    const DY_T dy[MAX_M * MAX_N],
    DW_T dw[MAX_K * MAX_N],
    int active_m, int active_k, int active_n
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=left cyclic factor=INPUT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=dy cyclic factor=GRAD_PARTITION dim=1
    for (int k = 0; k < MAX_K; ++k) {
#pragma HLS UNROLL factor=K_UNROLL
        for (int n = 0; n < MAX_N; ++n) {
#pragma HLS UNROLL factor=N_UNROLL
#pragma HLS PIPELINE II=PIPELINE_II
            if (k < active_k && n < active_n) {
                ACC_T acc = (ACC_T)0;
                for (int m = 0; m < MAX_M; ++m) {
#pragma HLS UNROLL factor=M_UNROLL
                    if (m < active_m) {
                        acc += (ACC_T)left[m * active_k + k] *
                               (ACC_T)dy[m * active_n + n];
                    }
                }
                dw[k * active_n + n] = (DW_T)acc;
            }
        }
    }
}

} // namespace fpgai
'''
