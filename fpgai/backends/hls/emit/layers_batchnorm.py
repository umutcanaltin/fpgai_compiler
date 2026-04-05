from __future__ import annotations


def emit_batchnorm_h() -> str:
    return r'''#pragma once
#include "fpgai_types.h"
#include <hls_math.h>

#ifndef FPGAI_PIPELINE_II
#define FPGAI_PIPELINE_II 1
#endif

namespace fpgai {

// HWC-flat convention to match conv/pool emitters:
// flat index = hw * C + c

template<int C, int HW>
void batchnorm_train_forward(
    const act_t x[C * HW],
    act_t y[C * HW],
    const wgt_t gamma[C],
    const bias_t beta[C],
    acc_t mean[C],
    acc_t var[C],
    act_t xhat[C * HW],
    act_t eps = (act_t)1e-5
) {
#pragma HLS INLINE off
    for (int c = 0; c < C; ++c) {
        acc_t m = 0;
        acc_t v = 0;

        for (int hw = 0; hw < HW; ++hw) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
            int idx = hw * C + c;
            m += (acc_t)x[idx];
        }
        m /= (acc_t)HW;

        for (int hw = 0; hw < HW; ++hw) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
            int idx = hw * C + c;
            acc_t d = (acc_t)x[idx] - m;
            v += d * d;
        }
        v /= (acc_t)HW;

        mean[c] = m;
        var[c] = v;
    }

    for (int c = 0; c < C; ++c) {
        acc_t inv_std = (acc_t)1.0 / hls::sqrt((acc_t)var[c] + (acc_t)eps);
        for (int hw = 0; hw < HW; ++hw) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
            int idx = hw * C + c;
            acc_t xn = ((acc_t)x[idx] - (acc_t)mean[c]) * inv_std;
            xhat[idx] = (act_t)xn;
            y[idx] = (act_t)((acc_t)gamma[c] * xn + (acc_t)beta[c]);
        }
    }
}

template<int C, int HW>
void batchnorm_param_grad(
    const grad_act_t dY[C * HW],
    const act_t xhat[C * HW],
    grad_wgt_t dGamma[C],
    grad_bias_t dBeta[C]
) {
#pragma HLS INLINE off
    for (int c = 0; c < C; ++c) {
        acc_t dg = 0;
        acc_t db = 0;

        for (int hw = 0; hw < HW; ++hw) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
            int idx = hw * C + c;
            dg += (acc_t)dY[idx] * (acc_t)xhat[idx];
            db += (acc_t)dY[idx];
        }

        dGamma[c] = (grad_wgt_t)dg;
        dBeta[c] = (grad_bias_t)db;
    }
}

// Simple input gradient for now.
// This keeps training structurally complete; exact BN backward can be swapped in later.
template<int C, int HW>
void batchnorm_backward_input_simple(
    const grad_act_t dY[C * HW],
    const wgt_t gamma[C],
    grad_act_t dX[C * HW]
) {
#pragma HLS INLINE off
    for (int c = 0; c < C; ++c) {
        for (int hw = 0; hw < HW; ++hw) {
#pragma HLS PIPELINE II=FPGAI_PIPELINE_II
            int idx = hw * C + c;
            dX[idx] = (grad_act_t)((acc_t)dY[idx] * (acc_t)gamma[c]);
        }
    }
}

} // namespace fpgai
'''


def emit_batchnorm_cpp() -> str:
    return '#include "layers/batchnorm.h"\n'