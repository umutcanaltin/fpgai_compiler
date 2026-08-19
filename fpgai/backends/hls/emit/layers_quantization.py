from __future__ import annotations


def emit_quantization_h() -> str:
    return r'''#pragma once

#include <ap_int.h>

namespace fpgai {

static inline ap_int<64> quantized_round_shift(
    ap_int<64> value,
    int shift,
    int rounding_mode
) {
#pragma HLS INLINE
    if (shift <= 0) return value;
    const ap_int<64> divisor = ((ap_int<64>)1) << shift;
    if (rounding_mode == 1) { // floor
        if (value >= 0) return value >> shift;
        const ap_int<64> magnitude = -value;
        return -((magnitude + divisor - 1) >> shift);
    }
    if (rounding_mode == 2) { // ceil
        if (value >= 0) return (value + divisor - 1) >> shift;
        return -((-value) >> shift);
    }
    // nearest, symmetric half-away-from-zero
    const ap_int<64> half = divisor >> 1;
    if (value >= 0) return (value + half) >> shift;
    return -(((-value) + half) >> shift);
}

static inline ap_int<64> quantized_apply_overflow(
    ap_int<64> value,
    int qmin,
    int qmax,
    int saturation_mode
) {
#pragma HLS INLINE
    if (saturation_mode == 0) { // saturate
        if (value < qmin) return qmin;
        if (value > qmax) return qmax;
        return value;
    }
    const ap_int<64> width = (ap_int<64>)qmax - (ap_int<64>)qmin + 1;
    ap_int<64> shifted = value - qmin;
    ap_int<64> wrapped = shifted % width;
    if (wrapped < 0) wrapped += width;
    return wrapped + qmin;
}

static inline ap_int<64> quantized_requantize_centered(
    ap_int<64> centered_value,
    int multiplier,
    int shift,
    int destination_zero,
    int qmin,
    int qmax,
    int rounding_mode,
    int saturation_mode
) {
#pragma HLS INLINE
    ap_int<64> product = centered_value * (ap_int<64>)multiplier;
    ap_int<64> shifted = quantized_round_shift(product, shift, rounding_mode);
    return quantized_apply_overflow(
        shifted + destination_zero,
        qmin,
        qmax,
        saturation_mode
    );
}

} // namespace fpgai
'''
