from __future__ import annotations

from typing import Any, Dict
import numpy as np


def spec_fractional_bits(spec: Dict[str, Any]) -> int:
    return int(spec["total_bits"]) - int(spec["int_bits"])


def spec_scale(spec: Dict[str, Any]) -> float:
    return float(2 ** spec_fractional_bits(spec))


def spec_min_value(spec: Dict[str, Any]) -> float:
    total_bits = int(spec["total_bits"])
    frac_bits = spec_fractional_bits(spec)
    return float(-(2 ** (total_bits - 1)) / (2 ** frac_bits))


def spec_max_value(spec: Dict[str, Any]) -> float:
    total_bits = int(spec["total_bits"])
    frac_bits = spec_fractional_bits(spec)
    return float(((2 ** (total_bits - 1)) - 1) / (2 ** frac_bits))


def quantize_array(
    x: np.ndarray,
    spec: Dict[str, Any],
    rounding: str = "nearest",
    overflow: str = "saturate",
) -> np.ndarray:
    """Quantize values to a signed fixed-point storage format.

    ``rounding="trunc"`` preserves the historical FPGAI behavior of
    truncating toward zero.  Xilinx ``ap_fixed`` defaults are represented by
    ``rounding="ap_trn"`` and ``overflow="wrap"``; AP_TRN removes low
    bits from a two's-complement value, which is floor-like for negative
    numbers rather than truncation toward zero.
    """
    x = np.asarray(x, dtype=np.float32)
    scale = spec_scale(spec)
    total_bits = int(spec["total_bits"])
    qmin_int = -(2 ** (total_bits - 1))
    qmax_int = (2 ** (total_bits - 1)) - 1

    scaled = x.astype(np.float64) * scale
    if rounding == "nearest":
        q = np.round(scaled)
    elif rounding == "trunc":
        q = np.trunc(scaled)
    elif rounding in {"floor", "ap_trn"}:
        q = np.floor(scaled)
    else:
        raise ValueError(f"Unsupported rounding mode: {rounding}")

    if overflow in {"saturate", "clip"}:
        q = np.clip(q, qmin_int, qmax_int)
    elif overflow in {"wrap", "ap_wrap"}:
        modulus = float(2 ** total_bits)
        q = np.mod(q - qmin_int, modulus) + qmin_int
    else:
        raise ValueError(f"Unsupported overflow mode: {overflow}")

    return (q / scale).astype(np.float32, copy=False)


def quantize_ap_fixed_array(x: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    """Emulate the default ``ap_fixed<W,I>`` cast (AP_TRN, AP_WRAP)."""
    return quantize_array(x, spec, rounding="ap_trn", overflow="ap_wrap")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return 0.0
    d = a - b
    return float(np.mean(d * d))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.abs(a - b)))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return 1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))