from __future__ import annotations

import numpy as np

from fpgai.quantization.contracts import QuantizationParameters, QuantizationSpec


def _as_tuple_or_scalar(values: np.ndarray | float, *, integer: bool = False):
    array = np.asarray(values)
    if array.ndim == 0:
        return int(array) if integer else float(array)
    if integer:
        return tuple(int(value) for value in array.reshape(-1))
    return tuple(float(value) for value in array.reshape(-1))


def derive_quantization_parameters(
    minimum: np.ndarray | float,
    maximum: np.ndarray | float,
    spec: QuantizationSpec,
) -> QuantizationParameters:
    lo = np.asarray(minimum, dtype=np.float64)
    hi = np.asarray(maximum, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError("minimum and maximum calibration shapes differ")
    if np.any(lo > hi):
        raise ValueError("minimum exceeds maximum in calibration range")

    qmin = spec.qmin
    qmax = spec.qmax

    if spec.scheme == "symmetric":
        bound = np.maximum(np.abs(lo), np.abs(hi))
        positive_qmax = max(abs(qmin), abs(qmax))
        scale = np.where(bound > 0.0, bound / float(positive_qmax), 1.0)
        zero = np.zeros_like(scale, dtype=np.int64)
    else:
        span = hi - lo
        scale = np.where(span > 0.0, span / float(qmax - qmin), 1.0)
        zero_float = qmin - lo / scale
        zero = np.clip(np.rint(zero_float), qmin, qmax).astype(np.int64)

    return QuantizationParameters(
        spec=spec,
        scale=_as_tuple_or_scalar(scale),
        zero_point=_as_tuple_or_scalar(zero, integer=True),
        observed_min=_as_tuple_or_scalar(lo),
        observed_max=_as_tuple_or_scalar(hi),
    )
