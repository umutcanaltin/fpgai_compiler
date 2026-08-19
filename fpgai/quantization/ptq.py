from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from fpgai.quantization.contracts import QuantizationParameters, QuantizationSpec
from fpgai.quantization.observers import MinMaxObserver, PercentileObserver, observe_many
from fpgai.quantization.parameters import derive_quantization_parameters


@dataclass(frozen=True)
class PTQCalibrationResult:
    method: str
    parameters: QuantizationParameters
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "sample_count": self.sample_count,
            "parameters": self.parameters.to_dict(),
        }


def calibrate_ptq(
    samples: Iterable[np.ndarray],
    spec: QuantizationSpec,
    *,
    method: str = "min_max",
    percentile: float = 99.99,
) -> PTQCalibrationResult:
    axis = spec.axis if spec.granularity == "per_channel" else None
    if method == "min_max":
        observer = MinMaxObserver(axis=axis)
    elif method == "percentile":
        observer = PercentileObserver(percentile=percentile, axis=axis)
    else:
        raise ValueError(f"unsupported PTQ calibration method: {method!r}")
    observe_many(observer, samples)
    minimum, maximum = observer.range()
    parameters = derive_quantization_parameters(minimum, maximum, spec)
    return PTQCalibrationResult(method=method, parameters=parameters, sample_count=observer.sample_count)


def _parameter_array(value: float | int | tuple[float, ...] | tuple[int, ...], *, axis: int | None, rank: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0 or axis is None:
        return array
    resolved = axis if axis >= 0 else rank + axis
    shape = [1] * rank
    shape[resolved] = array.size
    return array.reshape(shape)


def quantize(values: np.ndarray, parameters: QuantizationParameters) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    spec = parameters.spec
    scale = _parameter_array(parameters.scale, axis=spec.axis, rank=array.ndim)
    zero = _parameter_array(parameters.zero_point, axis=spec.axis, rank=array.ndim)
    transformed = array / scale + zero
    if spec.rounding == "nearest":
        rounded = np.rint(transformed)
    elif spec.rounding == "floor":
        rounded = np.floor(transformed)
    else:
        rounded = np.ceil(transformed)
    if spec.saturation == "saturate":
        rounded = np.clip(rounded, spec.qmin, spec.qmax)
    else:
        width = spec.qmax - spec.qmin + 1
        rounded = ((rounded - spec.qmin) % width) + spec.qmin
    return rounded.astype(np.int64)


def dequantize(values: np.ndarray, parameters: QuantizationParameters) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    spec = parameters.spec
    scale = _parameter_array(parameters.scale, axis=spec.axis, rank=array.ndim)
    zero = _parameter_array(parameters.zero_point, axis=spec.axis, rank=array.ndim)
    return ((array - zero) * scale).astype(np.float32)


def fake_quantize(values: np.ndarray, parameters: QuantizationParameters) -> np.ndarray:
    return dequantize(quantize(values, parameters), parameters)
