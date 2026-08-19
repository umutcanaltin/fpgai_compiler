from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from fpgai.quantization.contracts import QuantizationParameters, QuantizationSpec


@dataclass(frozen=True)
class RequantizationContract:
    source: QuantizationParameters
    destination: QuantizationParameters
    multiplier: int
    shift: int
    scale_ratio: float
    approximated_scale_ratio: float
    scale_error: float
    lossy: bool

    def __post_init__(self) -> None:
        if self.source.spec.granularity != "per_tensor" or self.destination.spec.granularity != "per_tensor":
            raise ValueError("requantization hardware currently requires per_tensor quantization")
        if isinstance(self.source.scale, tuple) or isinstance(self.destination.scale, tuple):
            raise ValueError("requantization hardware requires scalar source and destination scales")
        if isinstance(self.source.zero_point, tuple) or isinstance(self.destination.zero_point, tuple):
            raise ValueError("requantization hardware requires scalar source and destination zero-points")
        if self.shift < 0:
            raise ValueError("requantization shift must be non-negative")
        if self.multiplier <= 0:
            raise ValueError("requantization multiplier must be positive")

    @property
    def source_bits(self) -> int:
        return self.source.spec.bits

    @property
    def destination_bits(self) -> int:
        return self.destination.spec.bits

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpgai.requantization-contract/v1",
            "source": self.source.to_dict(),
            "destination": self.destination.to_dict(),
            "multiplier": self.multiplier,
            "shift": self.shift,
            "scale_ratio": self.scale_ratio,
            "approximated_scale_ratio": self.approximated_scale_ratio,
            "scale_error": self.scale_error,
            "lossy": self.lossy,
        }


def _scalar(value: float | int | tuple[float, ...] | tuple[int, ...], *, name: str) -> float:
    if isinstance(value, tuple):
        raise ValueError(f"{name} must be scalar for hardware requantization")
    return float(value)


def derive_requantization_contract(
    source: QuantizationParameters,
    destination: QuantizationParameters,
    *,
    max_shift: int = 30,
    max_multiplier_bits: int = 31,
) -> RequantizationContract:
    if source.spec.granularity != "per_tensor" or destination.spec.granularity != "per_tensor":
        raise ValueError("hardware requantization currently supports per_tensor quantization only")
    source_scale = _scalar(source.scale, name="source scale")
    destination_scale = _scalar(destination.scale, name="destination scale")
    if not math.isfinite(source_scale) or source_scale <= 0.0:
        raise ValueError("source quantization scale must be finite and positive")
    if not math.isfinite(destination_scale) or destination_scale <= 0.0:
        raise ValueError("destination quantization scale must be finite and positive")
    if max_shift < 0 or max_shift > 62:
        raise ValueError("max_shift must be in [0, 62]")
    if max_multiplier_bits < 2 or max_multiplier_bits > 62:
        raise ValueError("max_multiplier_bits must be in [2, 62]")

    ratio = source_scale / destination_scale
    maximum_multiplier = (1 << max_multiplier_bits) - 1

    selected_multiplier = 0
    selected_shift = 0
    for shift in range(max_shift, -1, -1):
        multiplier = int(round(ratio * float(1 << shift)))
        if 0 < multiplier <= maximum_multiplier:
            selected_multiplier = multiplier
            selected_shift = shift
            break
    if selected_multiplier <= 0:
        raise ValueError("requantization scale ratio cannot be represented by the configured multiplier/shift limits")

    approximated = selected_multiplier / float(1 << selected_shift)
    scale_error = abs(approximated - ratio)

    src_qmin, src_qmax = source.spec.qmin, source.spec.qmax
    src_zero = int(_scalar(source.zero_point, name="source zero-point"))
    dst_zero = int(_scalar(destination.zero_point, name="destination zero-point"))
    mapped_min = (src_qmin - src_zero) * ratio + dst_zero
    mapped_max = (src_qmax - src_zero) * ratio + dst_zero
    lo, hi = sorted((mapped_min, mapped_max))
    destination_covers = lo >= destination.spec.qmin and hi <= destination.spec.qmax
    exact_ratio = scale_error <= max(1e-12, abs(ratio) * 1e-12)
    integer_map = abs(ratio - round(ratio)) <= 1e-12
    lossy = not (destination_covers and exact_ratio and integer_map)

    return RequantizationContract(
        source=source,
        destination=destination,
        multiplier=selected_multiplier,
        shift=selected_shift,
        scale_ratio=ratio,
        approximated_scale_ratio=approximated,
        scale_error=scale_error,
        lossy=lossy,
    )


def _round_shift_scalar(value: int, *, shift: int, rounding: str) -> int:
    if shift == 0:
        return int(value)
    divisor = 1 << shift
    if rounding == "floor":
        return math.floor(value / divisor)
    if rounding == "ceil":
        return math.ceil(value / divisor)
    if rounding == "nearest":
        if value >= 0:
            return (value + (divisor // 2)) // divisor
        return -(((-value) + (divisor // 2)) // divisor)
    raise ValueError(f"unsupported requantization rounding: {rounding!r}")


def requantize_integer(values: np.ndarray, contract: RequantizationContract) -> np.ndarray:
    source_zero = int(_scalar(contract.source.zero_point, name="source zero-point"))
    destination_zero = int(_scalar(contract.destination.zero_point, name="destination zero-point"))
    source_values = np.asarray(values, dtype=np.int64)
    centered = source_values.astype(object) - source_zero

    flat = []
    for value in centered.reshape(-1):
        product = int(value) * contract.multiplier
        shifted = _round_shift_scalar(
            product,
            shift=contract.shift,
            rounding=contract.destination.spec.rounding,
        )
        flat.append(shifted + destination_zero)
    result = np.asarray(flat, dtype=np.int64).reshape(source_values.shape)

    spec = contract.destination.spec
    if spec.saturation == "saturate":
        return np.clip(result, spec.qmin, spec.qmax).astype(np.int64)

    width = spec.qmax - spec.qmin + 1
    return (((result - spec.qmin) % width) + spec.qmin).astype(np.int64)


def quantization_parameters_from_tensor(tensor: Any) -> QuantizationParameters:
    raw = getattr(tensor, "quantization", None)
    if not isinstance(raw, dict):
        raise ValueError(f"tensor {getattr(tensor, 'name', '<unknown>')!r} has no quantization parameters")
    spec_raw = raw.get("spec")
    if not isinstance(spec_raw, dict):
        raise ValueError("tensor quantization metadata is missing spec")
    spec = QuantizationSpec(
        bits=int(spec_raw["bits"]),
        scheme=str(spec_raw.get("scheme", "symmetric")),
        granularity=str(spec_raw.get("granularity", "per_tensor")),
        signed=bool(spec_raw.get("signed", True)),
        axis=None if spec_raw.get("axis") is None else int(spec_raw["axis"]),
        rounding=str(spec_raw.get("rounding", "nearest")),
        saturation=str(spec_raw.get("saturation", "saturate")),
    )

    def value(name: str):
        raw_value = raw[name]
        if isinstance(raw_value, list):
            return tuple(raw_value)
        return raw_value

    return QuantizationParameters(
        spec=spec,
        scale=value("scale"),
        zero_point=value("zero_point"),
        observed_min=value("observed_min"),
        observed_max=value("observed_max"),
    )
