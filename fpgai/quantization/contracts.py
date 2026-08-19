from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_SUPPORTED_SCHEMES = {"symmetric", "asymmetric"}
_SUPPORTED_GRANULARITIES = {"per_tensor", "per_channel"}
_SUPPORTED_ROUNDING = {"nearest", "floor", "ceil"}
_SUPPORTED_SATURATION = {"saturate", "wrap"}


@dataclass(frozen=True)
class QuantizationSpec:
    bits: int
    scheme: str = "symmetric"
    granularity: str = "per_tensor"
    signed: bool = True
    axis: int | None = None
    rounding: str = "nearest"
    saturation: str = "saturate"

    def __post_init__(self) -> None:
        if type(self.bits) is not int or self.bits < 2 or self.bits > 32:
            raise ValueError("quantization bits must be an integer in [2, 32]")
        if self.scheme not in _SUPPORTED_SCHEMES:
            raise ValueError(f"unsupported quantization scheme: {self.scheme!r}")
        if self.granularity not in _SUPPORTED_GRANULARITIES:
            raise ValueError(f"unsupported quantization granularity: {self.granularity!r}")
        if self.rounding not in _SUPPORTED_ROUNDING:
            raise ValueError(f"unsupported quantization rounding: {self.rounding!r}")
        if self.saturation not in _SUPPORTED_SATURATION:
            raise ValueError(f"unsupported quantization saturation: {self.saturation!r}")
        if self.granularity == "per_channel" and self.axis is None:
            raise ValueError("per_channel quantization requires axis")
        if self.granularity == "per_tensor" and self.axis is not None:
            raise ValueError("per_tensor quantization must not set axis")

    @property
    def qmin(self) -> int:
        if self.signed:
            return -(1 << (self.bits - 1))
        return 0

    @property
    def qmax(self) -> int:
        if self.signed:
            return (1 << (self.bits - 1)) - 1
        return (1 << self.bits) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "scheme": self.scheme,
            "granularity": self.granularity,
            "signed": self.signed,
            "axis": self.axis,
            "rounding": self.rounding,
            "saturation": self.saturation,
        }


@dataclass(frozen=True)
class QuantizationParameters:
    spec: QuantizationSpec
    scale: float | tuple[float, ...]
    zero_point: int | tuple[int, ...]
    observed_min: float | tuple[float, ...]
    observed_max: float | tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        def _jsonable(value: Any) -> Any:
            return list(value) if isinstance(value, tuple) else value

        return {
            "spec": self.spec.to_dict(),
            "scale": _jsonable(self.scale),
            "zero_point": _jsonable(self.zero_point),
            "observed_min": _jsonable(self.observed_min),
            "observed_max": _jsonable(self.observed_max),
        }


def quantization_spec_from_mapping(raw: Mapping[str, Any], *, path: str = "quantization") -> QuantizationSpec:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return QuantizationSpec(
        bits=int(raw.get("bits", 8)),
        scheme=str(raw.get("scheme", "symmetric")),
        granularity=str(raw.get("granularity", "per_tensor")),
        signed=bool(raw.get("signed", True)),
        axis=None if raw.get("axis") is None else int(raw["axis"]),
        rounding=str(raw.get("rounding", "nearest")),
        saturation=str(raw.get("saturation", "saturate")),
    )
