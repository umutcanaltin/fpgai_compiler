from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class InterfaceRequirement:
    name: str
    direction: str
    protocol: str
    data_type: str | None = None
    data_width: int | None = None
    layout: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "name": self.name,
                "direction": self.direction,
                "protocol": self.protocol,
                "data_type": self.data_type,
                "data_width": self.data_width,
                "layout": self.layout,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class ImplementationMetrics:
    latency_cycles: int | None = None
    initiation_interval: int | None = None
    lut: int | None = None
    ff: int | None = None
    dsp: int | None = None
    bram18: int | None = None
    uram: int | None = None
    power_w: float | None = None

    def to_dict(self) -> dict[str, int | float]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))
