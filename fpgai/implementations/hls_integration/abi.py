from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from fpgai.implementations.implementation_contract import ImplementationContract
from .errors import HLSIntegrationIssue

@dataclass(frozen=True)
class HLSAttributeParameter:
    name: str
    cpp_type: str = "float"
    default: Any = 0.0

@dataclass(frozen=True)
class HLSTensorPort:
    name: str
    direction: str
    scalar_type: str | None = None
    shape: tuple[int, ...] = ()
    layout: str | None = None

@dataclass(frozen=True)
class HLSFlatArrayABI:
    abi: str
    scalar_type: str
    attributes: tuple[HLSAttributeParameter, ...]

@dataclass(frozen=True)
class HLSTensorPortsABI:
    abi: str
    scalar_type: str
    inputs: tuple[HLSTensorPort, ...]
    outputs: tuple[HLSTensorPort, ...]
    attributes: tuple[HLSAttributeParameter, ...]
    count_mode: str = "shared"

    def scalar_for(self, port: HLSTensorPort) -> str:
        return port.scalar_type or self.scalar_type


def _integration_mapping(contract: ImplementationContract) -> Mapping[str, Any]:
    value = contract.metadata.get("integration", {})
    return value if isinstance(value, Mapping) else {}


def _hls_mapping(contract: ImplementationContract) -> Mapping[str, Any]:
    integration = _integration_mapping(contract)
    hls = integration.get("hls", {})
    if not isinstance(hls, Mapping):
        raise ValueError("HLSINT003: integration.hls must be an object")
    return hls


def _attributes(hls: Mapping[str, Any]) -> tuple[HLSAttributeParameter, ...]:
    result: list[HLSAttributeParameter] = []
    for index, raw in enumerate(hls.get("attributes", []) or []):
        if not isinstance(raw, Mapping) or not raw.get("name"):
            raise ValueError(f"HLSINT006: invalid integration.hls.attributes[{index}]")
        cpp_type = str(raw.get("cpp_type", "float"))
        if cpp_type not in {"float", "double", "int", "unsigned"}:
            raise ValueError(f"HLSINT006: unsupported attribute C++ type {cpp_type}")
        result.append(HLSAttributeParameter(str(raw["name"]), cpp_type, raw.get("default", 0.0)))
    return tuple(result)


def _validate_scalar_type(value: str) -> str:
    if value not in {"float", "double", "int", "unsigned", "int16_t", "uint16_t", "int32_t", "uint32_t"}:
        raise ValueError(f"HLSINT005: unsupported scalar type {value!r}")
    return value


def _scalar_type(hls: Mapping[str, Any]) -> str:
    return _validate_scalar_type(str(hls.get("scalar_type", "float")))


def parse_flat_array_abi(contract: ImplementationContract) -> HLSFlatArrayABI:
    hls = _hls_mapping(contract)
    abi = str(hls.get("abi", ""))
    if abi != "flat_array_v1":
        raise ValueError("HLSINT004: expected flat_array_v1")
    return HLSFlatArrayABI(abi=abi, scalar_type=_scalar_type(hls), attributes=_attributes(hls))


def _shape(raw: Any, *, path: str) -> tuple[int, ...]:
    if raw in (None, []):
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"HLSINT017: {path}.shape must be a list")
    dims = tuple(int(x) for x in raw)
    if any(x <= 0 for x in dims):
        raise ValueError(f"HLSINT017: {path}.shape must contain positive static dimensions")
    return dims


def _ports(raw_ports: Any, direction: str, default_scalar: str) -> tuple[HLSTensorPort, ...]:
    if not isinstance(raw_ports, list) or not raw_ports:
        raise ValueError(f"HLSINT015: tensor_ports_v1 requires non-empty {direction} ports")
    result: list[HLSTensorPort] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_ports):
        if not isinstance(raw, Mapping) or not raw.get("name"):
            raise ValueError(f"HLSINT015: invalid {direction} port at index {index}")
        name = str(raw["name"])
        if name in seen:
            raise ValueError(f"HLSINT016: duplicate tensor port {name!r}")
        seen.add(name)
        scalar = _validate_scalar_type(str(raw.get("scalar_type", default_scalar)))
        layout = None if raw.get("layout") is None else str(raw.get("layout"))
        result.append(HLSTensorPort(name=name, direction=direction, scalar_type=scalar, shape=_shape(raw.get("shape"), path=f"{direction}[{index}]"), layout=layout))
    return tuple(result)


def parse_tensor_ports_abi(contract: ImplementationContract) -> HLSTensorPortsABI:
    hls = _hls_mapping(contract)
    abi = str(hls.get("abi", ""))
    if abi != "tensor_ports_v1":
        raise ValueError("HLSINT004: expected tensor_ports_v1")
    scalar = _scalar_type(hls)
    count_mode = str(hls.get("count_mode", "shared"))
    if count_mode not in {"shared", "per_port"}:
        raise ValueError("HLSINT018: tensor_ports_v1 count_mode must be shared or per_port")
    return HLSTensorPortsABI(
        abi=abi,
        scalar_type=scalar,
        inputs=_ports(hls.get("inputs"), "input", scalar),
        outputs=_ports(hls.get("outputs"), "output", scalar),
        attributes=_attributes(hls),
        count_mode=count_mode,
    )


def parse_hls_abi(contract: ImplementationContract) -> HLSFlatArrayABI | HLSTensorPortsABI:
    abi = str(_hls_mapping(contract).get("abi", ""))
    if abi == "flat_array_v1":
        return parse_flat_array_abi(contract)
    if abi == "tensor_ports_v1":
        return parse_tensor_ports_abi(contract)
    raise ValueError(f"HLSINT004: unsupported HLS ABI {abi!r}")


def validate_hls_integration_contract(contract: ImplementationContract) -> tuple[HLSIntegrationIssue, ...]:
    issues: list[HLSIntegrationIssue] = []
    if contract.language != "hls_cpp":
        issues.append(HLSIntegrationIssue("HLSINT001", "language", "External HLS integration requires hls_cpp"))
    if contract.backend != "vitis_hls":
        issues.append(HLSIntegrationIssue("HLSINT002", "backend", "External HLS integration requires the vitis_hls backend"))
    try:
        parse_hls_abi(contract)
    except ValueError as exc:
        code, _, message = str(exc).partition(": ")
        issues.append(HLSIntegrationIssue(code or "HLSINT003", "integration.hls", message or str(exc)))
    return tuple(issues)
