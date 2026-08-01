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
class HLSFlatArrayABI:
    abi: str
    scalar_type: str
    attributes: tuple[HLSAttributeParameter, ...]


def _integration_mapping(contract: ImplementationContract) -> Mapping[str, Any]:
    value = contract.metadata.get("integration", {})
    return value if isinstance(value, Mapping) else {}


def parse_flat_array_abi(contract: ImplementationContract) -> HLSFlatArrayABI:
    integration = _integration_mapping(contract)
    hls = integration.get("hls", {})
    if not isinstance(hls, Mapping):
        raise ValueError("HLSINT003: integration.hls must be an object")
    abi = str(hls.get("abi", ""))
    if abi != "flat_array_v1":
        raise ValueError("HLSINT004: only flat_array_v1 is supported in E4A")
    scalar_type = str(hls.get("scalar_type", "float"))
    if scalar_type not in {"float", "double"}:
        raise ValueError("HLSINT005: unsupported scalar type")
    attributes: list[HLSAttributeParameter] = []
    for index, raw in enumerate(hls.get("attributes", []) or []):
        if not isinstance(raw, Mapping) or not raw.get("name"):
            raise ValueError(f"HLSINT006: invalid integration.hls.attributes[{index}]")
        cpp_type = str(raw.get("cpp_type", "float"))
        if cpp_type not in {"float", "double", "int", "unsigned"}:
            raise ValueError(f"HLSINT006: unsupported attribute C++ type {cpp_type}")
        attributes.append(HLSAttributeParameter(str(raw["name"]), cpp_type, raw.get("default", 0.0)))
    return HLSFlatArrayABI(abi=abi, scalar_type=scalar_type, attributes=tuple(attributes))


def validate_hls_integration_contract(contract: ImplementationContract) -> tuple[HLSIntegrationIssue, ...]:
    issues: list[HLSIntegrationIssue] = []
    if contract.language != "hls_cpp":
        issues.append(HLSIntegrationIssue("HLSINT001", "language", "External HLS integration requires hls_cpp"))
    if contract.backend != "vitis_hls":
        issues.append(HLSIntegrationIssue("HLSINT002", "backend", "E4A supports the vitis_hls backend"))
    try:
        parse_flat_array_abi(contract)
    except ValueError as exc:
        code, _, message = str(exc).partition(": ")
        issues.append(HLSIntegrationIssue(code or "HLSINT003", "integration.hls", message or str(exc)))
    return tuple(issues)
