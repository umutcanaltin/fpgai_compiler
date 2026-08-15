from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from fpgai.implementations.implementation_contract import ImplementationContract


@dataclass(frozen=True)
class VHDLScalarStreamABI:
    abi: str = "scalar_stream_v1"
    data_width: int = 16
    signed: bool = True
    clock: str = "clk"
    reset_n: str = "rst_n"
    input_valid: str = "input_valid"
    input_data: str = "input_data"
    output_valid: str = "output_valid"
    output_data: str = "output_data"
    input_ready: str = "input_ready"
    output_ready: str = "output_ready"
    reference_behavior: str = "identity"


@dataclass(frozen=True)
class VHDLTensorPort:
    name: str
    data: str
    data_width: int
    signed: bool = True


@dataclass(frozen=True)
class VHDLTensorPortsReadyValidABI:
    abi: str
    inputs: tuple[VHDLTensorPort, ...]
    outputs: tuple[VHDLTensorPort, ...]
    clock: str = "clk"
    reset_n: str = "rst_n"
    input_valid: str = "input_valid"
    input_ready: str = "input_ready"
    output_valid: str = "output_valid"
    output_ready: str = "output_ready"
    handshake_policy: str = "grouped_transaction"

    @property
    def data_widths(self) -> tuple[int, ...]:
        return tuple(port.data_width for port in (*self.inputs, *self.outputs))


def _integration_mapping(contract: ImplementationContract) -> Mapping[str, Any]:
    integration = contract.metadata.get("integration", {})
    return integration if isinstance(integration, Mapping) else {}


def _vhdl_mapping(contract: ImplementationContract) -> Mapping[str, Any]:
    raw = _integration_mapping(contract).get("vhdl", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("VHDLINT003: integration.vhdl must be an object")
    return raw


def _positive_width(value: Any, *, path: str) -> int:
    width = int(value)
    if width <= 0:
        raise ValueError(f"VHDLINT005: {path} must be positive")
    return width


def parse_vhdl_scalar_stream_abi(contract: ImplementationContract) -> VHDLScalarStreamABI:
    raw = _vhdl_mapping(contract)
    abi = str(raw.get("abi", "scalar_stream_v1"))
    if abi not in {"scalar_stream_v1", "scalar_ready_valid_v1"}:
        raise ValueError(f"VHDLINT004: expected scalar VHDL ABI, got {abi!r}")

    width = _positive_width(raw.get("data_width", 16), path="data_width")
    behavior = str(raw.get("reference_behavior", "identity"))
    if behavior not in {"identity"}:
        raise ValueError(f"VHDLINT010: unsupported reference_behavior {behavior!r}")

    defaults = VHDLScalarStreamABI()
    fields = (
        "clock",
        "reset_n",
        "input_valid",
        "input_data",
        "output_valid",
        "output_data",
    )
    if abi == "scalar_ready_valid_v1":
        fields += ("input_ready", "output_ready")
    values = {name: str(raw.get(name, getattr(defaults, name))) for name in fields}
    return VHDLScalarStreamABI(
        abi=abi,
        data_width=width,
        signed=bool(raw.get("signed", True)),
        reference_behavior=behavior,
        **values,
    )


def _tensor_ports(
    raw_ports: Any,
    *,
    direction: str,
    default_width: int,
    default_signed: bool,
) -> tuple[VHDLTensorPort, ...]:
    if not isinstance(raw_ports, list) or not raw_ports:
        raise ValueError(f"VHDLINT012: tensor_ports_ready_valid_v1 requires non-empty {direction}s")

    result: list[VHDLTensorPort] = []
    logical_names: set[str] = set()
    rtl_names: set[str] = set()
    for index, raw in enumerate(raw_ports):
        if not isinstance(raw, Mapping):
            raise ValueError(f"VHDLINT013: integration.vhdl.{direction}s[{index}] must be an object")
        name = str(raw.get("name", "")).strip()
        data = str(raw.get("data", "")).strip()
        if not name or not data:
            raise ValueError(
                f"VHDLINT013: integration.vhdl.{direction}s[{index}] requires name and data"
            )
        if name in logical_names:
            raise ValueError(f"VHDLINT014: duplicate logical VHDL tensor port {name!r}")
        if data in rtl_names:
            raise ValueError(f"VHDLINT015: duplicate VHDL RTL data port {data!r}")
        logical_names.add(name)
        rtl_names.add(data)
        result.append(
            VHDLTensorPort(
                name=name,
                data=data,
                data_width=_positive_width(raw.get("data_width", default_width), path=f"{direction}s[{index}].data_width"),
                signed=bool(raw.get("signed", default_signed)),
            )
        )
    return tuple(result)


def parse_vhdl_tensor_ports_ready_valid_abi(contract: ImplementationContract) -> VHDLTensorPortsReadyValidABI:
    raw = _vhdl_mapping(contract)
    abi = str(raw.get("abi", ""))
    if abi != "tensor_ports_ready_valid_v1":
        raise ValueError(f"VHDLINT004: expected tensor_ports_ready_valid_v1, got {abi!r}")

    handshake_policy = str(raw.get("handshake_policy", "grouped_transaction"))
    if handshake_policy != "grouped_transaction":
        raise ValueError(
            "VHDLINT016: tensor_ports_ready_valid_v1 currently supports handshake_policy=grouped_transaction"
        )

    default_width = _positive_width(raw.get("data_width", 16), path="data_width")
    default_signed = bool(raw.get("signed", True))
    return VHDLTensorPortsReadyValidABI(
        abi=abi,
        inputs=_tensor_ports(
            raw.get("inputs"),
            direction="input",
            default_width=default_width,
            default_signed=default_signed,
        ),
        outputs=_tensor_ports(
            raw.get("outputs"),
            direction="output",
            default_width=default_width,
            default_signed=default_signed,
        ),
        clock=str(raw.get("clock", "clk")),
        reset_n=str(raw.get("reset_n", "rst_n")),
        input_valid=str(raw.get("input_valid", "input_valid")),
        input_ready=str(raw.get("input_ready", "input_ready")),
        output_valid=str(raw.get("output_valid", "output_valid")),
        output_ready=str(raw.get("output_ready", "output_ready")),
        handshake_policy=handshake_policy,
    )


def parse_vhdl_abi(contract: ImplementationContract) -> VHDLScalarStreamABI | VHDLTensorPortsReadyValidABI:
    abi = str(_vhdl_mapping(contract).get("abi", "scalar_stream_v1"))
    if abi in {"scalar_stream_v1", "scalar_ready_valid_v1"}:
        return parse_vhdl_scalar_stream_abi(contract)
    if abi == "tensor_ports_ready_valid_v1":
        return parse_vhdl_tensor_ports_ready_valid_abi(contract)
    raise ValueError(f"VHDLINT004: unsupported VHDL ABI {abi!r}")
