from pathlib import Path

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.vhdl_integration import (
    VHDLTensorPortsReadyValidABI,
    parse_vhdl_abi,
    parse_vhdl_tensor_ports_ready_valid_abi,
    validate_vhdl_integration_contract,
)


def test_split_package_declares_grouped_multi_output_abi() -> None:
    contract = implementation_contract_from_manifest(Path("examples/packages/split_grouped_ready_valid_vhdl"))
    abi = parse_vhdl_tensor_ports_ready_valid_abi(contract)

    assert isinstance(abi, VHDLTensorPortsReadyValidABI)
    assert abi.handshake_policy == "grouped_transaction"
    assert [port.name for port in abi.inputs] == ["input"]
    assert [port.name for port in abi.outputs] == ["left", "right"]
    assert [port.data for port in abi.outputs] == ["left_data", "right_data"]
    assert abi.data_widths == (16, 16, 16)
    assert not validate_vhdl_integration_contract(contract)


def test_add_package_declares_grouped_multi_input_abi() -> None:
    contract = implementation_contract_from_manifest(Path("examples/packages/add_grouped_ready_valid_vhdl"))
    abi = parse_vhdl_abi(contract)

    assert isinstance(abi, VHDLTensorPortsReadyValidABI)
    assert [port.name for port in abi.inputs] == ["left", "right"]
    assert [port.name for port in abi.outputs] == ["output"]
    assert [port.data for port in abi.inputs] == ["left_data", "right_data"]
