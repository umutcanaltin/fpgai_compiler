from pathlib import Path

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.hls_integration import HLSTensorPortsABI, parse_hls_abi
import importlib.util


def test_split_scale_operator_package_declares_two_outputs():
    module_path = Path("examples/packages/split_scale_operator/python/operator.py")
    spec = importlib.util.spec_from_file_location("split_scale_operator_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    definition = module.define_operator()
    assert definition.contract.operator_id == "community.operator.split_scale"
    assert [port.name for port in definition.contract.outputs] == ["identity", "scaled"]


def test_split_scale_hls_package_declares_tensor_ports_multi_output_abi():
    contract = implementation_contract_from_manifest(Path("examples/packages/split_scale_tensor_ports_hls"))
    abi = parse_hls_abi(contract)
    assert isinstance(abi, HLSTensorPortsABI)
    assert [port.name for port in abi.inputs] == ["input"]
    assert [port.name for port in abi.outputs] == ["identity", "scaled"]
