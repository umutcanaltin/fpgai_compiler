from pathlib import Path

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.hls_integration import parse_flat_array_abi, validate_hls_integration_contract


def test_scale_bias_hls_declares_supported_flat_array_abi() -> None:
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    abi = parse_flat_array_abi(contract)
    assert abi.abi == "flat_array_v1"
    assert abi.scalar_type == "float"
    assert [item.name for item in abi.attributes] == ["scale", "bias"]
    assert validate_hls_integration_contract(contract) == ()
