from pathlib import Path

from fpgai.implementations import implementation_contract_from_manifest


def test_hls_and_vhdl_examples_build_implementation_contracts() -> None:
    hls = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    vhdl = implementation_contract_from_manifest(Path("examples/packages/scale_bias_vhdl"))
    assert hls.operator_id == "community.operator.scale_bias"
    assert hls.language == "hls_cpp"
    assert hls.backend == "vitis_hls"
    assert hls.training.forward is True
    assert vhdl.language == "vhdl"
    assert vhdl.backend == "vhdl"
    assert vhdl.training.forward is False
    assert hls.to_dict()["usage"]["production_path"] == "morfics"


def test_implementation_contract_binds_semantic_operator_version_and_numeric_policy() -> None:
    hls = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    payload = hls.to_dict()
    assert hls.semantics_version == 1
    assert payload["contribution_role"] == "operator_implementation"
    assert payload["implements"] == {"operator_id": "community.operator.scale_bias", "version": 1}
    assert hls.metadata["validation"]["numeric"]["required"] is True
