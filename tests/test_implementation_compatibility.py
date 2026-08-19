from fpgai.implementations import CompatibilityRequest, evaluate_implementation_compatibility, implementation_contract_from_manifest


def test_compatibility_reports_specific_rejection_reasons() -> None:
    hls = implementation_contract_from_manifest("examples/packages/scale_bias_hls")
    ok = evaluate_implementation_compatibility(
        hls,
        CompatibilityRequest(
            mode="training",
            backend="vitis_hls",
            board="kv260",
            toolchain_name="vitis_hls",
            toolchain_version="2023.2",
            precision="int16",
            input_protocol="axi_stream",
            output_protocol="axi_stream",
            weight_storage="bram",
        ),
    )
    assert ok.compatible
    rejected = evaluate_implementation_compatibility(
        hls,
        CompatibilityRequest(mode="inference", backend="vhdl", precision="fp64"),
    )
    assert rejected.compatible is False
    assert "backend_mismatch" in rejected.reasons
    assert "precision_not_supported" in rejected.reasons


def test_semantic_operator_version_is_part_of_implementation_compatibility():
    contract = implementation_contract_from_manifest("examples/packages/scale_bias_hls")
    from dataclasses import replace
    contract = replace(contract, semantics_version=2)
    result = evaluate_implementation_compatibility(
        contract,
        CompatibilityRequest(mode="inference", operator_semantics_version=1),
    )
    assert not result.compatible
    assert "operator_semantics_version_mismatch" in result.reasons
