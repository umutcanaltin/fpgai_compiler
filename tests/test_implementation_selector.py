from fpgai.implementations import (
    CompatibilityRequest,
    ImplementationSelectionRequest,
    implementation_contract_from_manifest,
    select_implementation,
)


def _contracts():
    return (
        implementation_contract_from_manifest("examples/packages/scale_bias_hls"),
        implementation_contract_from_manifest("examples/packages/scale_bias_vhdl"),
    )


def test_explicit_compatible_preference_wins() -> None:
    result = select_implementation(
        _contracts(),
        ImplementationSelectionRequest(
            operator_id="community.operator.scale_bias",
            compatibility=CompatibilityRequest(mode="inference", board="kv260", precision="int16"),
            preferred_packages=("community.scale_bias_vhdl",),
            policy="balanced",
        ),
    )
    assert result.ok
    assert result.selected is not None
    assert result.selected.package_id == "community.scale_bias_vhdl"
    selected = next(item for item in result.candidates if item.status == "selected")
    assert selected.reasons == ("explicitly_preferred_and_compatible",)


def test_backend_filter_rejects_vhdl_and_selects_hls() -> None:
    result = select_implementation(
        _contracts(),
        ImplementationSelectionRequest(
            operator_id="community.operator.scale_bias",
            compatibility=CompatibilityRequest(mode="training", backend="vitis_hls", precision="fp32"),
            policy="throughput",
        ),
    )
    assert result.ok
    assert result.selected is not None
    assert result.selected.package_id == "community.scale_bias_hls"
    rejected = next(item for item in result.candidates if item.contract.package_id == "community.scale_bias_vhdl")
    assert "training_forward_not_supported" in rejected.reasons
    assert "backend_mismatch" in rejected.reasons
