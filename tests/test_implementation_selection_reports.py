import json

from fpgai.implementations import CompatibilityRequest, ImplementationSelectionRequest, implementation_contract_from_manifest, select_implementation
from fpgai.implementations.selection_reports import write_implementation_selection_report


def test_selection_report_is_deterministic_and_research_scoped(tmp_path) -> None:
    result = select_implementation(
        [implementation_contract_from_manifest("examples/packages/scale_bias_hls")],
        ImplementationSelectionRequest(
            operator_id="community.operator.scale_bias",
            compatibility=CompatibilityRequest(mode="inference", backend="vitis_hls"),
        ),
    )
    json_path, md_path = write_implementation_selection_report(result, tmp_path)
    payload = json.loads(json_path.read_text())
    assert payload["selected"]["package_id"] == "community.scale_bias_hls"
    assert payload["usage"] == {"platform_scope": "research", "production_path": "morfics"}
    assert "community.scale_bias_hls" in md_path.read_text()
