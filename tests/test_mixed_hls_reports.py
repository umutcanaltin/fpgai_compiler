import json
from pathlib import Path

from fpgai.implementations import build_hls_composition_plan, implementation_contract_from_manifest, write_composition_report
from fpgai.ir.graph import Graph


def test_composition_report_records_node_provenance(tmp_path):
    graph = Graph("external")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_tensor("x", (1, 4), "float32")
    graph.add_tensor("y", (1, 4), "float32")
    graph.add_op("ScaleBias", ["x"], ["y"], "scale_bias_0", {
        "_fpgai_external_operator": {"operator_id": "community.operator.scale_bias", "package_id": "community.scale_bias_operator"},
    })
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    plan = build_hls_composition_plan(graph, selected_contracts={"scale_bias_0": contract})
    json_path, md_path = write_composition_report(plan, tmp_path)
    payload = json.loads(json_path.read_text())
    assert payload["nodes"][0]["implementation"]["package_id"] == "community.scale_bias_hls"
    assert md_path.is_file()
