from pathlib import Path

from fpgai.implementations import build_hls_composition_plan, implementation_contract_from_manifest
from fpgai.implementations.hls_composition import stage_external_sources
from fpgai.ir.graph import Graph


def test_stages_one_package_once_for_multiple_nodes(tmp_path):
    graph = Graph("two_external")
    graph.inputs = ["x"]
    graph.outputs = ["z"]
    for name in ("x", "y", "z"):
        graph.add_tensor(name, (1, 4), "float32")
    provenance = {"operator_id": "community.operator.scale_bias", "package_id": "community.scale_bias_operator"}
    graph.add_op("ScaleBias", ["x"], ["y"], "a", {"_fpgai_external_operator": provenance})
    graph.add_op("ScaleBias", ["y"], ["z"], "b", {"_fpgai_external_operator": provenance})
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    plan = build_hls_composition_plan(graph, selected_contracts={"a": contract, "b": contract})
    staged = stage_external_sources(plan, tmp_path / "hls")
    assert len(staged.sources) == 1
    assert len(staged.headers) == 1
