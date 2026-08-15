from pathlib import Path

from fpgai.ir.graph import Graph
from fpgai.implementations import build_hls_composition_plan, implementation_contract_from_manifest


def _graph():
    graph = Graph("mixed")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    for name in ("input", "relu", "scaled", "output"):
        graph.add_tensor(name, (1, 4), "float32")
    graph.add_op("Relu", ["input"], ["relu"], "relu_0")
    graph.add_op("ScaleBias", ["relu"], ["scaled"], "scale_bias_0", {
        "scale": 2.0,
        "bias": 1.0,
        "_fpgai_external_operator": {
            "operator_id": "community.operator.scale_bias",
            "package_id": "community.scale_bias_operator",
            "package_version": "1.0.0",
            "manifest_sha256": "sha256:test",
        },
    })
    graph.add_op("Sigmoid", ["scaled"], ["output"], "sigmoid_0")
    return graph


def test_builds_per_node_mixed_composition_plan():
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    plan = build_hls_composition_plan(_graph(), selected_contracts={"scale_bias_0": contract})
    assert len(plan.bindings) == 1
    binding = plan.bindings[0]
    assert binding.node_name == "scale_bias_0"
    assert binding.input_words == binding.output_words == 4
    assert plan.used_package_ids == ("community.scale_bias_operator", "community.scale_bias_hls")


def test_branch_graph_allows_external_node_to_consume_noncurrent_tensor():
    graph = Graph("mixed_residual")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    for name in ("input", "relu", "scaled", "output"):
        graph.add_tensor(name, (1, 4), "float32")
    graph.add_op("Relu", ["input"], ["relu"], "relu_0")
    graph.add_op("ScaleBias", ["input"], ["scaled"], "scale_bias_0", {
        "scale": 2.0,
        "bias": 1.0,
        "_fpgai_external_operator": {
            "operator_id": "community.operator.scale_bias",
            "package_id": "community.scale_bias_operator",
            "package_version": "1.0.0",
            "manifest_sha256": "sha256:test",
        },
    })
    graph.add_op("Add", ["relu", "scaled"], ["output"], "add_0")
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    plan = build_hls_composition_plan(graph, selected_contracts={"scale_bias_0": contract})
    assert plan.graph_mode == "dag_mixed_graph"
    assert plan.bindings[0].input_tensor == "input"
