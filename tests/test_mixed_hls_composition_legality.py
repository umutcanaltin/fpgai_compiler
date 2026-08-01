from pathlib import Path
import pytest

from fpgai.ir.graph import Graph
from fpgai.implementations import HLSCompositionError, build_hls_composition_plan, implementation_contract_from_manifest


def test_rejects_external_node_that_does_not_consume_current_tensor():
    graph = Graph("branch")
    graph.inputs = ["input"]
    graph.outputs = ["scaled"]
    for name in ("input", "relu", "scaled"):
        graph.add_tensor(name, (1, 4), "float32")
    graph.add_op("Relu", ["input"], ["relu"], "relu_0")
    graph.add_op("ScaleBias", ["input"], ["scaled"], "scale_bias_0", {
        "_fpgai_external_operator": {"operator_id": "community.operator.scale_bias"},
    })
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    with pytest.raises(HLSCompositionError, match="HLSCOMP003"):
        build_hls_composition_plan(graph, selected_contracts={"scale_bias_0": contract})
