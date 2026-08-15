from __future__ import annotations

import numpy as np

from fpgai.ir.graph import Graph
from fpgai.operators.external import ReferenceExecutionResult
from fpgai.validation.mixed_external_hls import execute_mixed_graph_reference


class _Context:
    def reference_for(self, operator_id: str):
        assert operator_id == "community.operator.split_scale"
        def callback(ctx):
            x = np.asarray(ctx.inputs[0], dtype=np.float32)
            return ReferenceExecutionResult((x.copy(), x * float(ctx.attributes["scale"])))
        return callback


def test_external_reference_supports_multiple_outputs_and_normalizes_shapes():
    graph = Graph("multi_output_reference")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    for name in ("input", "identity", "scaled", "summed", "output"):
        graph.add_tensor(name, (1, 4), "float32")
    graph.add_op("SplitScale", ["input"], ["identity", "scaled"], "split_scale_0", {
        "scale": 2.0,
        "_fpgai_external_operator": {"operator_id": "community.operator.split_scale"},
    })
    graph.add_op("Add", ["identity", "scaled"], ["summed"], "add_0")
    graph.add_op("Relu", ["summed"], ["output"], "relu_0")
    result = execute_mixed_graph_reference(graph, _Context(), np.asarray([-2.0, -0.5, 0.5, 2.0], dtype=np.float32))
    np.testing.assert_allclose(result, [0.0, 0.0, 1.5, 6.0])
