from __future__ import annotations

import numpy as np

from fpgai.ir import Graph
from fpgai.ir.passes import infer_shapes, plan_attention_lowering


def _graph() -> Graph:
    g = Graph("attention_core")
    g.inputs = ["q", "k", "v"]
    g.outputs = ["context"]
    g.add_tensor("q", (1, 4, 8))
    g.add_tensor("k", (1, 4, 8))
    g.add_tensor("v", (1, 4, 8))
    g.add_tensor("scale", ())
    g.constants["scale"] = np.asarray(0.35355339, dtype=np.float32)
    g.add_op("Transpose", ["k"], ["kt"], name="k_t", attrs={"perm": [0, 2, 1]})
    g.add_op("MatMul", ["q", "kt"], ["scores"], name="scores")
    g.add_op("Mul", ["scores", "scale"], ["scaled"], name="scale_scores")
    g.add_op("Softmax", ["scaled"], ["prob"], name="prob", attrs={"axis": -1})
    g.add_op("MatMul", ["prob", "v"], ["context"], name="context")
    return infer_shapes(g)


def test_attention_lowering_attaches_tiling_buffer_and_backend_candidates() -> None:
    graph = _graph()
    plans = plan_attention_lowering(graph, tile_m=4, tile_n=4, tile_k=8, score_buffer_storage="bram")
    assert len(plans) == 1
    plan = plans[0]
    assert plan.sequence_length == 4
    assert plan.head_dimension == 8
    assert plan.score_matmul == "scores"
    assert plan.softmax == "prob"
    assert plan.value_matmul == "context"
    softmax = next(op for op in graph.ops if op.name == "prob")
    assert softmax.semantics.schedule["attention_role"] == "attention_softmax"
    assert softmax.semantics.schedule["tile_k"] == 8
    assert {x.backend for x in softmax.semantics.implementation_candidates} == {"hls", "vhdl"}
    assert graph.tensors["prob"].semantics.memory.storage == "bram"
    assert graph.tensors["prob"].semantics.memory.residency == "on_chip"
    assert graph.metadata["attention_lowering_plans"][0]["schema"] == "fpgai.attention-lowering-plan/v1"
