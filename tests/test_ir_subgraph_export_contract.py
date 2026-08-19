import numpy as np
import pytest

from fpgai.ir.graph import Graph


def _graph():
    g = Graph("demo")
    for name in ("x", "a", "b", "c"):
        g.add_tensor(name, (1, 4))
    g.add_tensor("w", (4, 4))
    g.inputs = ["x"]
    g.outputs = ["c"]
    g.constants["w"] = np.arange(16, dtype=np.float32).reshape(4, 4)
    dense = g.add_op("MatMul", ["x", "w"], ["a"], name="dense0")
    relu = g.add_op("Relu", ["a"], ["b"], name="relu0")
    out = g.add_op("Identity", ["b"], ["c"], name="out0")
    dense.semantics.execution = {"pipeline": {"ii": 2}}
    relu.semantics.provenance = {"source_op": "onnx::Relu"}
    return g


def test_ir_owned_subgraph_promotes_external_dependencies_and_preserves_semantics():
    g = _graph()
    sub = g.extract_subgraph(["relu0"])
    assert sub.inputs == ["a"]
    assert sub.outputs == ["b"]
    assert [op.name for op in sub.ops] == ["relu0"]
    assert sub.ops[0].semantics.provenance["source_op"] == "onnx::Relu"
    assert sub.metadata["subgraph_export"]["source_graph"] == "demo"


def test_ir_owned_subgraph_keeps_selected_constants_and_boundary_outputs():
    g = _graph()
    sub = g.extract_subgraph(["dense0", "relu0"])
    assert sub.inputs == ["x"]
    assert sub.outputs == ["b"]
    assert set(sub.constants) == {"w"}
    assert [op.name for op in sub.ops] == ["dense0", "relu0"]
    assert sub.ops[0].semantics.execution["pipeline"]["ii"] == 2


def test_ir_owned_subgraph_rejects_unknown_selection():
    with pytest.raises(KeyError, match="IRSUB002"):
        _graph().extract_subgraph(["missing"])
