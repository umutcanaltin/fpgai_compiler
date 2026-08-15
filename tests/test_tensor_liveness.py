from fpgai.ir.graph import Graph
from fpgai.ir.liveness import analyze_tensor_liveness


def test_tensor_liveness_detects_sequential_reuse():
    graph = Graph("seq")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    for name in ("input", "a", "out"):
        graph.add_tensor(name, (4,), "float32")
    graph.add_op("Relu", ["input"], ["a"], name="relu0")
    graph.add_op("Sigmoid", ["a"], ["out"], name="sigmoid0")

    report = analyze_tensor_liveness(graph)

    assert report["has_branching"] is False
    assert report["sequential_current_buffer_compatible"] is True
    assert report["tensors"]["a"]["producer"] == "relu0"
    assert report["tensors"]["a"]["consumers"] == ["sigmoid0"]
    assert report["activation_buffer_slots"] >= 1


def test_tensor_liveness_detects_branch_and_merge():
    graph = Graph("residual")
    graph.inputs = ["input"]
    graph.outputs = ["out"]
    for name in ("input", "a", "b", "out"):
        graph.add_tensor(name, (4,), "float32")
    graph.add_op("Relu", ["input"], ["a"], name="relu0")
    graph.add_op("Sigmoid", ["input"], ["b"], name="sigmoid0")
    graph.add_op("Add", ["a", "b"], ["out"], name="add0")

    report = analyze_tensor_liveness(graph)

    assert report["has_branching"] is True
    assert "input" in report["branch_tensors"]
    assert "add0" in report["merge_ops"]
    assert report["sequential_current_buffer_compatible"] is False
    assert report["maximum_simultaneously_live_tensors"] >= 2
