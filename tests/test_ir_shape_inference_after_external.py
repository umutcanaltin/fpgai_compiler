from types import SimpleNamespace

from fpgai.ir.graph import Graph
from fpgai.ir.passes.infer_shapes import infer_shapes


def test_add_shape_propagates_after_external_branch_output_is_known():
    g = Graph("branch")
    g.inputs = ["input"]
    g.outputs = ["output"]
    g.add_tensor("input", (1, 4), "float32")
    g.add_tensor("relu_out", (1, 4), "float32")
    # Represents shape supplied by approved external-operator callback.
    g.add_tensor("scaled", (1, 4), "float32")
    g.ops = [
        SimpleNamespace(op_type="Add", name="add_0", inputs=["relu_out", "scaled"], outputs=["summed"], attrs={}),
        SimpleNamespace(op_type="Sigmoid", name="sigmoid_0", inputs=["summed"], outputs=["output"], attrs={}),
    ]

    infer_shapes(g)

    assert g.get_tensor("summed").shape == (1, 4)
    assert g.get_tensor("output").shape == (1, 4)


def test_add_shape_uses_onnx_broadcasting_rules():
    g = Graph("broadcast")
    g.add_tensor("lhs", (2, 4), "float32")
    g.add_tensor("rhs", (4,), "float32")
    g.ops = [
        SimpleNamespace(op_type="Add", name="add_0", inputs=["lhs", "rhs"], outputs=["out"], attrs={}),
    ]

    infer_shapes(g)

    assert g.get_tensor("out").shape == (2, 4)
