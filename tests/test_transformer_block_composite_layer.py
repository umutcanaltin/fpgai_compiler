import numpy as np

from fpgai.ir import Graph
from fpgai.layers.composites import composite_layer_registry, expand_composite_layers


def _graph():
    g = Graph("composite")
    g.add_tensor("x", (1, 4, 8))
    g.add_tensor("y", (1, 4, 8))
    for name in ("wq", "wk", "wv", "wo"):
        g.add_tensor(name, (8, 8))
        g.constants[name] = np.eye(8, dtype=np.float32)
    g.add_tensor("norm_scale", (8,))
    g.constants["norm_scale"] = np.ones((8,), dtype=np.float32)
    for name in ("rope_cos", "rope_sin"):
        g.add_tensor(name, (4, 4))
        g.constants[name] = np.ones((4, 4), dtype=np.float32)
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_op(
        "TransformerBlock",
        ["x", "wq", "wk", "wv", "wo", "norm_scale", "rope_cos", "rope_sin"],
        ["y"],
        name="block0",
        attrs={"num_heads": 2, "causal": True, "execution_mode": "serialized"},
    )
    return g


def test_transformer_block_is_composite_not_backend_special_case():
    assert "TransformerBlock" in composite_layer_registry()
    g = expand_composite_layers(_graph())
    assert all(op.op_type != "TransformerBlock" for op in g.ops)
    assert [op.op_type for op in g.ops] == [
        "MatMul", "MatMul", "MatMul", "RotaryEmbedding", "RotaryEmbedding",
        "MultiHeadAttention", "MatMul", "RMSNorm",
    ]
    assert all(op.attrs.get("expanded_from") == "TransformerBlock" for op in g.ops)
    assert g.metadata["composite_expansion"]["expanded_count"] == 1


def test_modified_graph_can_mix_composite_and_ordinary_layers():
    g = _graph()
    g.add_tensor("z", (1, 4, 8))
    g.ops[0].outputs = ["z"]
    g.add_op("Add", ["z", "x"], ["y"], name="custom_residual")
    expand_composite_layers(g)
    assert g.ops[-1].op_type == "Add"
    assert g.ops[-1].name == "custom_residual"
