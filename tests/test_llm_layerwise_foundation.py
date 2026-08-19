from __future__ import annotations

import numpy as np

from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.benchmark.graph_reference import execute_graph_reference
from fpgai.ir import Graph
from fpgai.ir.passes.transformer_lowering import configure_kv_cache_state, plan_token_decoding
from fpgai.layers.composites import expand_composite_layers


def _add_const(g: Graph, name: str, shape: tuple[int, ...], value: np.ndarray) -> None:
    g.add_tensor(name, shape)
    g.constants[name] = np.asarray(value, dtype=np.float32).reshape(shape)


def test_gated_mlp_composite_expands_layerwise_with_hidden_dimension():
    g = Graph("gated_mlp")
    g.add_tensor("x", (1, 4, 8))
    g.add_tensor("y", (1, 4, 8))
    _add_const(g, "wg", (8, 16), np.ones((8, 16)))
    _add_const(g, "wu", (8, 16), np.ones((8, 16)))
    _add_const(g, "wd", (16, 8), np.ones((16, 8)))
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_op("GatedMLP", ["x", "wg", "wu", "wd"], ["y"], name="ffn")

    expand_composite_layers(g)

    assert [op.op_type for op in g.ops] == ["MatMul", "MatMul", "SiLU", "Mul", "MatMul"]
    assert tuple(g.get_tensor("ffn__gate").shape) == (1, 4, 16)
    assert tuple(g.get_tensor("ffn__up").shape) == (1, 4, 16)
    assert tuple(g.get_tensor("y").shape) == (1, 4, 8)


def test_full_transformer_block_expands_to_residual_and_gated_mlp_layers():
    g = Graph("full_transformer")
    g.add_tensor("x", (1, 4, 8))
    g.add_tensor("y", (1, 4, 8))
    for name in ("wq", "wk", "wv", "wo"):
        _add_const(g, name, (8, 8), np.eye(8, dtype=np.float32))
    _add_const(g, "attn_norm", (8,), np.ones(8))
    _add_const(g, "rope_cos", (8, 4), np.ones((8, 4)))
    _add_const(g, "rope_sin", (8, 4), np.zeros((8, 4)))
    _add_const(g, "wg", (8, 16), np.ones((8, 16)) * 0.1)
    _add_const(g, "wu", (8, 16), np.ones((8, 16)) * 0.1)
    _add_const(g, "wd", (16, 8), np.ones((16, 8)) * 0.1)
    _add_const(g, "ffn_norm", (8,), np.ones(8))
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_op(
        "TransformerBlock",
        ["x", "wq", "wk", "wv", "wo", "attn_norm", "rope_cos", "rope_sin", "wg", "wu", "wd", "ffn_norm"],
        ["y"],
        name="block",
        attrs={"num_heads": 2, "causal": True, "position_offset": 2},
    )

    expand_composite_layers(g)
    types = [op.op_type for op in g.ops]
    assert "TransformerBlock" not in types
    assert types.count("Add") == 2
    assert "SiLU" in types
    assert "Mul" in types
    ropes = [op for op in g.ops if op.op_type == "RotaryEmbedding"]
    assert len(ropes) == 2
    assert all(op.attrs["position_offset"] == 2 for op in ropes)


def test_tensor_mul_silu_and_rope_offset_reach_generic_hls_emitter():
    g = Graph("ops")
    g.add_tensor("x", (1, 2, 4))
    g.add_tensor("z", (1, 2, 4))
    g.add_tensor("r", (1, 2, 4))
    g.add_tensor("a", (1, 2, 4))
    g.add_tensor("y", (1, 2, 4))
    _add_const(g, "cos", (4, 2), np.ones((4, 2)))
    _add_const(g, "sin", (4, 2), np.zeros((4, 2)))
    g.inputs = ["x", "z"]
    g.outputs = ["y"]
    g.add_op("RotaryEmbedding", ["x", "cos", "sin"], ["r"], name="rope", attrs={"rotary_dim": 4, "position_offset": 2})
    g.add_op("SiLU", ["r"], ["a"], name="silu")
    g.add_op("Mul", ["a", "z"], ["y"], name="mul")

    src = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg={})
    assert "rotary_embedding_pairs<2, 4" in src
    assert ", 2);" in src
    assert "silu_vector<8" in src
    assert "mul_vectors<8" in src


def test_rope_position_offset_matches_reference_table_slice():
    g = Graph("rope_ref")
    g.add_tensor("x", (1, 2, 4))
    g.add_tensor("y", (1, 2, 4))
    cos = np.arange(8, dtype=np.float32).reshape(4, 2) / 10.0 + 0.5
    sin = np.arange(8, dtype=np.float32).reshape(4, 2) / 20.0
    _add_const(g, "cos", (4, 2), cos)
    _add_const(g, "sin", (4, 2), sin)
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_op("RotaryEmbedding", ["x", "cos", "sin"], ["y"], name="rope", attrs={"rotary_dim": 4, "position_offset": 2})
    x = np.arange(8, dtype=np.float32).reshape(1, 2, 4) / 7.0
    out = execute_graph_reference(g, {"x": x})
    flat = x.reshape(2, 4)
    expected = np.empty_like(flat)
    c = cos[2:4]
    s = sin[2:4]
    expected[:, 0::2] = flat[:, 0::2] * c - flat[:, 1::2] * s
    expected[:, 1::2] = flat[:, 0::2] * s + flat[:, 1::2] * c
    np.testing.assert_allclose(out, expected.reshape(1, 2, 4), rtol=1e-6, atol=1e-6)


def test_token_decoding_plan_sets_explicit_persistent_state_contracts():
    g = Graph("decode")
    g.add_tensor("x", (1, 1, 8))
    g.add_tensor("k_cache", (1, 16, 8))
    g.add_tensor("v_cache", (1, 16, 8))
    g.add_tensor("q", (1, 1, 8))
    g.add_tensor("k", (1, 1, 8))
    g.add_tensor("v", (1, 1, 8))
    g.add_tensor("qr", (1, 1, 8))
    g.add_tensor("kr", (1, 1, 8))
    g.add_tensor("y", (1, 1, 8))
    _add_const(g, "cos", (16, 4), np.ones((16, 4)))
    _add_const(g, "sin", (16, 4), np.zeros((16, 4)))
    g.add_op("RotaryEmbedding", ["q", "cos", "sin"], ["qr"], name="q_rope", attrs={"rotary_dim": 8})
    g.add_op("RotaryEmbedding", ["k", "cos", "sin"], ["kr"], name="k_rope", attrs={"rotary_dim": 8})
    g.add_op("MultiHeadAttention", ["qr", "kr", "v"], ["y"], name="mha", attrs={"num_heads": 2})

    plan = plan_token_decoding(g, key_cache="k_cache", value_cache="v_cache", max_sequence_length=16, position_offset=7)

    assert plan.position_offset == 7
    assert g.get_tensor("k_cache").semantics.state.persistent_across_invocations is True
    assert g.get_tensor("v_cache").semantics.state.update_policy == "append"
    assert g.semantics.runtime_contract["persistent_kv_cache_backend_required"] is True
    assert all(op.attrs.get("position_offset") == 7 for op in g.ops if op.op_type == "RotaryEmbedding")
