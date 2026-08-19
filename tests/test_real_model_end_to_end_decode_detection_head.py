from __future__ import annotations

import numpy as np

from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.backends.hls.emit.top_train_cpp import emit_top_train_cpp
from fpgai.capabilities.capabilities import capability_for
from fpgai.ir.graph import Graph
from fpgai.ir.passes.transformer_lowering import configure_kv_cache_state


def _cfg():
    return {
        "numerics": {
            "defaults": {
                "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
                "accum": {"type": "ap_fixed", "total_bits": 32, "int_bits": 12},
            }
        },
        "targets": {"hls": {"control_protocol": "s_axilite"}},
    }


def _const(g: Graph, name: str, shape: tuple[int, ...], value: float = 0.05) -> None:
    g.add_tensor(name, shape, "float32")
    g.constants[name] = np.full(shape, value, dtype=np.float32)


def _one_token_decode_graph() -> Graph:
    model = 12
    heads = 3
    kv_heads = 1
    head_dim = model // heads
    kv_width = kv_heads * head_dim
    hidden = 24
    vocab = 32
    capacity = 8

    g = Graph("generic_one_token_decode")
    g.inputs = ["token_id"]
    g.outputs = ["logits"]
    g.add_tensor("token_id", (1,), "int64")

    _const(g, "embedding", (vocab, model), 0.02)
    _const(g, "norm1_scale", (model,), 1.0)
    _const(g, "wq", (model, model), 0.03)
    _const(g, "wk", (model, kv_width), 0.03)
    _const(g, "wv", (model, kv_width), 0.03)
    _const(g, "wo", (model, model), 0.03)
    _const(g, "norm2_scale", (model,), 1.0)
    _const(g, "wg", (model, hidden), 0.02)
    _const(g, "wu", (model, hidden), 0.02)
    _const(g, "wd", (hidden, model), 0.02)
    _const(g, "final_norm_scale", (model,), 1.0)
    _const(g, "lm_head", (model, vocab), 0.01)

    # RoPE tables use the flattened projection widths. This remains generic:
    # the operator itself owns its rotary dimension rather than the model name.
    _const(g, "q_cos", (capacity, model // 2), 1.0)
    _const(g, "q_sin", (capacity, model // 2), 0.0)
    _const(g, "k_cos", (capacity, kv_width // 2), 1.0)
    _const(g, "k_sin", (capacity, kv_width // 2), 0.0)

    for name, shape, dtype in (
        ("x", (1, 1, model), "float32"),
        ("n1", (1, 1, model), "float32"),
        ("q", (1, 1, model), "float32"),
        ("k", (1, 1, kv_width), "float32"),
        ("v", (1, 1, kv_width), "float32"),
        ("position", (1,), "int32"),
        ("q_rope", (1, 1, model), "float32"),
        ("k_rope", (1, 1, kv_width), "float32"),
        ("k_cache", (1, capacity, kv_width), "float32"),
        ("v_cache", (1, capacity, kv_width), "float32"),
        ("k_after", (1, capacity, kv_width), "float32"),
        ("v_after", (1, capacity, kv_width), "float32"),
        ("valid_length", (1,), "int32"),
        ("k_read", (1, capacity, kv_width), "float32"),
        ("v_read", (1, capacity, kv_width), "float32"),
        ("context", (1, 1, model), "float32"),
        ("attn_out", (1, 1, model), "float32"),
        ("res1", (1, 1, model), "float32"),
        ("n2", (1, 1, model), "float32"),
        ("gate", (1, 1, hidden), "float32"),
        ("up", (1, 1, hidden), "float32"),
        ("gate_act", (1, 1, hidden), "float32"),
        ("mixed", (1, 1, hidden), "float32"),
        ("down", (1, 1, model), "float32"),
        ("res2", (1, 1, model), "float32"),
        ("final_norm", (1, 1, model), "float32"),
        ("logits", (1, 1, vocab), "float32"),
    ):
        g.add_tensor(name, shape, dtype)

    configure_kv_cache_state(
        g,
        key_cache="k_cache",
        value_cache="v_cache",
        capacity=capacity,
        sequence_axis=1,
        storage="bram",
    )

    g.add_op("Gather", ["embedding", "token_id"], ["x"], name="embedding_lookup", attrs={"axis": 0})
    g.add_op("RMSNorm", ["x", "norm1_scale"], ["n1"], name="attn_norm", attrs={"axis": -1})
    g.add_op("MatMul", ["n1", "wq"], ["q"], name="q_proj")
    g.add_op("MatMul", ["n1", "wk"], ["k"], name="k_proj")
    g.add_op("MatMul", ["n1", "wv"], ["v"], name="v_proj")
    g.add_op("PersistentStateLength", ["k_cache"], ["position"], name="decode_position")
    g.add_op("RotaryEmbedding", ["q", "q_cos", "q_sin", "position"], ["q_rope"], name="q_rope", attrs={"rotary_dim": model})
    g.add_op("RotaryEmbedding", ["k", "k_cos", "k_sin", "position"], ["k_rope"], name="k_rope", attrs={"rotary_dim": kv_width})
    g.add_op("KVCacheUpdate", ["k_cache", "k_rope"], ["k_after"], name="append_k", attrs={"sequence_axis": 1, "capacity": capacity, "update_policy": "append"})
    g.add_op("KVCacheUpdate", ["v_cache", "v"], ["v_after"], name="append_v", attrs={"sequence_axis": 1, "capacity": capacity, "update_policy": "append"})
    g.add_op("PersistentStateLength", ["k_cache"], ["valid_length"], name="valid_length")
    g.add_op("PersistentStateRead", ["k_cache"], ["k_read"], name="read_k")
    g.add_op("PersistentStateRead", ["v_cache"], ["v_read"], name="read_v")
    g.add_op("MultiHeadAttention", ["q_rope", "k_read", "v_read", "valid_length"], ["context"], name="cached_gqa", attrs={"num_heads": heads, "num_kv_heads": kv_heads, "causal": True})
    g.add_op("MatMul", ["context", "wo"], ["attn_out"], name="o_proj")
    g.add_op("Add", ["x", "attn_out"], ["res1"], name="attn_residual")
    g.add_op("RMSNorm", ["res1", "norm2_scale"], ["n2"], name="ffn_norm", attrs={"axis": -1})
    g.add_op("MatMul", ["n2", "wg"], ["gate"], name="gate_proj")
    g.add_op("MatMul", ["n2", "wu"], ["up"], name="up_proj")
    g.add_op("SiLU", ["gate"], ["gate_act"], name="gate_act")
    g.add_op("Mul", ["gate_act", "up"], ["mixed"], name="swiglu")
    g.add_op("MatMul", ["mixed", "wd"], ["down"], name="down_proj")
    g.add_op("Add", ["res1", "down"], ["res2"], name="ffn_residual")
    g.add_op("RMSNorm", ["res2", "final_norm_scale"], ["final_norm"], name="final_norm", attrs={"axis": -1})
    g.add_op("MatMul", ["final_norm", "lm_head"], ["logits"], name="lm_head")
    return g


def test_complete_generic_one_token_decode_graph_reaches_hls_codegen() -> None:
    source = emit_dag_top_cpp(_one_token_decode_graph(), top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    for token in (
        "gather_rows<32, 12, 1",
        "persistent_state_length<",
        "rotary_embedding_pairs<1, 12",
        "rotary_embedding_pairs<1, 4",
        "persistent_state_append_axis<1, 8, 1, 4",
        "persistent_state_snapshot<32",
        "multi_head_attention_cached_serialized<1, 8, 12, 3, 1",
        "silu_vector<24",
        "matmul_tiled<1, 12, 32",
    ):
        assert token in source
    assert "static int fpgai_state_k_cache_cursor = 0;" in source
    assert "static int fpgai_state_v_cache_cursor = 0;" in source


def test_detection_distribution_expectation_uses_generic_softmax_mul_reduce_sum() -> None:
    g = Graph("generic_distribution_expectation")
    g.inputs = ["logits"]
    g.outputs = ["distances"]
    g.add_tensor("logits", (1, 4, 10, 16), "float32")
    g.add_tensor("prob", (1, 4, 10, 16), "float32")
    g.add_tensor("bins", (16,), "float32")
    g.constants["bins"] = np.arange(16, dtype=np.float32)
    g.add_tensor("weighted", (1, 4, 10, 16), "float32")
    g.add_tensor("distances", (1, 4, 10), "float32")
    g.add_op("Softmax", ["logits"], ["prob"], name="distribution_softmax", attrs={"axis": -1})
    g.add_op("Mul", ["prob", "bins"], ["weighted"], name="bin_weighting")
    g.add_op("ReduceSum", ["weighted"], ["distances"], name="expectation", attrs={"axes": [-1], "keepdims": 0})

    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "softmax_rows<40, 16" in source
    assert "mul_rows_by_col_vector<40, 16" in source
    assert "reduce_sum_axis_typed<40, 16, 1" in source
    assert capability_for("ReduceSum", "inference").status == "supported"


def test_reduce_sum_training_emits_forward_and_backward() -> None:
    g = Graph("reduce_sum_training")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_tensor("x", (1, 4, 3), "float32")
    g.add_tensor("y", (1, 4), "float32")
    g.add_op("ReduceSum", ["x"], ["y"], name="sum", attrs={"axes": [-1], "keepdims": 0})
    source = emit_top_train_cpp(
        graph=g,
        top_name="deeplearn_train",
        weights_mode="embedded",
        training_cfg={"loss": {"type": "mse"}, "optimizer": {"learning_rate": 0.01}},
    )
    assert "reduce_sum_axis_typed<4, 3, 1" in source
    assert "reduce_sum_axis_backward_typed<4, 3, 1" in source
