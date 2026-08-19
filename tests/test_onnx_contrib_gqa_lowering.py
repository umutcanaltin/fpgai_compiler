from __future__ import annotations

import numpy as np

from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.ir.graph import Graph
from fpgai.ir.passes.transformer_lowering import internalize_explicit_group_query_attention_state


def _cfg():
    return {
        "numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}, "accum": {"type": "ap_fixed", "total_bits": 32, "int_bits": 12}}},
        "targets": {"hls": {"control_protocol": "s_axilite"}},
    }


def _gqa_graph():
    g = Graph("ort_gqa")
    g.inputs = ["q", "k", "v", "past_k", "past_v", "seqlens", "total"]
    g.outputs = ["context", "present_k", "present_v"]
    for name, shape, dtype in [
        ("q", (1, 1, 576), "float32"), ("k", (1, 1, 192), "float32"), ("v", (1, 1, 192), "float32"),
        ("past_k", (1, 3, 1, 64), "float32"), ("past_v", (1, 3, 1, 64), "float32"),
        ("seqlens", (1,), "int32"), ("total", (1,), "int32"),
        ("context", (1, 1, 576), "float32"), ("present_k", (1, 3, 16, 64), "float32"), ("present_v", (1, 3, 16, 64), "float32"),
    ]:
        g.add_tensor(name, shape, dtype)
    g.add_op("GroupQueryAttention", g.inputs[:7], g.outputs, name="layer0_gqa", attrs={"num_heads": 9, "num_kv_heads": 3, "causal": True})
    return g


def test_explicit_gqa_cache_ports_become_persistent_state():
    g = _gqa_graph()
    plan = internalize_explicit_group_query_attention_state(g, max_sequence_length=16, cache_storage="ddr")
    assert plan["layer_count"] == 1
    assert all(op.op_type != "GroupQueryAttention" for op in g.ops)
    assert "past_k" not in g.inputs and "past_v" not in g.inputs
    assert "present_k" not in g.outputs and "present_v" not in g.outputs
    assert g.outputs == ["context"]
    ks = g.get_tensor("layer0_gqa__key_cache")
    assert tuple(ks.shape) == (1, 16, 192)
    assert ks.semantics.state.persistent_across_invocations is True
    assert ks.semantics.memory.storage == "ddr"
    assert [op.op_type for op in g.ops].count("KVCacheUpdate") == 2
    assert any(op.op_type == "MultiHeadAttention" for op in g.ops)


def test_head_aware_rope_and_internalized_gqa_emit_hls():
    g = _gqa_graph()
    # Add static RoPE tables and enable fused RoPE input slots.
    g.add_tensor("cos", (16, 32), "float32")
    g.add_tensor("sin", (16, 32), "float32")
    g.constants["cos"] = np.ones((16, 32), dtype=np.float32)
    g.constants["sin"] = np.zeros((16, 32), dtype=np.float32)
    op = g.ops[0]
    op.inputs += ["cos", "sin"]
    op.attrs.update({"do_rotary": True, "rotary_interleaved": False})
    internalize_explicit_group_query_attention_state(g, max_sequence_length=16, cache_storage="bram")
    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "rotary_embedding_heads<1, 576, 9, 64, false" in source
    assert "rotary_embedding_heads<1, 192, 3, 64, false" in source
    assert "multi_head_attention_cached_serialized<1, 16, 576, 9, 3" in source
