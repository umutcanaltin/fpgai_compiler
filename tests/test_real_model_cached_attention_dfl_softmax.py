from __future__ import annotations

from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.backends.hls.emit.layers_activations import emit_activations_h
from fpgai.backends.hls.emit.layers_attention import emit_attention_h
from fpgai.backends.hls.emit.top_train_cpp import emit_top_train_cpp
from fpgai.ir.graph import Graph


def _cfg():
    return {
        "numerics": {
            "defaults": {
                "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
                "accum": {"type": "ap_fixed", "total_bits": 32, "int_bits": 12},
            }
        },
        "targets": {"hls": {"control_protocol": "s_axilite"}},
        "training": {"loss": {"type": "mse"}, "optimizer": {"type": "sgd", "learning_rate": 0.01}},
    }


def test_cached_attention_accepts_short_query_full_cache_and_valid_length():
    g = Graph("cached_attention")
    g.inputs = ["q", "k_cache_view", "v_cache_view", "valid_length"]
    g.outputs = ["context"]
    g.add_tensor("q", (1, 1, 8), "float32")
    g.add_tensor("k_cache_view", (1, 8, 8), "float32")
    g.add_tensor("v_cache_view", (1, 8, 8), "float32")
    g.add_tensor("valid_length", (1,), "int32")
    g.add_tensor("context", (1, 1, 8), "float32")
    g.add_op(
        "MultiHeadAttention",
        ["q", "k_cache_view", "v_cache_view", "valid_length"],
        ["context"],
        name="cached_mha",
        attrs={"num_heads": 2, "causal": True},
    )
    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "multi_head_attention_cached_serialized<1, 8, 8, 2" in source
    assert "ap_int<32>" in source
    header = emit_attention_h()
    assert "void multi_head_attention_cached_serialized" in header
    assert "col < valid_length" in header
    assert "query_base" in header


def test_middle_axis_softmax_lowers_for_yolo_dfl_inference_and_training():
    g = Graph("dfl_softmax")
    g.inputs = ["x"]
    g.outputs = ["z"]
    g.add_tensor("x", (1, 4, 16, 3), "float32")
    g.add_tensor("probs", (1, 4, 16, 3), "float32")
    g.add_tensor("z", (1, 4, 16, 3), "float32")
    g.add_op("Softmax", ["x"], ["probs"], name="dfl_softmax", attrs={"axis": 2})
    g.add_op("Identity", ["probs"], ["z"], name="keep_softmax_internal")
    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    # outer=1*4, axis=16, inner=3
    assert "softmax_axis_typed<4, 16, 3" in source
    train = emit_top_train_cpp(graph=g, top_name="train", weights_mode="embedded", training_cfg=_cfg()["training"])
    assert "softmax_axis_typed<4, 16, 3" in train
    assert "softmax_axis_backward_typed<4, 16, 3" in train
    activations = emit_activations_h()
    assert "void softmax_axis_typed" in activations
    assert "void softmax_axis_backward_typed" in activations
