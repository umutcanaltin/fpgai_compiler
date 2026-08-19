from __future__ import annotations

import numpy as np

from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.backends.hls.emit.top_train_cpp import emit_top_train_cpp
from fpgai.backends.hls.emit.layers_tensor import emit_tensor_h
from fpgai.capabilities.capabilities import capability_for
from fpgai.ir.graph import Graph
from fpgai.ir.passes.infer_shapes import infer_shapes
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
        "training": {"loss": {"type": "mse"}, "optimizer": {"type": "sgd", "learning_rate": 0.01}},
    }


def _persistent_graph(op_type: str) -> Graph:
    g = Graph("persistent_control")
    g.add_tensor("k_cache", (1, 2, 8, 4), "float32")
    configure_kv_cache_state(
        g,
        key_cache="k_cache",
        value_cache="k_cache",
        capacity=8,
        sequence_axis=2,
        storage="bram",
    )
    if op_type == "PersistentStateRead":
        g.inputs = ["dummy"]
        g.outputs = ["state_view"]
        g.add_tensor("dummy", (1,), "uint8")
        g.add_tensor("state_view", ())
        g.add_op(op_type, ["k_cache"], ["state_view"], name="state_read")
    elif op_type == "PersistentStateLength":
        g.inputs = ["dummy"]
        g.outputs = ["length"]
        g.add_tensor("dummy", (1,), "uint8")
        g.add_tensor("length", ())
        g.add_op(op_type, ["k_cache"], ["length"], name="state_length")
    else:
        g.inputs = ["reset"]
        g.outputs = ["length"]
        g.add_tensor("reset", (1,), "uint8")
        g.add_tensor("length", (1,), "int32")
        g.add_op(op_type, ["k_cache", "reset"], ["length"], name="state_reset")
    infer_shapes(g)
    return g


def test_persistent_state_read_length_reset_lower_to_generic_hls_primitives() -> None:
    read = _persistent_graph("PersistentStateRead")
    assert read.get_tensor("state_view").shape == (1, 2, 8, 4)
    read_hls = emit_dag_top_cpp(read, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "persistent_state_snapshot<64" in read_hls

    length = _persistent_graph("PersistentStateLength")
    assert length.get_tensor("length").shape == (1,)
    assert length.get_tensor("length").dtype == "int32"
    length_hls = emit_dag_top_cpp(length, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "persistent_state_length<ap_int<32>>" in length_hls

    reset = _persistent_graph("PersistentStateReset")
    reset_hls = emit_dag_top_cpp(reset, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "persistent_state_reset_if<64" in reset_hls
    assert "fpgai_state_k_cache_cursor" in reset_hls
    assert capability_for("PersistentStateReset", "inference").status == "limited"


def test_rotary_embedding_accepts_runtime_decode_position() -> None:
    g = Graph("rope_runtime_position")
    g.inputs = ["x", "position"]
    g.outputs = ["y"]
    g.add_tensor("x", (1, 1, 8), "float32")
    g.add_tensor("position", (1,), "int32")
    g.add_tensor("cos", (16, 4), "float32")
    g.add_tensor("sin", (16, 4), "float32")
    g.add_tensor("y", (1, 1, 8), "float32")
    g.constants["cos"] = np.ones((16, 4), dtype=np.float32)
    g.constants["sin"] = np.zeros((16, 4), dtype=np.float32)
    g.add_op(
        "RotaryEmbedding",
        ["x", "cos", "sin", "position"],
        ["y"],
        name="rope",
        attrs={"rotary_dim": 8},
    )
    hls = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "int fpgai_rope_position_0 = (int)" in hls
    assert "rotary_embedding_pairs<1, 8" in hls
    assert "fpgai_rope_position_0);" in hls


def test_concat_supports_three_way_fan_in_inference_and_training() -> None:
    g = Graph("concat3")
    g.inputs = ["a", "b", "c"]
    g.outputs = ["y"]
    g.add_tensor("a", (1, 2, 2))
    g.add_tensor("b", (1, 3, 2))
    g.add_tensor("c", (1, 1, 2))
    g.add_tensor("y", ())
    g.add_op("Concat", ["a", "b", "c"], ["y"], name="cat3", attrs={"axis": 1})
    infer_shapes(g)
    assert g.get_tensor("y").shape == (1, 6, 2)
    hls = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "concat_axis_segment<1, 6, 2, 0, 2" in hls
    assert "concat_axis_segment<1, 6, 3, 2, 2" in hls
    assert "concat_axis_segment<1, 6, 1, 5, 2" in hls
    train = emit_top_train_cpp(graph=g, top_name="train", weights_mode="embedded", training_cfg=_cfg()["training"])
    assert "concat_axis_backward_segment<1, 6, 2, 0, 2" in train
    assert "concat_axis_backward_segment<1, 6, 1, 5, 2" in train


def test_resize_emits_explicit_onnx_coordinate_and_nearest_modes() -> None:
    g = Graph("resize_modes")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_tensor("x", (1, 1, 3, 3))
    g.add_tensor("y", ())
    g.add_op(
        "Resize",
        ["x"],
        ["y"],
        name="resize",
        attrs={
            "mode": "nearest",
            "sizes": [1, 1, 6, 6],
            "coordinate_transformation_mode": "half_pixel",
            "nearest_mode": "round_prefer_floor",
        },
    )
    infer_shapes(g)
    hls = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "resize_nearest_nchw<1, 1, 3, 3, 6, 6, 1, 1" in hls
    assert "resize_nearest_source_index" in emit_tensor_h()
    train = emit_top_train_cpp(graph=g, top_name="train", weights_mode="embedded", training_cfg=_cfg()["training"])
    assert "resize_nearest_nchw_backward<1, 1, 3, 3, 6, 6, 1, 1" in train
