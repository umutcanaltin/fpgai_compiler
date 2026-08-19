"""Maintained small model graphs used for compiler validation and examples."""

from __future__ import annotations

import numpy as np

from fpgai.ir import Graph
from fpgai.ir.passes.attention_lowering import plan_attention_lowering
from fpgai.ir.passes.transformer_lowering import configure_kv_cache_state, plan_transformer_execution


def build_demo_attention_graph(*, sequence_length: int = 4, head_dimension: int = 8) -> Graph:
    g = Graph("attention_core")
    g.inputs = ["q", "k", "v"]
    g.outputs = ["context"]
    s, d = int(sequence_length), int(head_dimension)
    for name, shape in {
        "q": (1, s, d), "k": (1, s, d), "v": (1, s, d),
        "kt": (1, d, s), "scores": (1, s, s), "scaled": (1, s, s),
        "probs": (1, s, s), "context": (1, s, d),
    }.items():
        g.add_tensor(name, shape, "float32")
    g.add_tensor("scale", (), "float32")
    g.constants["scale"] = np.asarray([1.0 / np.sqrt(float(d))], dtype=np.float32)
    g.add_op("Transpose", ["k"], ["kt"], name="transpose_k", attrs={"perm": [0, 2, 1]})
    g.add_op("MatMul", ["q", "kt"], ["scores"], name="score_matmul")
    g.add_op("Mul", ["scores", "scale"], ["scaled"], name="scale_scores")
    g.add_op("Softmax", ["scaled"], ["probs"], name="attention_softmax", attrs={"axis": -1})
    g.add_op("MatMul", ["probs", "v"], ["context"], name="value_matmul")
    plan_attention_lowering(g, tile_m=2, tile_n=2, tile_k=4)
    return g


def _identity_projection(dim: int, *, gain: float = 1.0) -> np.ndarray:
    return (np.eye(dim, dtype=np.float32) * np.float32(gain)).astype(np.float32)


def _rope_tables(sequence_length: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if rotary_dim % 2:
        raise ValueError("MODELGRAPH001: rotary_dim must be even")
    positions = np.arange(sequence_length, dtype=np.float32)[:, None]
    pair = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = 1.0 / np.power(10000.0, (2.0 * pair) / float(rotary_dim))
    angles = positions * inv_freq
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def build_demo_transformer_block_graph(
    *,
    sequence_length: int = 4,
    model_dimension: int = 8,
    num_heads: int = 2,
    max_sequence_length: int = 16,
    execution_mode: str = "auto",
) -> Graph:
    s, d, h = int(sequence_length), int(model_dimension), int(num_heads)
    if d % h:
        raise ValueError("MODELGRAPH002: model_dimension must be divisible by num_heads")

    g = Graph("tiny_transformer_block")
    g.inputs = ["x"]
    g.outputs = ["norm_out"]
    for name in ("x", "q", "k", "v", "q_rope", "k_rope", "context", "projected", "norm_out"):
        g.add_tensor(name, (1, s, d), "float32")
    for name in ("wq", "wk", "wv", "wo"):
        g.add_tensor(name, (d, d), "float32")
    g.add_tensor("norm_scale", (d,), "float32")
    g.add_tensor("rope_cos", (s, d // 2), "float32")
    g.add_tensor("rope_sin", (s, d // 2), "float32")
    g.add_tensor("k_cache", (1, max_sequence_length, d), "float32")
    g.add_tensor("v_cache", (1, max_sequence_length, d), "float32")

    g.constants["wq"] = _identity_projection(d, gain=0.90)
    g.constants["wk"] = _identity_projection(d, gain=1.05)
    g.constants["wv"] = _identity_projection(d, gain=0.85)
    g.constants["wo"] = _identity_projection(d, gain=0.95)
    g.constants["norm_scale"] = np.linspace(0.9, 1.1, d, dtype=np.float32)
    cos, sin = _rope_tables(s, d)
    g.constants["rope_cos"] = cos
    g.constants["rope_sin"] = sin

    for role, weight, output in (("q", "wq", "q"), ("k", "wk", "k"), ("v", "wv", "v")):
        g.add_op("MatMul", ["x", weight], [output], name=f"{role}_projection", attrs={"projection_role": role})
    g.add_op("RotaryEmbedding", ["q", "rope_cos", "rope_sin"], ["q_rope"], name="q_rope", attrs={"rotary_dim": d})
    g.add_op("RotaryEmbedding", ["k", "rope_cos", "rope_sin"], ["k_rope"], name="k_rope", attrs={"rotary_dim": d})
    g.add_op(
        "MultiHeadAttention", ["q_rope", "k_rope", "v"], ["context"], name="mha",
        attrs={"num_heads": h, "causal": True, "execution_mode": execution_mode},
    )
    g.add_op("MatMul", ["context", "wo"], ["projected"], name="o_projection", attrs={"projection_role": "o"})
    g.add_op("RMSNorm", ["projected", "norm_scale"], ["norm_out"], name="rmsnorm", attrs={"axis": -1, "epsilon": 1e-5})

    configure_kv_cache_state(
        g, key_cache="k_cache", value_cache="v_cache", capacity=max_sequence_length,
        sequence_axis=1, storage="auto",
    )
    plan_transformer_execution(
        g,
        model_dimension=d,
        num_heads=h,
        max_sequence_length=max_sequence_length,
        execution_mode=execution_mode,
        weight_storage="auto",
        kv_cache_storage="auto",
    )
    return g


__all__ = ["build_demo_attention_graph", "build_demo_transformer_block_graph"]


def build_yolo_like_multiscale_detector_graph(*, class_count: int = 4, dfl_bins: int = 8) -> Graph:
    """Build a compact generic multi-scale one-stage detector graph.

    This is a maintained compiler-validation architecture, not an implementation
    of a named detector. It deliberately exercises the generic mechanisms used
    by modern YOLO-like detectors.
    """
    from fpgai.ir.passes.detection_lowering import plan_detection_decode, plan_detection_output

    classes, bins = int(class_count), int(dfl_bins)
    if classes <= 0 or bins <= 1:
        raise ValueError("MODELGRAPH003: detector requires positive classes and dfl_bins > 1")
    g = Graph("yolo_like_multiscale_detector")
    g.inputs = ["image"]
    g.outputs = ["raw_predictions"]
    g.add_tensor("image", (1, 3, 8, 8), "float32")

    constants = {
        "w_stem": np.full((8, 3, 3, 3), 0.02, np.float32),
        "b_stem": np.zeros((8,), np.float32),
        "w_dw": np.full((8, 1, 3, 3), 0.02, np.float32),
        "b_dw": np.zeros((8,), np.float32),
        "w_down": np.full((8, 8, 3, 3), 0.02, np.float32),
        "b_down": np.zeros((8,), np.float32),
        "w_head": np.full((4 * bins + classes, 16, 1, 1), 0.01, np.float32),
        "b_head": np.zeros((4 * bins + classes,), np.float32),
        "dfl_bins": np.arange(bins, dtype=np.float32),
        "decode_zero": np.zeros((1, 4, 16), np.float32),
    }
    for name, value in constants.items():
        g.add_tensor(name, tuple(value.shape), "float32")
        g.constants[name] = value

    shapes = {
        "stem": (1, 8, 4, 4), "dw": (1, 8, 4, 4), "coarse": (1, 8, 2, 2),
        "coarse_up": (1, 8, 4, 4), "fused": (1, 16, 4, 4),
        "head": (1, 4 * bins + classes, 4, 4), "box_logits_chw": (1, 4 * bins, 4, 4),
        "class_logits_chw": (1, classes, 4, 4), "box_logits": (1, 4, 16, bins),
        "box_prob": (1, 4, 16, bins), "box_weighted": (1, 4, 16, bins),
        "distances": (1, 4, 16), "decoded_boxes": (1, 4, 16),
        "class_logits": (1, classes, 16), "raw_predictions": (1, 4 + classes, 16),
    }
    for name, shape in shapes.items():
        g.add_tensor(name, shape, "float32")

    g.add_op("Conv", ["image", "w_stem", "b_stem"], ["stem"], name="stem", attrs={"strides": [2, 2], "pads": [1, 1, 1, 1]})
    g.add_op("Conv", ["stem", "w_dw", "b_dw"], ["dw"], name="depthwise", attrs={"strides": [1, 1], "pads": [1, 1, 1, 1], "group": 8})
    g.add_op("Conv", ["dw", "w_down", "b_down"], ["coarse"], name="downsample", attrs={"strides": [2, 2], "pads": [1, 1, 1, 1]})
    g.add_op("Resize", ["coarse"], ["coarse_up"], name="upsample", attrs={"mode": "nearest", "sizes": [1, 8, 4, 4]})
    g.add_op("Concat", ["dw", "coarse_up"], ["fused"], name="feature_fusion", attrs={"axis": 1})
    g.add_op("Conv", ["fused", "w_head", "b_head"], ["head"], name="detection_head", attrs={"strides": [1, 1], "pads": [0, 0, 0, 0]})
    g.add_op("Slice", ["head"], ["box_logits_chw"], name="box_channels", attrs={"starts": [0], "ends": [4 * bins], "axes": [1], "steps": [1]})
    g.add_op("Slice", ["head"], ["class_logits_chw"], name="class_channels", attrs={"starts": [4 * bins], "ends": [4 * bins + classes], "axes": [1], "steps": [1]})
    g.add_op("Reshape", ["box_logits_chw"], ["box_logits"], name="box_reshape", attrs={"shape": [1, 4, 16, bins]})
    g.add_op("Softmax", ["box_logits"], ["box_prob"], name="distribution_softmax", attrs={"axis": -1})
    g.add_op("Mul", ["box_prob", "dfl_bins"], ["box_weighted"], name="distribution_bins")
    g.add_op("ReduceSum", ["box_weighted"], ["distances"], name="distribution_expectation", attrs={"axes": [-1], "keepdims": 0})
    g.add_op("Add", ["distances", "decode_zero"], ["decoded_boxes"], name="grid_stride_decode")
    g.add_op("Reshape", ["class_logits_chw"], ["class_logits"], name="class_reshape", attrs={"shape": [1, classes, 16]})
    g.add_op("Concat", ["decoded_boxes", "class_logits"], ["raw_predictions"], name="raw_prediction_concat", attrs={"axis": 1})

    plan_detection_decode(g, distance_tensor="distances", decoded_box_tensor="decoded_boxes", dfl_bins=bins, pyramid_strides=(2, 4), grid_origin=0.5)
    plan_detection_output(g, output_tensor="raw_predictions", class_count=classes, pyramid_strides=(2, 4), postprocess_partition="ps_or_host", nms_required=True)
    g.metadata["reference_architecture"] = "yolo_like_multiscale_detector"
    return g


def build_single_stage_detector_graph(*, class_count: int = 4) -> Graph:
    """Build a compact generic single-stage detector without distribution regression."""
    from fpgai.ir.passes.detection_lowering import plan_detection_output

    classes = int(class_count)
    if classes <= 0:
        raise ValueError("MODELGRAPH004: class_count must be positive")
    g = Graph("single_stage_detector")
    g.inputs = ["image"]
    g.outputs = ["raw_predictions"]
    g.add_tensor("image", (1, 3, 8, 8), "float32")
    for name, value in {
        "w_backbone": np.full((8, 3, 3, 3), 0.02, np.float32),
        "b_backbone": np.zeros((8,), np.float32),
        "w_head": np.full((4 + classes, 8, 1, 1), 0.01, np.float32),
        "b_head": np.zeros((4 + classes,), np.float32),
    }.items():
        g.add_tensor(name, tuple(value.shape), "float32"); g.constants[name] = value
    g.add_tensor("features", (1, 8, 4, 4), "float32")
    g.add_tensor("head", (1, 4 + classes, 4, 4), "float32")
    g.add_tensor("raw_predictions", (1, 4 + classes, 16), "float32")
    g.add_op("Conv", ["image", "w_backbone", "b_backbone"], ["features"], name="backbone", attrs={"strides": [2, 2], "pads": [1, 1, 1, 1]})
    g.add_op("Conv", ["features", "w_head", "b_head"], ["head"], name="prediction_head", attrs={"strides": [1, 1], "pads": [0, 0, 0, 0]})
    g.add_op("Reshape", ["head"], ["raw_predictions"], name="prediction_reshape", attrs={"shape": [1, 4 + classes, 16]})
    plan_detection_output(g, output_tensor="raw_predictions", class_count=classes, pyramid_strides=(2,), postprocess_partition="ps_or_host", nms_required=True)
    g.metadata["reference_architecture"] = "single_stage_detector"
    return g


def build_llm_like_decoder_graph(
    *, model_dimension: int = 12, num_heads: int = 3, num_kv_heads: int = 1,
    hidden_dimension: int = 24, vocabulary_size: int = 32, max_sequence_length: int = 16,
    cache_storage: str = "bram",
) -> Graph:
    """Build a compact one-token decoder representing modern LLM mechanisms."""
    from fpgai.ir.passes.transformer_lowering import configure_kv_cache_state, plan_autoregressive_runtime

    d, h, kh = int(model_dimension), int(num_heads), int(num_kv_heads)
    if d <= 0 or h <= 0 or kh <= 0 or d % h or h % kh:
        raise ValueError("MODELGRAPH005: invalid model/head dimensions")
    head_dim = d // h; kv_width = kh * head_dim
    g = Graph("llm_like_decoder")
    g.inputs = ["token_id"]; g.outputs = ["logits"]
    g.add_tensor("token_id", (1,), "int64")
    embedding = np.arange(vocabulary_size * d, dtype=np.float32).reshape(vocabulary_size, d) / 1000.0
    constants = {
        "embedding": embedding, "lm_head": embedding.T.copy(),
        "norm1_scale": np.ones((d,), np.float32), "norm2_scale": np.ones((d,), np.float32), "final_norm_scale": np.ones((d,), np.float32),
        "wq": _identity_projection(d, gain=0.9), "wk": np.full((d, kv_width), 0.02, np.float32), "wv": np.full((d, kv_width), 0.02, np.float32),
        "wo": _identity_projection(d, gain=0.95), "wg": np.full((d, hidden_dimension), 0.02, np.float32), "wu": np.full((d, hidden_dimension), 0.02, np.float32),
        "wd": np.full((hidden_dimension, d), 0.02, np.float32),
    }
    cos, sin = _rope_tables(max_sequence_length, head_dim)
    constants["rope_cos"] = cos; constants["rope_sin"] = sin
    for name, value in constants.items(): g.add_tensor(name, tuple(value.shape), "float32"); g.constants[name] = value
    shapes = {
        "x": (1,1,d), "n1": (1,1,d), "q": (1,1,d), "k": (1,1,kv_width), "v": (1,1,kv_width), "position": (1,),
        "q_rope": (1,1,d), "k_rope": (1,1,kv_width), "k_cache": (1,max_sequence_length,kv_width), "v_cache": (1,max_sequence_length,kv_width),
        "k_after": (1,max_sequence_length,kv_width), "v_after": (1,max_sequence_length,kv_width), "valid_length": (1,), "k_read": (1,max_sequence_length,kv_width),
        "v_read": (1,max_sequence_length,kv_width), "context": (1,1,d), "attn_out": (1,1,d), "res1": (1,1,d), "n2": (1,1,d),
        "gate": (1,1,hidden_dimension), "up": (1,1,hidden_dimension), "gate_act": (1,1,hidden_dimension), "mixed": (1,1,hidden_dimension),
        "down": (1,1,d), "res2": (1,1,d), "final_norm": (1,1,d), "logits": (1,1,vocabulary_size),
    }
    for name, shape in shapes.items(): g.add_tensor(name, shape, "int32" if name in {"position","valid_length"} else "float32")
    configure_kv_cache_state(g, key_cache="k_cache", value_cache="v_cache", capacity=max_sequence_length, sequence_axis=1, storage=cache_storage)
    g.add_op("Gather", ["embedding","token_id"], ["x"], name="embedding_lookup", attrs={"axis":0})
    g.add_op("RMSNorm", ["x","norm1_scale"], ["n1"], name="attn_norm", attrs={"axis":-1})
    g.add_op("MatMul", ["n1","wq"], ["q"], name="q_proj"); g.add_op("MatMul", ["n1","wk"], ["k"], name="k_proj"); g.add_op("MatMul", ["n1","wv"], ["v"], name="v_proj")
    g.add_op("PersistentStateLength", ["k_cache"], ["position"], name="decode_position")
    g.add_op("RotaryEmbedding", ["q","rope_cos","rope_sin","position"], ["q_rope"], name="q_rope", attrs={"rotary_dim":head_dim,"num_heads":h})
    g.add_op("RotaryEmbedding", ["k","rope_cos","rope_sin","position"], ["k_rope"], name="k_rope", attrs={"rotary_dim":head_dim,"num_heads":kh})
    g.add_op("KVCacheUpdate", ["k_cache","k_rope"], ["k_after"], name="append_k", attrs={"sequence_axis":1,"capacity":max_sequence_length,"update_policy":"append"})
    g.add_op("KVCacheUpdate", ["v_cache","v"], ["v_after"], name="append_v", attrs={"sequence_axis":1,"capacity":max_sequence_length,"update_policy":"append"})
    g.add_op("PersistentStateLength", ["k_cache"], ["valid_length"], name="valid_length"); g.add_op("PersistentStateRead", ["k_cache"], ["k_read"], name="read_k"); g.add_op("PersistentStateRead", ["v_cache"], ["v_read"], name="read_v")
    g.add_op("MultiHeadAttention", ["q_rope","k_read","v_read","valid_length"], ["context"], name="attention", attrs={"num_heads":h,"num_kv_heads":kh,"causal":True})
    g.add_op("MatMul", ["context","wo"], ["attn_out"], name="o_proj"); g.add_op("Add", ["x","attn_out"], ["res1"], name="attn_residual")
    g.add_op("RMSNorm", ["res1","norm2_scale"], ["n2"], name="ffn_norm", attrs={"axis":-1}); g.add_op("MatMul", ["n2","wg"], ["gate"], name="gate_proj"); g.add_op("MatMul", ["n2","wu"], ["up"], name="up_proj")
    g.add_op("SiLU", ["gate"], ["gate_act"], name="gate_act"); g.add_op("Mul", ["gate_act","up"], ["mixed"], name="swiglu"); g.add_op("MatMul", ["mixed","wd"], ["down"], name="down_proj"); g.add_op("Add", ["res1","down"], ["res2"], name="ffn_residual")
    g.add_op("RMSNorm", ["res2","final_norm_scale"], ["final_norm"], name="final_norm", attrs={"axis":-1}); g.add_op("MatMul", ["final_norm","lm_head"], ["logits"], name="lm_head")
    plan_autoregressive_runtime(g, layer_caches=[{"owner":"decoder.layer0","state_group":"decoder.layer0.kv","key":"k_cache","value":"v_cache"}], max_sequence_length=max_sequence_length, prefill_sequence_length=min(4,max_sequence_length), decode_sequence_length=1, cache_storage=cache_storage, tied_parameter_groups=[{"name":"token_embedding_lm_head","members":[{"tensor":"embedding","view":"native"},{"tensor":"lm_head","view":"transpose"}]}])
    g.metadata["reference_architecture"] = "llm_like_gqa_decoder" if kh < h else "llm_like_mha_decoder"
    return g


__all__ = [
    "build_demo_attention_graph", "build_demo_transformer_block_graph",
    "build_yolo_like_multiscale_detector_graph", "build_single_stage_detector_graph",
    "build_llm_like_decoder_graph",
]
