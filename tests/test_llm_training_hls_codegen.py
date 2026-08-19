from __future__ import annotations

import numpy as np

from fpgai.analysis.training_capability import audit_training_capabilities
from fpgai.backends.hls.emit.top_train_cpp import emit_top_train_cpp
from fpgai.ir import Graph
from fpgai.layers.composites import expand_composite_layers


def _add_const(g: Graph, name: str, shape: tuple[int, ...], value: np.ndarray) -> None:
    g.add_tensor(name, shape)
    g.constants[name] = np.asarray(value, dtype=np.float32).reshape(shape)


def _full_transformer_graph() -> Graph:
    g = Graph("full_transformer_training")
    g.add_tensor("x", (1, 4, 8))
    g.add_tensor("y", (1, 4, 8))
    for name in ("wq", "wk", "wv", "wo"):
        _add_const(g, name, (8, 8), np.eye(8, dtype=np.float32))
    _add_const(g, "attn_norm", (8,), np.ones(8, dtype=np.float32))
    _add_const(g, "rope_cos", (8, 4), np.ones((8, 4), dtype=np.float32))
    _add_const(g, "rope_sin", (8, 4), np.zeros((8, 4), dtype=np.float32))
    _add_const(g, "wg", (8, 16), np.ones((8, 16), dtype=np.float32) * 0.1)
    _add_const(g, "wu", (8, 16), np.ones((8, 16), dtype=np.float32) * 0.1)
    _add_const(g, "wd", (16, 8), np.ones((16, 8), dtype=np.float32) * 0.1)
    _add_const(g, "ffn_norm", (8,), np.ones(8, dtype=np.float32))
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.add_op(
        "TransformerBlock",
        [
            "x", "wq", "wk", "wv", "wo", "attn_norm",
            "rope_cos", "rope_sin", "wg", "wu", "wd", "ffn_norm",
        ],
        ["y"],
        name="block",
        attrs={"num_heads": 2, "causal": True, "position_offset": 2},
    )
    expand_composite_layers(g)
    return g


def test_full_transformer_layerwise_graph_emits_forward_backward_and_updates() -> None:
    g = _full_transformer_graph()
    source = emit_top_train_cpp(
        graph=g,
        top_name="deeplearn_train",
        weights_mode="embedded",
        training_cfg={"loss": {"type": "mse"}, "optimizer": {"learning_rate": 0.01}},
    )

    for token in (
        "matmul_tiled<",
        "rms_norm_rows<",
        "silu_vector<",
        "mul_vectors<",
        "rotary_embedding_pairs<",
        "multi_head_attention_serialized<",
        "matmul_backward_left_accumulate<",
        "matmul_weight_grad<",
        "rms_norm_backward_rows<",
        "silu_backward_accumulate<",
        "mul_backward_accumulate<",
        "rotary_embedding_backward_pairs<",
        "multi_head_attention_backward_serialized<",
        "sgd_update_wgt<",
    ):
        assert token in source

    # The generic accumulation path must discover MatMul/norm parameters too,
    # rather than requiring a Dense/Conv layer to exist in the graph.
    assert "FPGAI_NATIVE_ACC_BATCH_COUNT" in source
    assert "ACC_dW_block__q_projection" in source
    assert "ACC_dN_G_block__attn_norm" in source

    # RoPE lookup tables are non-trainable compiler constants, but they must be
    # materialized in the generated translation unit before both forward and
    # backward kernels reference them. This is especially important for
    # source-driven MLIR graphs where tensor shapes are already known and shape
    # inference must not be responsible for emitting C++ declarations.
    for tag in ("block__q_rope", "block__k_rope"):
        cos_decl = f"static acc_t ROPE_COS_{tag}["
        sin_decl = f"static acc_t ROPE_SIN_{tag}["
        assert cos_decl in source
        assert sin_decl in source
        first_use = source.index(f"ROPE_COS_{tag}", source.index(cos_decl) + len(cos_decl))
        assert source.index(cos_decl) < first_use
        assert source.index(sin_decl) < first_use


def test_expanded_transformer_training_capability_is_complete_at_operator_contract_level() -> None:
    g = _full_transformer_graph()
    report = audit_training_capabilities(g)
    assert report["complete"] is True
    assert report["hardware_complete"] is True
    assert report["semantic_incomplete_count"] == 0
    assert report["hardware_incomplete_count"] == 0


def test_matmul_backward_kernels_declare_and_consume_scoped_controls() -> None:
    from fpgai.backends.hls.emit.layers_attention import emit_attention_h

    header = emit_attention_h()
    assert "void matmul_backward_left_accumulate" in header
    assert "void matmul_backward_right_accumulate" in header
    assert "int PIPELINE_II = 1" in header
    assert "#pragma HLS PIPELINE II=PIPELINE_II" in header
    assert "factor=M_UNROLL" in header
    assert "factor=K_UNROLL" in header
    assert "factor=N_UNROLL" in header
    assert "factor=GRAD_PARTITION" in header
    assert "factor=WEIGHT_PARTITION" in header
    assert "factor=INPUT_PARTITION" in header

    graph = _full_transformer_graph()
    source = emit_top_train_cpp(
        graph=graph,
        top_name="deeplearn_train",
        weights_mode="embedded",
        training_cfg={"loss": {"type": "mse"}, "optimizer": {"learning_rate": 0.01}},
    )
    # Default codegen still emits the complete template argument list, so the
    # HLS header and generated call site cannot drift independently again.
    assert "matmul_backward_left_accumulate<4, 8, 8, grad_act_t, wgt_t, grad_act_t, acc_t, 1, 1, 1, 1, 1, 1>" in source
