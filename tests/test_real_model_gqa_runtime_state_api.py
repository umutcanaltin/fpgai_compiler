from __future__ import annotations

from pathlib import Path

from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.backends.hls.emit.layers_attention import emit_attention_h
from fpgai.ir.graph import Graph
from fpgai.ir.passes.transformer_lowering import plan_transformer_execution
from fpgai.runtime.package_builder import emit_runtime_package


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


def test_cached_gqa_maps_query_heads_to_shared_kv_heads() -> None:
    g = Graph("gqa_cached")
    g.inputs = ["q", "k", "v", "valid_length"]
    g.outputs = ["context"]
    # SmolLM2-like ratio: 9 query heads, 3 KV heads, head dim 64.
    g.add_tensor("q", (1, 1, 576), "float32")
    g.add_tensor("k", (1, 16, 192), "float32")
    g.add_tensor("v", (1, 16, 192), "float32")
    g.add_tensor("valid_length", (1,), "int32")
    g.add_tensor("context", (1, 1, 576), "float32")
    g.add_op(
        "MultiHeadAttention",
        ["q", "k", "v", "valid_length"],
        ["context"],
        name="gqa",
        attrs={"num_heads": 9, "num_kv_heads": 3, "causal": True},
    )
    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "multi_head_attention_cached_serialized<1, 16, 576, 9, 3" in source
    header = emit_attention_h()
    assert "const int Q_PER_KV = HEADS / KV_HEADS" in header
    assert "const int kv_head = head / Q_PER_KV" in header
    assert "col * KV_MODEL + kv_head * HEAD_DIM + d" in header


def test_transformer_plan_records_num_kv_heads() -> None:
    g = Graph("plan")
    g.add_tensor("q", (1, 1, 576), "float32")
    g.add_tensor("k", (1, 1, 192), "float32")
    g.add_tensor("v", (1, 1, 192), "float32")
    g.add_tensor("o", (1, 1, 576), "float32")
    g.add_op("MultiHeadAttention", ["q", "k", "v"], ["o"], name="attn", attrs={})
    plans = plan_transformer_execution(g, model_dimension=576, num_heads=9, num_kv_heads=3, max_sequence_length=8192)
    assert plans[0].num_kv_heads == 3
    assert plans[0].to_dict()["num_kv_heads"] == 3
    assert g.ops[0].attrs["num_kv_heads"] == 3


def test_generated_runtime_api_exposes_persistent_state_control(tmp_path: Path) -> None:
    out = tmp_path / "build"
    out.mkdir()
    state = {
        "schema": "fpgai.persistent-state-plan/v1",
        "tensor_count": 1,
        "backend_required": True,
        "required_operations": ["reset", "import", "export", "read", "write"],
        "tensors": [{"name": "k_cache", "kind": "kv_key_cache"}],
    }
    emit_runtime_package(out, pipeline_mode="inference", top_name="deeplearn", persistent_state_plan=state)
    api = (out / "runtime_package" / "runtime_api.py").read_text(encoding="utf-8")
    assert "def reset_state(" in api
    assert "def import_state(" in api
    assert "def export_state(" in api
    assert "def read_state(" in api
    assert "def write_state(" in api
    assert "hasattr(_BOUND_BACKEND, 'reset_state')" in api
    assert "hasattr(_BOUND_BACKEND, 'import_state')" in api
    assert "hasattr(_BOUND_BACKEND, 'export_state')" in api
    assert "hasattr(_BOUND_BACKEND, 'read_state')" in api
    assert "hasattr(_BOUND_BACKEND, 'write_state')" in api
    board = (out / "runtime_package" / "board_runtime.py").read_text(encoding="utf-8")
    assert "def reset_state(" in board
    assert "def import_state(" in board
    assert "def export_state(" in board
    assert "def read_state(" in board
    assert "def write_state(" in board
