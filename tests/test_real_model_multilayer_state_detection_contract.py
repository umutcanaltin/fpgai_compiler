from __future__ import annotations

from fpgai.analysis.model_gap import audit_model_gaps
from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
from fpgai.ir.graph import Graph
from fpgai.ir.passes.detection_lowering import plan_detection_output
from fpgai.ir.passes.transformer_lowering import (
    configure_kv_cache_state,
    plan_layered_token_decoding,
)
from fpgai.runtime.runtime_plans import build_persistent_state_plan


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


def test_layered_decode_assigns_unique_kv_ownership_groups() -> None:
    g = Graph("layered")
    for layer in range(3):
        g.add_tensor(f"k{layer}", (1, 8, 4), "float32")
        g.add_tensor(f"v{layer}", (1, 8, 4), "float32")
    plan = plan_layered_token_decoding(
        g,
        layer_caches=[
            {"key": "k0", "value": "v0"},
            {"key": "k1", "value": "v1", "storage": "uram"},
            {"key": "k2", "value": "v2"},
        ],
        max_sequence_length=8,
    )
    assert plan["layer_count"] == 3
    assert plan["cursor_policy"] == "independent_per_state_group"
    assert g.get_tensor("k1").semantics.state.owner == "transformer.layer.1"
    assert g.get_tensor("v1").semantics.state.state_group == "kv.layer.1"
    assert g.get_tensor("k2").semantics.state.overflow_policy == "saturate"

    state = build_persistent_state_plan(g)
    by_name = {row["name"]: row for row in state["tensors"]}
    assert by_name["k0"]["owner"] == "transformer.layer.0"
    assert by_name["v2"]["state_group"] == "kv.layer.2"
    assert by_name["k2"]["overflow_policy"] == "saturate"


def test_unsupported_cache_overflow_policy_rejects_explicitly() -> None:
    g = Graph("overflow_state")
    g.add_tensor("cache", (1, 4, 4), "float32")
    try:
        configure_kv_cache_state(
            g, key_cache="cache", value_cache="cache", capacity=4,
            sequence_axis=1, storage="bram", overflow_policy="wrap",
        )
    except ValueError as exc:
        assert "overflow_policy=saturate only" in str(exc)
    else:
        raise AssertionError("unsupported KV-cache overflow policy must not be accepted silently")

def test_model_gap_does_not_call_supported_onchip_state_a_blocker() -> None:
    g = Graph("state_gap")
    g.add_tensor("cache", (1, 8, 4), "float32")
    configure_kv_cache_state(
        g,
        key_cache="cache",
        value_cache="cache",
        capacity=8,
        sequence_axis=1,
        storage="bram",
    )
    report = audit_model_gaps(g, pipeline_mode="inference")
    assert report["runtime_state_requirements"][0]["hls_on_chip_state_supported"] is True
    assert report["runtime_state_blockers"] == []


def test_detection_output_plan_is_generic_runtime_contract() -> None:
    g = Graph("detector")
    g.outputs = ["predictions"]
    g.add_tensor("predictions", (1, 84, 8400), "float32")
    plan = plan_detection_output(
        g,
        output_tensor="predictions",
        class_count=80,
        box_format="xywh",
        pyramid_strides=[8, 16, 32],
        postprocess_partition="ps_or_host",
    )
    assert plan.class_count == 80
    assert plan.pyramid_strides == (8, 16, 32)
    assert "detection_output" in g.semantics.runtime_contract
    report = audit_model_gaps(g, pipeline_mode="inference")
    assert report["detection_output_contract"]["output_tensor"] == "predictions"
    assert report["detection_output_contract"]["postprocess_partition"] == "ps_or_host"


def test_runtime_package_preserves_graph_runtime_contract(tmp_path) -> None:
    from fpgai.runtime.package_builder import emit_runtime_package

    out = tmp_path / "compile"
    out.mkdir()
    contract = {
        "layered_kv_cache": {"schema": "fpgai.layered-token-decoding-plan/v1", "layer_count": 30},
        "detection_output": {"schema": "fpgai.detection-output-plan/v1", "output_tensor": "predictions"},
    }
    result = emit_runtime_package(out, graph_runtime_contract=contract)
    manifest = __import__("json").loads((out / "runtime_package" / "package_manifest.json").read_text())
    assert manifest["graph_runtime_contract"] == contract
    api = (out / "runtime_package" / "runtime_api.py").read_text()
    assert "def load_graph_runtime_contract()" in api
    assert result["graph_runtime_contract"] == contract


def test_detection_box_math_accepts_static_grid_stride_broadcasts() -> None:
    import numpy as np

    g = Graph("box_math")
    g.inputs = ["dist"]
    g.outputs = ["scaled"]
    g.add_tensor("dist", (1, 4, 3), "float32")
    g.add_tensor("grid", (1, 1, 3), "float32")
    g.constants["grid"] = np.asarray([[[0.5, 1.5, 2.5]]], dtype=np.float32)
    g.add_tensor("shifted", (1, 4, 3), "float32")
    g.add_tensor("stride", (1, 1, 3), "float32")
    g.constants["stride"] = np.asarray([[[8.0, 16.0, 32.0]]], dtype=np.float32)
    g.add_tensor("scaled", (1, 4, 3), "float32")
    g.add_op("Add", ["dist", "grid"], ["shifted"], name="grid_add")
    g.add_op("Mul", ["shifted", "stride"], ["scaled"], name="stride_mul")
    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg=_cfg())
    assert "fpgai_add_const_0[12]" in source
    assert "add_vec_typed<12" in source
    assert "fpgai_mul_scale_1[3]" in source
    assert "mul_rows_by_col_vector<4, 3" in source
