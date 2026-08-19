from __future__ import annotations

import json
import numpy as np

from fpgai.analysis.model_gap import audit_model_gaps
from fpgai.ir.graph import Graph
from fpgai.ir.passes.detection_lowering import plan_detection_decode, plan_detection_output
from fpgai.ir.passes.transformer_lowering import plan_autoregressive_runtime
from fpgai.runtime.package_builder import emit_runtime_package


def _const(g: Graph, name: str, shape: tuple[int, ...], value: float = 0.1) -> None:
    g.add_tensor(name, shape, "float32")
    g.constants[name] = np.full(shape, value, dtype=np.float32)


def test_autoregressive_prefill_decode_contract_preserves_layered_state_and_tied_parameters() -> None:
    g = Graph("autoregressive")
    for layer in range(3):
        g.add_tensor(f"k{layer}", (1, 16, 4), "float32")
        g.add_tensor(f"v{layer}", (1, 16, 4), "float32")
    _const(g, "embedding", (32, 12))
    _const(g, "lm_head", (12, 32))

    plan = plan_autoregressive_runtime(
        g,
        layer_caches=[
            {"key": "k0", "value": "v0"},
            {"key": "k1", "value": "v1"},
            {"key": "k2", "value": "v2"},
        ],
        max_sequence_length=16,
        prefill_sequence_length=8,
        decode_sequence_length=1,
        cache_storage="uram",
        tied_parameter_groups=[{
            "name": "token_embedding_output_projection",
            "members": [
                {"tensor": "embedding", "view": "native"},
                {"tensor": "lm_head", "view": "transpose"},
            ],
        }],
    )
    assert plan.supported_modes == ("prefill", "decode")
    assert plan.layer_count == 3
    assert plan.position_source == "persistent_state_cursor"
    assert g.semantics.runtime_contract["autoregressive_session"]["reset_state_on_prefill"] is True
    assert g.get_tensor("embedding").semantics.tags[-2] == "tied_parameter"
    assert g.get_tensor("k1").semantics.state.owner == "transformer.layer.1"

    audit = audit_model_gaps(g, pipeline_mode="inference")
    assert audit["autoregressive_runtime_contract"]["prefill_sequence_length"] == 8
    assert audit["runtime_state_blockers"] == []


def test_detection_decode_contract_and_raw_output_contract_coexist() -> None:
    g = Graph("detector")
    g.outputs = ["predictions"]
    g.add_tensor("distances", (1, 4, 8400), "float32")
    g.add_tensor("boxes", (1, 4, 8400), "float32")
    g.add_tensor("predictions", (1, 84, 8400), "float32")
    decode = plan_detection_decode(
        g,
        distance_tensor="distances",
        decoded_box_tensor="boxes",
        dfl_bins=16,
        pyramid_strides=[8, 16, 32],
        grid_origin=0.5,
        input_box_format="ltrb",
        output_box_format="xywh",
    )
    output = plan_detection_output(
        g,
        output_tensor="predictions",
        class_count=80,
        box_format="xywh",
        pyramid_strides=[8, 16, 32],
        postprocess_partition="ps_or_host",
    )
    assert decode.dfl_bins == 16
    assert output.postprocess_partition == "ps_or_host"
    audit = audit_model_gaps(g, pipeline_mode="inference")
    assert audit["detection_decode_contract"]["pyramid_strides"] == [8, 16, 32]
    assert audit["detection_output_contract"]["class_count"] == 80


def test_runtime_package_exposes_prefill_decode_and_detection_postprocess_api(tmp_path) -> None:
    out = tmp_path / "compile"
    out.mkdir()
    contract = {
        "autoregressive_session": {
            "schema": "fpgai.autoregressive-runtime-plan/v1",
            "supported_modes": ["prefill", "decode"],
            "reset_state_on_prefill": True,
        },
        "detection_output": {
            "schema": "fpgai.detection-output-plan/v1",
            "postprocess_partition": "ps_or_host",
        },
    }
    emit_runtime_package(
        out,
        persistent_state_plan={"schema": "fpgai.persistent-state-plan/v1", "tensor_count": 2},
        graph_runtime_contract=contract,
    )
    api = (out / "runtime_package" / "runtime_api.py").read_text()
    manifest = json.loads((out / "runtime_package" / "package_manifest.json").read_text())
    assert "def prepare_prefill(" in api
    assert "def prepare_decode(" in api
    assert "def postprocess_detections(" in api
    for name in ("prepare_prefill", "prepare_decode", "postprocess_detections"):
        assert name in manifest["runtime_api"]["functions"]


def test_tied_embedding_and_lm_head_share_one_physical_hls_constant() -> None:
    from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp

    g = Graph("tied_decode_projection")
    g.inputs = ["token"]
    g.outputs = ["logits"]
    g.add_tensor("token", (1,), "int64")
    embedding = np.arange(32, dtype=np.float32).reshape(8, 4) / 32.0
    g.add_tensor("embedding", (8, 4), "float32")
    g.constants["embedding"] = embedding
    g.add_tensor("hidden", (1, 4), "float32")
    g.add_op("Gather", ["embedding", "token"], ["hidden"], name="embedding_lookup", attrs={"axis": 0})
    g.add_tensor("lm_head", (4, 8), "float32")
    g.constants["lm_head"] = embedding.T.copy()
    g.add_tensor("logits", (1, 8), "float32")
    g.add_op("MatMul", ["hidden", "lm_head"], ["logits"], name="output_projection")

    plan_autoregressive_runtime(
        g,
        layer_caches=[],
        max_sequence_length=8,
        prefill_sequence_length=1,
        tied_parameter_groups=[{
            "name": "token_embedding_output_projection",
            "members": [
                {"tensor": "embedding", "view": "native"},
                {"tensor": "lm_head", "view": "transpose"},
            ],
        }],
    )
    source = emit_dag_top_cpp(g, top_name="deeplearn", weights_mode="embedded", raw_cfg={
        "numerics": {"defaults": {"activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6}, "accum": {"type": "ap_fixed", "total_bits": 32, "int_bits": 12}}},
        "targets": {"hls": {"control_protocol": "s_axilite"}},
    })
    assert source.count("static const ap_fixed<32,12") == 1
    assert "FPGAI_TIED_PARAMETER owner=embedding physical_symbol=fpgai_tied_parameter_0" in source
    assert "gather_rows<8, 4, 1" in source and "fpgai_tied_parameter_0" in source
    assert "matmul_tiled_right_transposed<1, 4, 8" in source
    assert "fpgai_matmul_right_1" not in source


def test_tied_parameter_contract_rejects_nonmatching_duplicate_values() -> None:
    g = Graph("bad_tied")
    _const(g, "embedding", (8, 4), 0.1)
    _const(g, "lm_head", (4, 8), 0.2)
    try:
        plan_autoregressive_runtime(
            g,
            layer_caches=[],
            max_sequence_length=8,
            prefill_sequence_length=1,
            tied_parameter_groups=[{
                "members": [
                    {"tensor": "embedding", "view": "native"},
                    {"tensor": "lm_head", "view": "transpose"},
                ],
            }],
        )
    except ValueError as exc:
        assert "IRLLM018" in str(exc)
    else:
        raise AssertionError("mismatched tied parameter values must not be silently deduplicated")
