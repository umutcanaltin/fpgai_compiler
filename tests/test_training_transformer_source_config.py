from __future__ import annotations

from pathlib import Path

import yaml

from fpgai.analysis.model_inspection import inspect_config
from fpgai.config.loader import load_config
from fpgai.engine.compiler import Compiler
from fpgai.frontend import import_model_source


def test_training_transformer_source_is_expanded_layerwise_mlir() -> None:
    graph = import_model_source(
        "examples/reference/tiny_transformer_training.mlir",
        format_hint="mlir",
        pipeline_mode="training_on_device",
        target_board="kv260",
    )
    types = [op.op_type for op in graph.ops]
    assert "TransformerBlock" not in types
    assert types == [
        "RMSNorm", "MatMul", "MatMul", "MatMul", "RotaryEmbedding", "RotaryEmbedding",
        "MultiHeadAttention", "MatMul", "Add", "RMSNorm", "MatMul", "MatMul", "SiLU",
        "Mul", "MatMul", "Add",
    ]


def test_training_transformer_config_is_source_driven_and_hardware_complete() -> None:
    cfg = load_config("configs/examples/training_transformer_layerwise.yml")
    inspection = inspect_config(cfg)
    assert inspection.compilation_ready is True
    assert inspection.gap_audit["unsupported_operator_types"] == []
    assert inspection.gap_audit["training_capability_audit"]["hardware_complete"] is True


def test_training_transformer_config_emits_generic_hls_project_without_vitis(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/examples/training_transformer_layerwise.yml").read_text(encoding="utf-8"))
    raw["project"]["out_dir"] = str(tmp_path / "build")
    raw["backends"]["hls"]["vitis"]["enabled"] = False
    raw["toolchain"]["vitis_hls"]["enabled"] = False
    cfg_path = tmp_path / "compile.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = Compiler(load_config(str(cfg_path))).compile()
    assert result.hls_ran is False
    assert (result.out_dir / "manifest.json").is_file()
    assert (result.out_dir / "ir" / "training_capability_audit.json").is_file()
    assert (result.out_dir / "ir" / "layer_mechanism_resolution.json").is_file()

    hls_sources = list((result.out_dir / "hls").rglob("*.cpp"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in hls_sources)
    assert "multi_head_attention_backward_serialized" in text
    assert "rms_norm_backward_rows" in text
    assert "silu_backward_accumulate" in text
    assert "matmul_weight_grad" in text


def test_training_transformer_network_layer_and_loop_controls_reach_hls(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/examples/training_transformer_layerwise.yml").read_text(encoding="utf-8"))
    raw["project"]["out_dir"] = str(tmp_path / "knob_build")
    raw["backends"]["hls"]["vitis"]["enabled"] = False
    raw["toolchain"]["vitis_hls"]["enabled"] = False
    raw["architecture"]["network"] = {
        "pipeline": {"ii": 3, "loops": {"element": 3}},
        "parallelism": {"pe": 1, "simd": 1, "unroll": {"element": 2}},
        "partitioning": {"factor": 2, "mode": "cyclic", "targets": {"input": 2, "output": 2, "weight": 2, "gradient": 2}},
    }
    raw["architecture"]["layers"].append({
        "match": {"index": 1},
        "pipeline": {"ii": 4, "loops": {"k": 5}},
        "parallelism": {"pe": 2, "simd": 2, "unroll": {"m": 1, "n": 2, "k": 2}},
        "partitioning": {"factor": 2, "mode": "cyclic", "targets": {"input": 2, "output": 2, "weight": 4, "gradient": 2}},
        "tiling": {"sizes": {"m": 1, "n": 2, "k": 2}},
    })
    raw["architecture"]["layers"].append({
        "match": {"op_type": "MultiHeadAttention"},
        "pipeline": {"ii": 2},
        "parallelism": {"unroll": {"head": 1, "k": 2}},
    })
    cfg_path = tmp_path / "knobs.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = Compiler(load_config(str(cfg_path))).compile()
    source = (result.out_dir / "hls" / "src" / "deeplearn.cpp").read_text(encoding="utf-8")
    # MatMul op index 1: tile 1/2/2, k-loop II=5, m/n/k unroll 1/2/2,
    # input/output/weight partition 2/2/4.
    assert "matmul_tiled<4, 8, 8, act_t, wgt_t, act_t, acc_t, 1, 2, 2, 5, 1, 2, 2, 2, 2, 4>" in source
    # Network-wide element controls reach SiLU and Add; layer MHA override wins.
    assert "silu_vector<64, act_t, act_t, acc_t, 3, 2, 2, 2>" in source
    assert "add_vec_typed<32, act_t, act_t, act_t, acc_t, 3, 2, 2, 2>" in source
    assert "multi_head_attention_serialized<4, 8, 2, act_t, act_t, acc_t, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2>" in source
    # Backward path consumes the same resolved controls.
    assert "matmul_weight_grad<4, 8, 8, act_t, grad_act_t, grad_wgt_t, acc_t, 4, 1, 2, 2, 2, 2>" in source
    assert "silu_backward_accumulate<64, act_t, grad_act_t, grad_act_t, acc_t, 3, 2>" in source
