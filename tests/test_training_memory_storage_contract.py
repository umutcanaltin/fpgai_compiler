from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml


def _load_training_config() -> dict:
    candidates = [
        Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix/configs/training_kv260_aggressive_fx8_3.yml"),
        Path("paper_experiments/full_pipeline_gate/sprint27h_full_rerun/configs_hls/training_kv260_aggressive_fx8_3.yml"),
        Path("configs/examples/training_compile_smoke.yml"),
    ]
    for p in candidates:
        if p.exists():
            data = yaml.safe_load(p.read_text())
            if isinstance(data, dict):
                return data
    pytest.skip("training config not available")




def _force_dense_training_model(raw: dict) -> None:
    """Use a Dense-only ONNX graph for DDR tiled mutable training tests.

    The default training smoke config is CNN-based in some repo snapshots.
    Sprint 29J intentionally implements Dense DDR tiled mutable training first; Sprint 29K extends the same path to Conv graphs,
    so the Dense positive compile test still uses a Dense-only model.
    """
    candidates = [
        Path("models/suite/mlp_mnist.onnx"),
        Path("models/mlp_mnist.onnx"),
        Path("models/mnist_mlp.onnx"),
    ]
    for model_path in candidates:
        if model_path.exists():
            raw.setdefault("model", {})["format"] = "onnx"
            raw.setdefault("model", {})["path"] = str(model_path)
            return
    pytest.skip("Dense-only ONNX model not available for DDR tiled mutable training test")


def _make_config(raw: dict, tmp_path: Path):
    cfg_path = tmp_path / "compile.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    from fpgai.config.loader import load_config

    return load_config(str(cfg_path))


def _compile_raw(raw: dict, tmp_path: Path):
    pytest.importorskip("onnx")
    from fpgai.engine.compiler import Compiler

    cfg = _make_config(raw, tmp_path)
    return Compiler(cfg).compile()


def test_training_uram_weight_storage_rejects_on_no_uram_board(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_uram_no_uram_board")
    raw.setdefault("targets", {})["board"] = "pynq_z2"

    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("memory", {}).setdefault("storage", {})["weights"] = "uram"
    raw.setdefault("memory", {})["weight_storage"] = "uram"
    raw.setdefault("training", {}).setdefault("storage", {})["weights"] = "uram"

    with pytest.raises(ValueError, match="requires URAM"):
        _compile_raw(raw, tmp_path)


def test_training_ddr_weight_storage_compiles_as_dense_tiled_mutable_backend(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    _force_dense_training_model(raw)
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_ddr_tiled_mutable")

    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("memory", {}).setdefault("storage", {})["weights"] = "ddr"
    raw.setdefault("memory", {})["weight_storage"] = "ddr"
    raw.setdefault("training", {}).setdefault("storage", {})["weights"] = "ddr"
    raw.setdefault("data_movement", {}).setdefault("weights", {})["import"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "tiled",
    }
    raw["data_movement"]["weights"]["export"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "tiled",
    }

    result = _compile_raw(raw, tmp_path)
    assert result is not None

    out_dir = Path(raw["project"]["out_dir"])
    source = (out_dir / "hls/src/deeplearn.cpp").read_text()
    tb = (out_dir / "hls/src/tb.cpp").read_text()
    assert "ap_uint<32>* weights_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in source
    assert "FPGAI training DDR tiled mutable backend" in source
    assert "weight_tile" in source
    assert "grad_tile" in source
    assert "weights_mem[" in source
    assert "weights_mem.data(), 2" in tb



def test_training_ddr_weight_storage_compiles_conv_pool_tiled_mutable_backend(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_ddr_tiled_conv_mutable")

    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("memory", {}).setdefault("storage", {})["weights"] = "ddr"
    raw.setdefault("memory", {})["weight_storage"] = "ddr"
    raw.setdefault("training", {}).setdefault("storage", {})["weights"] = "ddr"
    raw.setdefault("data_movement", {}).setdefault("weights", {})["import"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "tiled",
    }
    raw["data_movement"]["weights"]["export"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "tiled",
    }

    result = _compile_raw(raw, tmp_path)
    assert result is not None

    out_dir = Path(raw["project"]["out_dir"])
    source = (out_dir / "hls/src/deeplearn.cpp").read_text()
    tb = (out_dir / "hls/src/tb.cpp").read_text()
    assert "ap_uint<32>* weights_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in source
    assert "FPGAI training DDR tiled mutable backend" in source
    assert "conv_weight_tile" in source
    assert "conv_grad_tile" in source
    assert "#pragma HLS BIND_STORAGE variable=conv_weight_tile" in source
    assert "weights_mem[" in source
    assert "weights_mem.data(), 2" in tb

def test_training_bram_weight_storage_still_compiles(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_bram_ok")

    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("memory", {}).setdefault("storage", {})["weights"] = "bram"
    raw.setdefault("memory", {})["weight_storage"] = "bram"
    raw.setdefault("training", {}).setdefault("storage", {})["weights"] = "bram"

    raw.get("data_movement", {}).pop("weights", None)
    if "ps_pl" in raw.get("data_movement", {}):
        raw["data_movement"]["ps_pl"].pop("weights", None)

    result = _compile_raw(raw, tmp_path)

    assert result is not None
    out_dir = Path(raw["project"]["out_dir"])
    assert (out_dir / "hls/src/deeplearn.cpp").exists()



def test_training_bram_static_source_has_real_bram_bindings_and_python_compare_artifacts(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_bram_static_real")

    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("memory", {}).setdefault("storage", {})["weights"] = "bram"
    raw.setdefault("memory", {})["weight_storage"] = "bram"
    raw.setdefault("training", {}).setdefault("storage", {})["weights"] = "bram"
    raw.setdefault("data_movement", {}).setdefault("weights", {})["import"] = {
        "interface": "compile_time",
        "transport": "none",
        "policy": "static",
    }
    raw["data_movement"]["weights"]["export"] = {
        "interface": "none",
        "transport": "none",
        "policy": "none",
    }

    result = _compile_raw(raw, tmp_path)
    assert result is not None

    out_dir = Path(raw["project"]["out_dir"])
    source = (out_dir / "hls/src/deeplearn.cpp").read_text()
    assert "FPGAI training weight storage: bram_mutable" in source
    assert "#pragma HLS BIND_STORAGE variable=W_" in source
    assert "impl=bram" in source
    assert "fpgai::sgd_update_wgt" in source
    assert "FPGAI_MODE_RUN_TRAINING" in source
    assert "FPGAI_MODE_EXPORT_WEIGHTS_STREAM" in source
    assert "weights_before_ref.bin" in (out_dir / "manifest.json").read_text()
    assert "weights_after_ref.bin" in (out_dir / "manifest.json").read_text()
    assert "grads_ref.bin" in (out_dir / "manifest.json").read_text()


def test_training_bram_import_export_full_source_has_m_axi_commands_and_testbench_calls(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_bram_import_export")

    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("memory", {}).setdefault("storage", {})["weights"] = "bram"
    raw.setdefault("memory", {})["weight_storage"] = "bram"
    raw.setdefault("training", {}).setdefault("storage", {})["weights"] = "bram"
    raw.setdefault("data_movement", {}).setdefault("weights", {})["import"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "full",
    }
    raw["data_movement"]["weights"]["export"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "full",
    }

    result = _compile_raw(raw, tmp_path)
    assert result is not None

    out_dir = Path(raw["project"]["out_dir"])
    source = (out_dir / "hls/src/deeplearn.cpp").read_text()
    tb = (out_dir / "hls/src/tb.cpp").read_text()
    manifest = (out_dir / "runtime_package/package_manifest.json").read_text()

    assert "ap_uint<32>* weights_mem" in source
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in source
    assert "FPGAI_MODE_IMPORT_WEIGHTS" in source
    assert "FPGAI_MODE_EXPORT_WEIGHTS" in source
    assert "if (mode == FPGAI_MODE_IMPORT_WEIGHTS)" in source
    assert "if (mode == FPGAI_MODE_EXPORT_WEIGHTS)" in source
    assert "#pragma HLS BIND_STORAGE variable=W_" in source
    assert "impl=bram" in source
    assert "hls::stream<axis_t>& aux,\n    ap_uint<32>* weights_mem," in tb
    assert "weights_mem.data(), 0" in tb
    assert "weights_mem.data(), 2" in tb
    assert '"weights_mode": "bram_import_export_full"' in manifest
    assert '"export_supported": true' in manifest
    assert '"reload_before_each_compute": false' in manifest



def test_training_storage_wrapper_inserts_real_bram_pragmas_without_fake_comments() -> None:
    from types import SimpleNamespace
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_insert_training_storage_bindings

    source = """
static wgt_t W_dense0[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
static bias_t B_dense0[2] = { 0.0f, 0.0f };
static grad_wgt_t dW_dense0[4];
static grad_bias_t dB_dense0[2];
extern "C" void deeplearn(
  hls::stream<axis_t>& in,
  hls::stream<axis_t>& out,
  hls::stream<axis_t>& aux,
  int mode
) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE axis port=aux
#pragma HLS INTERFACE s_axilite port=mode bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
}
"""
    plan = SimpleNamespace(notes={"resolved_weight_storage": "bram", "memory_semantics_mode": "bram_static"})
    out = _fpgai_insert_training_storage_bindings(source, compile_plan=plan, memory_plan=None)
    assert "FPGAI training weight storage: bram_mutable" in out
    assert "#pragma HLS BIND_STORAGE variable=W_dense0 type=ram_2p impl=bram" in out
    assert "#pragma HLS BIND_STORAGE variable=B_dense0 type=ram_2p impl=bram" in out
    assert "#pragma HLS BIND_STORAGE variable=dW_dense0 type=ram_2p impl=bram" in out
    assert "requires synthesis-safe local/runtime buffers" not in out
    assert "file-scope BIND_STORAGE disabled" not in out


def test_training_storage_wrapper_inserts_m_axi_import_export_commands() -> None:
    from types import SimpleNamespace
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_insert_training_storage_bindings

    source = """
#include <ap_axi_sdata.h>
static wgt_t W_dense0[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
static bias_t B_dense0[2] = { 0.0f, 0.0f };
static grad_wgt_t dW_dense0[4];
static grad_bias_t dB_dense0[2];
extern "C" void deeplearn(
  hls::stream<axis_t>& in,
  hls::stream<axis_t>& out,
  hls::stream<axis_t>& aux,
  int mode
) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE axis port=aux
#pragma HLS INTERFACE s_axilite port=mode bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
  if (mode == 0) {
    for (int i = 0; i < 4; ++i) W_dense0[i] = (wgt_t)read_f32(aux);
    for (int i = 0; i < 2; ++i) B_dense0[i] = (bias_t)read_f32(aux);
    return;
  }

}
"""
    notes = {
        "resolved_weight_storage": "bram",
        "memory_semantics_mode": "bram_import_export_full",
        "weight_import_interface": "m_axi",
        "weight_import_policy": "full",
        "weight_export_interface": "m_axi",
        "weight_export_policy": "full",
    }
    plan = SimpleNamespace(notes=notes)
    out = _fpgai_insert_training_storage_bindings(source, compile_plan=plan, memory_plan=None)
    assert "ap_uint<32>* weights_mem" in out
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in out
    assert "FPGAI_MODE_IMPORT_WEIGHTS" in out
    assert "FPGAI_MODE_EXPORT_WEIGHTS" in out
    assert "if (mode == FPGAI_MODE_IMPORT_WEIGHTS)" in out
    assert "if (mode == FPGAI_MODE_EXPORT_WEIGHTS)" in out
    assert "weights_mem[0 + i]" in out
    assert "weights_mem[4 + i]" in out
    assert "read_f32(aux)" not in out


def test_training_storage_wrapper_inserts_real_uram_pragmas_with_bram_gradients() -> None:
    from types import SimpleNamespace
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_insert_training_storage_bindings

    source = """
static wgt_t W_dense0[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
static bias_t B_dense0[2] = { 0.0f, 0.0f };
static grad_wgt_t dW_dense0[4];
static grad_bias_t dB_dense0[2];
extern "C" void deeplearn(
  hls::stream<axis_t>& in,
  hls::stream<axis_t>& out,
  hls::stream<axis_t>& aux,
  int mode
) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE axis port=aux
#pragma HLS INTERFACE s_axilite port=mode bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
}
"""
    notes = {
        "resolved_weight_storage": "uram",
        "memory_semantics_mode": "uram_static",
        "resolved_gradient_storage": "bram",
    }
    plan = SimpleNamespace(notes=notes)
    out = _fpgai_insert_training_storage_bindings(source, compile_plan=plan, memory_plan=None)
    assert "FPGAI training weight storage: uram_mutable" in out
    assert "#pragma HLS BIND_STORAGE variable=W_dense0 type=ram_2p impl=uram" in out
    assert "#pragma HLS BIND_STORAGE variable=B_dense0 type=ram_2p impl=uram" in out
    assert "#pragma HLS BIND_STORAGE variable=dW_dense0 type=ram_2p impl=bram" in out
    assert "#pragma HLS BIND_STORAGE variable=dB_dense0 type=ram_2p impl=bram" in out


def test_training_storage_wrapper_inserts_uram_m_axi_import_export_commands() -> None:
    from types import SimpleNamespace
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_insert_training_storage_bindings

    source = """
#include <ap_axi_sdata.h>
static wgt_t W_dense0[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
static bias_t B_dense0[2] = { 0.0f, 0.0f };
static grad_wgt_t dW_dense0[4];
static grad_bias_t dB_dense0[2];
extern "C" void deeplearn(
  hls::stream<axis_t>& in,
  hls::stream<axis_t>& out,
  hls::stream<axis_t>& aux,
  int mode
) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE axis port=aux
#pragma HLS INTERFACE s_axilite port=mode bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
  if (mode == 0) {
    for (int i = 0; i < 4; ++i) W_dense0[i] = (wgt_t)read_f32(aux);
    for (int i = 0; i < 2; ++i) B_dense0[i] = (bias_t)read_f32(aux);
    return;
  }

}
"""
    notes = {
        "resolved_weight_storage": "uram",
        "memory_semantics_mode": "uram_import_export_full",
        "weight_import_interface": "m_axi",
        "weight_import_policy": "full",
        "weight_export_interface": "m_axi",
        "weight_export_policy": "full",
        "resolved_gradient_storage": "bram",
    }
    plan = SimpleNamespace(notes=notes)
    out = _fpgai_insert_training_storage_bindings(source, compile_plan=plan, memory_plan=None)
    assert "ap_uint<32>* weights_mem" in out
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in out
    assert "FPGAI_MODE_IMPORT_WEIGHTS" in out
    assert "FPGAI_MODE_EXPORT_WEIGHTS" in out
    assert "if (mode == FPGAI_MODE_IMPORT_WEIGHTS)" in out
    assert "if (mode == FPGAI_MODE_EXPORT_WEIGHTS)" in out
    assert "#pragma HLS BIND_STORAGE variable=W_dense0 type=ram_2p impl=uram" in out
    assert "#pragma HLS BIND_STORAGE variable=dW_dense0 type=ram_2p impl=bram" in out
    assert "read_f32(aux)" not in out


def test_user_facing_weights_mode_tiled_mutable_expands_to_training_ddr_semantics() -> None:
    from fpgai.engine.compiler import Compiler
    from fpgai.engine.planner import _choose_weight_mode

    compiler = object.__new__(Compiler)
    raw = {
        "pipeline": {"mode": "training_on_device"},
        "memory": {"storage": {"weights": "ddr"}},
        "weights": {"mode": "tiled_mutable"},
    }

    assert _choose_weight_mode(None, raw) == "ddr"
    semantics = compiler._resolve_weight_movement_semantics(raw)
    assert semantics["memory_semantics_mode"] == "ddr_tiled_mutable"
    assert semantics["hls_weights_mode"] == "ddr_tiled_mutable"
    assert semantics["weight_import_interface"] == "m_axi"
    assert semantics["weight_import_policy"] == "tiled"
    assert semantics["weight_export_interface"] == "m_axi"
    assert semantics["weight_export_policy"] == "tiled"
    assert semantics["tile_weight_buffer"] is True


def test_user_facing_weights_mode_tiled_mutable_rejects_inference_pipeline() -> None:
    import pytest
    from fpgai.engine.compiler import Compiler

    compiler = object.__new__(Compiler)
    with pytest.raises(ValueError, match="tiled_mutable is a training weight mode"):
        compiler._resolve_weight_movement_semantics(
            {
                "pipeline": {"mode": "inference"},
                "memory": {"storage": {"weights": "ddr"}},
                "weights": {"mode": "tiled_mutable"},
            }
        )
