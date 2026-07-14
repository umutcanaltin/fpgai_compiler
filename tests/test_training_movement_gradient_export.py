from __future__ import annotations

import copy
import json
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
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    pytest.skip("training config not available")


def _make_config(raw: dict, tmp_path: Path):
    cfg_path = tmp_path / "compile.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    from fpgai.config.loader import load_config

    return load_config(str(cfg_path))


def _compile_raw(raw: dict, tmp_path: Path):
    pytest.importorskip("onnx")
    from fpgai.engine.compiler import Compiler

    return Compiler(_make_config(raw, tmp_path)).compile()


def test_training_m_axi_tiled_inputs_labels_outputs_generate_ports_and_reports(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_tiled_io")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    dm = raw.setdefault("data_movement", {})
    dm.setdefault("inputs", {})["import"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"}
    dm.setdefault("labels", {})["import"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"}
    dm.setdefault("outputs", {})["export"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"}

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "ap_uint<32>* input_mem" in source
    assert "ap_uint<32>* label_mem" in source
    assert "ap_uint<32>* output_mem" in source
    assert "m_axi port=input_mem" in source
    assert "m_axi port=label_mem" in source
    assert "m_axi port=output_mem" in source
    assert "input_tile[FPGAI_TRAIN_INPUT_TILE_SIZE]" in source
    assert "label_tile[FPGAI_TRAIN_LABEL_TILE_SIZE]" in source
    assert "output_tile[FPGAI_TRAIN_OUTPUT_TILE_SIZE]" in source

    report = json.loads((out_dir / "reports/training_io_movement.json").read_text(encoding="utf-8"))
    assert report["inputs"]["import"]["resolved"] == "m_axi_import_tiled"
    assert report["labels"]["import"]["resolved"] == "m_axi_import_tiled"
    assert report["outputs"]["export"]["resolved"] == "m_axi_export_tiled"

    # The training/Python comparison fixture path remains present.
    # The canonical location is the training_reference directory used by csim_train_tcl.
    ref_dir = out_dir / "training_reference"
    assert (ref_dir / "weights_before_ref.bin").exists()
    assert (ref_dir / "grads_ref.bin").exists()
    assert (ref_dir / "weights_after_ref.bin").exists()
    assert (ref_dir / "tiled_inputs_ref.bin").exists()
    assert (ref_dir / "tiled_labels_ref.bin").exists()
    assert (ref_dir / "tiled_outputs_ref.bin").exists()
    assert (ref_dir / "tiled_gradients_ref.bin").exists()
    assert (ref_dir / "tiled_weights_after_ref.bin").exists()

    numeric = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))
    tiled = numeric["training_tiled_io"]
    assert tiled["requested"] is True
    assert tiled["interface"] == "m_axi"
    assert tiled["input_tiled"] is True
    assert tiled["labels_tiled"] is True
    assert tiled["output_tiled"] is True
    assert tiled["compute_fused"] is True
    assert tiled["reference_available"] is True
    assert tiled["captures_available"] is False
    assert tiled["status"] == "artifact_missing"
    assert tiled["passed"] is False


def test_training_gradient_export_full_generates_m_axi_port_command_and_runtime_support(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_gradient_export")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("data_movement", {}).setdefault("gradients", {})["export"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "full",
    }
    raw.setdefault("runtime", {})["sequence"] = [
        {"run_training": {"steps": 1}},
        "export_gradients",
    ]

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "ap_uint<32>* gradients_mem" in source
    assert "m_axi port=gradients_mem" in source
    assert "FPGAI_MODE_EXPORT_GRADIENTS" in source
    assert "if (mode == FPGAI_MODE_EXPORT_GRADIENTS)" in source

    report = json.loads((out_dir / "reports/gradient_export.json").read_text(encoding="utf-8"))
    assert report["resolved"] == "m_axi_export_full"
    assert report["supported"] is True

    manifest = json.loads((out_dir / "runtime_package/package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_sequence"]["supported_commands"]["export_gradients"] is True
    assert manifest["runtime_sequence"]["sequence"][-1]["command"] == "export_gradients"


def test_training_gradient_export_tiled_generates_tile_buffer_command_and_runtime_support(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_gradient_export_tiled")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("data_movement", {}).setdefault("gradients", {})["export"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "tiled",
    }
    raw.setdefault("runtime", {})["sequence"] = [
        {"run_training": {"steps": 1}},
        "export_gradients",
    ]

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "ap_uint<32>* gradients_mem" in source
    assert "m_axi port=gradients_mem" in source
    assert "FPGAI_MODE_EXPORT_GRADIENTS" in source
    assert "gradient_export_tile[FPGAI_GRADIENT_EXPORT_TILE_SIZE]" in source
    assert "tile_base" in source
    assert "gradient_export tiled mode" in source

    report = json.loads((out_dir / "reports/gradient_export.json").read_text(encoding="utf-8"))
    assert report["resolved"] == "m_axi_export_tiled"
    assert report["supported"] is True

    manifest = json.loads((out_dir / "runtime_package/package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_sequence"]["supported_commands"]["export_gradients"] is True
    assert manifest["runtime_sequence"]["sequence"][-1]["command"] == "export_gradients"



def test_training_axi_stream_tiled_inputs_labels_outputs_generate_real_tile_readers_and_writers(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_axis_tiled_io")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    dm = raw.setdefault("data_movement", {})
    dm.setdefault("inputs", {})["import"] = {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}
    dm.setdefault("labels", {})["import"] = {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}
    dm.setdefault("outputs", {})["export"] = {"interface": "axi_stream", "transport": "dma", "policy": "tiled"}

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI training AXI-stream tiled input import" in source
    assert "FPGAI training AXI-stream tiled label import" in source
    assert "FPGAI training AXI-stream tiled output export" in source
    assert "FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE" in source
    assert "FPGAI_TRAIN_AXIS_LABEL_TILE_SIZE" in source
    assert "FPGAI_TRAIN_AXIS_OUTPUT_TILE_SIZE" in source
    assert "axis_input_tile[FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE]" in source
    assert "axis_label_tile[FPGAI_TRAIN_AXIS_LABEL_TILE_SIZE]" in source
    assert "emit_stream_tiled_block" in source
    assert "write_f32(out, axis_output_tile[lane], last)" in source

    report = json.loads((out_dir / "reports/stream_tiled_io.json").read_text(encoding="utf-8"))
    assert report["inputs"]["import"]["status"] == "generated_interface_supported"
    assert report["outputs"]["export"]["status"] == "generated_interface_supported"

    numeric = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))
    tiled = numeric["training_tiled_io"]
    assert tiled["requested"] is True
    assert tiled["interface"] == "axi_stream"
    assert tiled["axi_stream"]["tlast_required"] is True
    assert tiled["reference_available"] is True
    assert tiled["status"] == "artifact_missing"



def test_normal_training_omits_tiled_io_markers_and_reports_not_requested(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_no_tiled_io")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    # Force plain movement so stale defaults cannot request tiled training I/O.
    dm = raw.setdefault("data_movement", {})
    dm.setdefault("inputs", {})["import"] = {"interface": "none", "policy": "none"}
    dm.setdefault("labels", {})["import"] = {"interface": "none", "policy": "none"}
    dm.setdefault("outputs", {})["export"] = {"interface": "none", "policy": "none"}

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI training tiled input import" not in source
    assert "FPGAI training tiled label import" not in source
    assert "FPGAI training tiled output export" not in source
    assert "FPGAI training AXI-stream tiled input import" not in source
    assert "FPGAI training AXI-stream tiled label import" not in source
    assert "FPGAI training AXI-stream tiled output export" not in source

    ref_dir = out_dir / "training_reference"
    assert not (ref_dir / "tiled_inputs_ref.bin").exists()
    assert not (ref_dir / "tiled_labels_ref.bin").exists()
    assert not (ref_dir / "tiled_outputs_ref.bin").exists()

    numeric = json.loads((out_dir / "reports/numeric_validation.json").read_text(encoding="utf-8"))
    assert numeric["training_tiled_io"]["requested"] is False
    assert numeric["training_tiled_io"]["status"] == "not_requested"


def test_p2t_training_m_axi_tile_sizes_and_real_load_store_loops_follow_yaml() -> None:
    from fpgai.backends.hls.emit.top_train_cpp import _fpgai_p2t_materialize_training_m_axi_tiles

    source = "\n".join(
        [
            "using namespace fpgai;",
            "#ifndef FPGAI_TRAIN_INPUT_TILE_SIZE",
            "#define FPGAI_TRAIN_INPUT_TILE_SIZE 64",
            "#endif",
            "#ifndef FPGAI_TRAIN_LABEL_TILE_SIZE",
            "#define FPGAI_TRAIN_LABEL_TILE_SIZE 64",
            "#endif",
            "#ifndef FPGAI_TRAIN_OUTPUT_TILE_SIZE",
            "#define FPGAI_TRAIN_OUTPUT_TILE_SIZE 64",
            "#endif",
            "#ifndef FPGAI_GRADIENT_EXPORT_TILE_SIZE",
            "#define FPGAI_GRADIENT_EXPORT_TILE_SIZE 64",
            "#endif",
            "extern \"C\" void deeplearn(",
            "  hls::stream<axis_t>& in,",
            "  hls::stream<axis_t>& out,",
            "  hls::stream<axis_t>& aux,",
            "  ap_uint<32>* input_mem,",
            "  ap_uint<32>* label_mem,",
            "  ap_uint<32>* output_mem,",
            "  ap_uint<32>* gradients_mem,",
            "  int mode",
            ") {",
            "#pragma HLS INTERFACE s_axilite port=return bundle=CTRL",
            "  static act_t input_tile[FPGAI_TRAIN_INPUT_TILE_SIZE];",
            "  static act_t label_tile[FPGAI_TRAIN_LABEL_TILE_SIZE];",
            "  static act_t output_tile[FPGAI_TRAIN_OUTPUT_TILE_SIZE];",
            "  static act_t target_buf[2];",
            "  for (int i = 0; i < 8; ++i) buf_input[i] = (act_t)read_f32(in);",
            "  for (int i = 0; i < 2; ++i) target_buf[i] = (act_t)read_f32(aux);",
            "  acc_t difference = (acc_t)buf_linear_1[i] - (acc_t)target_buf[i];",
            "  loss_t loss_value = (loss_t)0;",
            "}",
        ]
    )
    cfg = {
        "data_movement": {
            "inputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 32}},
            "labels": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 32}},
            "outputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 32}},
            "gradients": {"export": {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "tile_size": 32}},
        }
    }

    updated = _fpgai_p2t_materialize_training_m_axi_tiles(source, raw_cfg=cfg)

    assert "#define FPGAI_TRAIN_INPUT_TILE_SIZE 32" in updated
    assert "#define FPGAI_TRAIN_LABEL_TILE_SIZE 32" in updated
    assert "#define FPGAI_TRAIN_OUTPUT_TILE_SIZE 32" in updated
    assert "#define FPGAI_GRADIENT_EXPORT_TILE_SIZE 32" in updated
    assert "FPGAI real training m_axi tiled input import" in updated
    assert "input_mem[idx].to_uint()" in updated
    assert "FPGAI real training m_axi tiled label import" in updated
    assert "label_mem[idx].to_uint()" in updated
    assert "FPGAI real training m_axi tiled output export" in updated
    assert "output_mem[idx] = f32_to_u32((float)output_tile[lane])" in updated
    assert "read_f32(in)" not in updated
    assert "read_f32(aux)" not in updated


def test_training_testbench_packs_m_axi_records_before_train_and_loss_calls(tmp_path: Path) -> None:
    from fpgai.backends.hls.testbench_train import emit_tb_train_cpp

    tb_dir = tmp_path / "tb"
    tb_dir.mkdir()
    raw_cfg = {
        "data_movement": {
            "inputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 32}},
            "labels": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 32}},
        }
    }

    emit_tb_train_cpp(
        tb_dir,
        graph=None,  # unused by the emitter
        top_name="deeplearn",
        in_words=8,
        out_words=2,
        weights_mode="ddr",
        weight_words=10,
        preload_weights=[0.0] * 10,
        training_cfg={"optimizer": {"learning_rate": 0.01}, "execution": {"convergence_smoke": True}},
        output_dir=str(tmp_path),
        raw_cfg=raw_cfg,
    )
    tb = (tb_dir / "tb.cpp").read_text(encoding="utf-8")
    assert "static void pack_record_mem" in tb
    assert 'pack_record_mem(input_mem, input_data, input_words_per_record, r, "input")' in tb
    assert 'pack_record_mem(label_mem, target_data, target_words_per_record, r, "label")' in tb
    assert 'pack_record_mem(input_mem, input_data, input_words_per_record, rec, "input")' in tb
    assert 'pack_record_mem(label_mem, target_data, target_words_per_record, rec, "label")' in tb


def test_training_m_axi_testbench_escapes_record_error_newlines(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_m_axi_tb_escape")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    dm = raw.setdefault("data_movement", {})
    dm.setdefault("inputs", {})["import"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "tile_size": 32}
    dm.setdefault("labels", {})["import"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "tile_size": 32}
    dm.setdefault("outputs", {})["export"] = {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled", "tile_size": 32}

    result = _compile_raw(raw, tmp_path)
    tb_source = (Path(result.out_dir) / "hls/src/tb.cpp").read_text(encoding="utf-8")

    assert 'record count is zero. words_per_record=%d size=%zu\\n", label' in tb_source
    assert 'buffer too small. got=%zu expected=%d\\n", label' in tb_source
    assert 'record count is zero. words_per_record=%d size=%zu\n", label' not in tb_source
    assert 'buffer too small. got=%zu expected=%d\n", label' not in tb_source


def test_optimizer_state_export_runtime_sequence_is_supported_for_training(tmp_path: Path) -> None:
    raw = copy.deepcopy(_load_training_config())
    raw.setdefault("project", {})["out_dir"] = str(tmp_path / "training_optimizer_state_export_sequence")
    raw.setdefault("pipeline", {})["mode"] = "training_on_device"
    raw.setdefault("training", {}).setdefault("optimizer", {})["type"] = "adam"
    raw.setdefault("memory", {})["optimizer_state_storage"] = "bram"
    raw.setdefault("data_movement", {}).setdefault("optimizer_state", {})["export"] = {
        "interface": "m_axi",
        "transport": "ps_runtime",
        "policy": "full",
    }
    raw.setdefault("runtime", {})["sequence"] = [
        {"run_training": {"steps": 1}},
        "export_optimizer_state",
    ]

    result = _compile_raw(raw, tmp_path)
    out_dir = Path(result.out_dir)
    manifest = json.loads((out_dir / "runtime_package/package_manifest.json").read_text(encoding="utf-8"))
    sequence = manifest["runtime_sequence"]

    assert sequence["supported_commands"]["export_optimizer_state"] is True
    assert sequence["sequence"][-1]["command"] == "export_optimizer_state"
    source = (out_dir / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")
    assert "FPGAI_MODE_EXPORT_OPTIMIZER_STATE" in source
    assert "ap_uint<32>* optimizer_state_mem" in source


def test_training_m_axi_testbench_sizes_label_and_output_buffers_from_runtime_records(tmp_path: Path) -> None:
    from fpgai.backends.hls.testbench_train import emit_tb_train_cpp

    tb_dir = tmp_path / "tb_dynamic_label_size"
    tb_dir.mkdir()
    raw_cfg = {
        "data_movement": {
            "inputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 64}},
            "labels": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 64}},
            "outputs": {"interface": "m_axi", "transport": "ps_runtime", "tiled": {"enabled": True, "tile_size": 64}},
        }
    }

    emit_tb_train_cpp(
        tb_dir,
        graph=None,
        top_name="deeplearn",
        in_words=8,
        out_words=0,
        weights_mode="ddr_tiled_mutable",
        weight_words=46,
        preload_weights=[0.0] * 46,
        training_cfg={"optimizer": {"learning_rate": 0.01}, "execution": {"train_steps": 1, "batch_size": 1}},
        output_dir=str(tmp_path),
        raw_cfg=raw_cfg,
    )

    tb_source = (tb_dir / "tb.cpp").read_text(encoding="utf-8")
    assert "if (target_words_per_record <= 0) target_words_per_record = (int)target_data.size();" in tb_source
    assert "std::vector<ap_uint<32> > label_mem((size_t)(target_words_per_record > 0 ? target_words_per_record : 1));" in tb_source
    assert "std::vector<ap_uint<32> > output_mem((size_t)(target_words_per_record > 0 ? target_words_per_record : 1));" in tb_source
    assert "std::vector<ap_uint<32> > label_mem(1);" not in tb_source
    assert "std::vector<ap_uint<32> > output_mem(1);" not in tb_source
