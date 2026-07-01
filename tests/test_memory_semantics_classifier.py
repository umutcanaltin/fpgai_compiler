from __future__ import annotations

from pathlib import Path

import pytest

from fpgai.reports.memory_semantics import classify_generated_memory_semantics


ROOT = Path("paper_experiments/full_pipeline_gate/sprint27j_paper_validation/runs_hls")


def _run(name: str) -> Path:
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"generated run not available: {p}")
    return p


def test_bram_memory_row_is_classified_as_legacy_compile_time_constants() -> None:
    sem = classify_generated_memory_semantics(_run("kv260_memory_bram"))
    assert sem.mode == "legacy_compile_time_constants"
    assert not sem.has_weights_mem
    assert not sem.has_runtime_payload
    assert sem.has_const_weight_arrays


def test_ddr_memory_row_is_classified_as_preload_not_tiled() -> None:
    sem = classify_generated_memory_semantics(_run("kv260_memory_ddr"))
    assert sem.mode == "bram_import_full"
    assert sem.has_weights_mem
    assert sem.has_weights_m_axi
    assert sem.has_runtime_payload
    assert sem.has_full_weight_arrays
    assert not sem.has_tile_weight_buffer


def test_new_schema_ddr_memory_row_is_classified_as_preload_not_tiled() -> None:
    sem = classify_generated_memory_semantics(_run("kv260_memory_ddr_new_schema"))
    assert sem.mode == "bram_import_full"
    assert sem.has_weights_mem
    assert sem.has_weights_m_axi
    assert sem.has_runtime_payload
    assert sem.has_full_weight_arrays
    assert not sem.has_tile_weight_buffer


def test_uram_memory_row_is_classified_as_uram_import_full() -> None:
    sem = classify_generated_memory_semantics(_run("kv260_memory_uram"))
    assert sem.mode == "uram_import_full"
    assert sem.has_weights_mem
    assert sem.has_weights_m_axi
    assert sem.has_runtime_payload
    assert sem.has_full_weight_arrays
    assert sem.has_uram_weight_bind


def test_precision_rows_are_legacy_compile_time_constants_not_memory_experiments() -> None:
    for name in ["kv260_precision_fx8_3", "kv260_precision_fx12_4", "kv260_precision_fx16_6"]:
        sem = classify_generated_memory_semantics(_run(name))
        assert sem.mode == "legacy_compile_time_constants"


def test_synthetic_bram_static_source_is_classified_as_bram_static(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "hls" / "src").mkdir(parents=True)
    (run / "hls" / "src" / "deeplearn.cpp").write_text(
        """
extern "C" void deeplearn() {
    static op0_wgt_t W0[8];
#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=bram
    static op0_bias_t B0[2];
#pragma HLS BIND_STORAGE variable=B0 type=ram_2p impl=bram
}
""",
        encoding="utf-8",
    )
    (run / "hls" / "src" / "fpgai_params.cpp").write_text(
        """
namespace fpgai {
const op0_wgt_t W0[8] = {0};
const op0_bias_t B0[2] = {0};
}
""",
        encoding="utf-8",
    )

    sem = classify_generated_memory_semantics(run)
    assert sem.mode == "bram_static"
    assert sem.has_bram_weight_bind
    assert not sem.has_runtime_payload
    assert not sem.has_weights_mem


def test_synthetic_uram_static_source_is_classified_as_uram_static(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "hls" / "src").mkdir(parents=True)
    (run / "hls" / "src" / "deeplearn.cpp").write_text(
        """
extern "C" void deeplearn() {
    static op0_wgt_t W0[8];
#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=uram
    static op0_bias_t B0[2];
#pragma HLS BIND_STORAGE variable=B0 type=ram_2p impl=uram
}
""",
        encoding="utf-8",
    )
    (run / "hls" / "src" / "fpgai_params.cpp").write_text(
        """
namespace fpgai {
const op0_wgt_t W0[8] = {0};
const op0_bias_t B0[2] = {0};
}
""",
        encoding="utf-8",
    )

    sem = classify_generated_memory_semantics(run)
    assert sem.mode == "uram_static"
    assert sem.has_uram_weight_bind
    assert not sem.has_runtime_payload
    assert not sem.has_weights_mem


def test_synthetic_ddr_tiled_source_is_classified_as_ddr_tiled(tmp_path: Path) -> None:
    from fpgai.reports.memory_semantics import classify_generated_memory_semantics

    run = tmp_path
    src = run / "hls/src"
    src.mkdir(parents=True)
    (src / "deeplearn.cpp").write_text(
        """
extern "C" void deeplearn(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream, const ap_uint<32>* weights_mem) {
#pragma HLS INTERFACE m_axi port=weights_mem offset=slave bundle=gmem_weights
    op0_wgt_t weight_tile[4][4];
    dense_out_in_ddr_tiled<16, 8, 4, 4, op0_act_t, op0_act_t, op0_wgt_t, op0_bias_t, op0_acc_t>(layer_in, layer_out, weights_mem, 0, 128);
}
"""
    )
    (src / "fpgai_params.cpp").write_text(
        """
namespace fpgai {
const op0_wgt_t W0_init[128] = { 0 };
const op0_bias_t B0_init[8] = { 0 };
}
"""
    )
    pkg = run / "runtime_package"
    (pkg / "weights").mkdir(parents=True)
    (pkg / "weights/weights.bin").write_bytes(b"\0" * 16)
    (pkg / "package_manifest.json").write_text(
        '{"runtime_weights":{"required":true,"present":true,"total_words":136}}'
    )

    sem = classify_generated_memory_semantics(run)
    assert sem.mode == "ddr_tiled"
    assert sem.has_weights_mem is True
    assert sem.has_weights_m_axi is True
    assert sem.has_tile_weight_buffer is True
    assert sem.has_full_weight_arrays is False


def test_synthetic_bram_import_export_source_is_classified_as_bram_import_export_full(tmp_path: Path) -> None:
    run = tmp_path / "run"
    src = run / "hls" / "src"
    src.mkdir(parents=True)
    (src / "deeplearn.cpp").write_text(
        """
extern "C" void deeplearn(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream, ap_uint<32>* weights_mem, int mode) {
#pragma HLS INTERFACE m_axi port=weights_mem offset=slave bundle=gmem_weights
    static const int FPGAI_MODE_IMPORT_WEIGHTS = 1;
    static const int FPGAI_MODE_EXPORT_WEIGHTS = 2;
    static op0_wgt_t W0[8];
#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=bram
    static op0_bias_t B0[2];
#pragma HLS BIND_STORAGE variable=B0 type=ram_2p impl=bram
    int offset = 0;
    if (mode == FPGAI_MODE_IMPORT_WEIGHTS) {
        fpgai_load_ddr_vector<op0_wgt_t, 8>(weights_mem, offset, W0);
        return;
    }
    if (mode == FPGAI_MODE_EXPORT_WEIGHTS) {
        fpgai_store_ddr_vector<op0_wgt_t, 8>(weights_mem, offset, W0);
        return;
    }
}
""",
        encoding="utf-8",
    )
    (src / "fpgai_params.cpp").write_text(
        """
namespace fpgai {
const op0_wgt_t W0_init[8] = {0};
const op0_bias_t B0_init[2] = {0};
}
""",
        encoding="utf-8",
    )
    pkg = run / "runtime_package"
    (pkg / "weights").mkdir(parents=True)
    (pkg / "weights" / "weights.bin").write_bytes(b"\0" * 16)
    (pkg / "package_manifest.json").write_text(
        '{"runtime_weights":{"required":true,"present":true,"total_words":10}}',
        encoding="utf-8",
    )

    sem = classify_generated_memory_semantics(run)
    assert sem.mode == "bram_import_export_full"
    assert sem.has_weights_mem
    assert sem.has_bram_weight_bind
    assert sem.has_full_weight_arrays


def test_synthetic_uram_import_export_source_is_classified_as_uram_import_export_full(tmp_path: Path) -> None:
    run = tmp_path / "run"
    src = run / "hls" / "src"
    src.mkdir(parents=True)
    (src / "deeplearn.cpp").write_text(
        """
extern "C" void deeplearn(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream, ap_uint<32>* weights_mem, int mode) {
#pragma HLS INTERFACE m_axi port=weights_mem offset=slave bundle=gmem_weights
    static const int FPGAI_MODE_IMPORT_WEIGHTS = 1;
    static const int FPGAI_MODE_EXPORT_WEIGHTS = 2;
    static op0_wgt_t W0[8];
#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=uram
    if (mode == FPGAI_MODE_IMPORT_WEIGHTS) {
        int offset = 0;
        fpgai_load_ddr_vector<op0_wgt_t, 8>(weights_mem, offset, W0);
        return;
    }
    if (mode == FPGAI_MODE_EXPORT_WEIGHTS) {
        int offset = 0;
        fpgai_store_ddr_vector<op0_wgt_t, 8>(weights_mem, offset, W0);
        return;
    }
}
""",
        encoding="utf-8",
    )
    (src / "fpgai_params.cpp").write_text("namespace fpgai { const op0_wgt_t W0_init[8] = {0}; }", encoding="utf-8")
    pkg = run / "runtime_package"
    (pkg / "weights").mkdir(parents=True)
    (pkg / "weights" / "weights.bin").write_bytes(b"\0" * 16)
    (pkg / "package_manifest.json").write_text(
        '{"runtime_weights":{"required":true,"present":true,"total_words":8}}',
        encoding="utf-8",
    )

    sem = classify_generated_memory_semantics(run)
    assert sem.mode == "uram_import_export_full"
    assert sem.has_uram_weight_bind


def test_classifier_reports_activation_storage_bindings(tmp_path: Path) -> None:
    run = tmp_path / "run"
    src = run / "hls" / "src"
    src.mkdir(parents=True)
    (src / "deeplearn.cpp").write_text(
        """
extern "C" void deeplearn() {
    static op0_wgt_t W0[8];
#pragma HLS BIND_STORAGE variable=W0 type=ram_2p impl=bram
    op0_act_t layer_in[4];
#pragma HLS BIND_STORAGE variable=layer_in type=ram_1p impl=uram
    op0_act_t layer_0_out[2];
#pragma HLS BIND_STORAGE variable=layer_0_out type=ram_1p impl=uram
}
""",
        encoding="utf-8",
    )
    (src / "fpgai_params.cpp").write_text("namespace fpgai { const op0_wgt_t W0[8] = {0}; }", encoding="utf-8")

    sem = classify_generated_memory_semantics(run)
    assert sem.mode == "bram_static"
    assert sem.activation_storage == "uram"
    assert sem.has_uram_activation_bind is True
