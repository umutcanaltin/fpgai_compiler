from __future__ import annotations

import json
from pathlib import Path

from fpgai.runtime.package import emit_runtime_package


def test_emit_runtime_package_copies_existing_artifacts_and_records_missing_hw(tmp_path: Path) -> None:
    out_dir = tmp_path / "build"
    out_dir.mkdir()

    (out_dir / "manifest.json").write_text('{"top_kernel_name": "deeplearn"}', encoding="utf-8")
    (out_dir / "input.bin").write_bytes(b"input")
    (out_dir / "hls_artifact_metadata.json").write_text('{"schema_version": 1}', encoding="utf-8")
    (out_dir / "hls_schedule_summary.json").write_text('{"report_count": 0}', encoding="utf-8")
    (out_dir / "hls_ii_comparison.json").write_text('{"summary": {}}', encoding="utf-8")

    logs = out_dir / "hls" / "logs"
    logs.mkdir(parents=True)
    (logs / "vitis_hls_stdout.log").write_text("ok", encoding="utf-8")

    result = emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="inference",
        top_name="deeplearn",
        hls_artifacts={"hls_ran": True},
    )

    assert result["status"] == "created"
    assert result["path"] == "runtime_package/package_manifest.json"
    assert result["deployable_overlay_present"] is False
    assert result["bitstream_present"] is False
    assert result["hwh_present"] is False
    assert result["xsa_present"] is False

    package_manifest = out_dir / "runtime_package" / "package_manifest.json"
    data = json.loads(package_manifest.read_text(encoding="utf-8"))

    assert data["board"] == "kv260"
    assert data["pipeline_mode"] == "inference"
    assert data["top_name"] == "deeplearn"
    assert data["hls_artifacts"]["hls_ran"] is True
    assert data["hardware"]["deployable_overlay_present"] is False
    assert data["files"]["compile_manifest"]["package_path"] == "manifest.json"
    assert data["files"]["input_bin"]["package_path"] == "inputs/input.bin"
    assert data["files"]["hls_logs"][0]["package_path"].startswith("hls/logs/")
    assert (out_dir / "runtime_package" / "README_RUNTIME.md").exists()


def test_emit_runtime_package_records_deployable_overlay_when_hw_files_exist(tmp_path: Path) -> None:
    out_dir = tmp_path / "build"
    hw = out_dir / "vivado_bridge" / "bitstream"
    hw.mkdir(parents=True)

    (hw / "design.bit").write_bytes(b"bit")
    (hw / "design.hwh").write_text("hwh", encoding="utf-8")

    result = emit_runtime_package(
        out_dir,
        board="pynq_z2",
        pipeline_mode="inference",
        top_name="deeplearn",
    )

    assert result["deployable_overlay_present"] is True
    assert result["bitstream_present"] is True
    assert result["hwh_present"] is True
    assert result["xsa_present"] is False

    data = json.loads((out_dir / "runtime_package" / "package_manifest.json").read_text(encoding="utf-8"))
    assert data["hardware"]["deployable_overlay_present"] is True
    assert data["files"]["bitstream"]["package_path"] == "hardware/design.bit"
    assert data["files"]["hwh"]["package_path"] == "hardware/design.hwh"


def test_compiler_wires_runtime_package_into_manifest_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "emit_runtime_package" in source
    assert "runtime_package=runtime_package" in source
    assert '"runtime_package": kwargs.get("runtime_package")' in source
    assert '"done" if kwargs.get("runtime_package") is not None else "skipped"' in source


def test_emit_runtime_package_creates_runtime_weight_payload_for_uram(tmp_path: Path) -> None:
    import struct

    out_dir = tmp_path / "compile"
    params_dir = out_dir / "hls" / "src"
    params_dir.mkdir(parents=True)

    (params_dir / "fpgai_params.cpp").write_text(
        """
#include "fpgai_params.h"

namespace fpgai {
op0_wgt_t W0[2];
op0_bias_t B0[1];
const op0_wgt_t W0_init[2] = { 1.0, -2.0 };
const op0_bias_t B0_init[1] = { 0.5 };
} // namespace fpgai
""",
        encoding="utf-8",
    )

    result = emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="inference",
        top_name="deeplearn",
        weights_mode="uram",
    )

    assert result["runtime_weight_payload_required"] is True
    assert result["runtime_weight_payload_present"] is True
    assert result["runtime_weight_total_words"] == 3

    package_dir = out_dir / "runtime_package"
    weights_bin = package_dir / "weights" / "weights.bin"
    layout_path = package_dir / "weights" / "weight_layout.json"
    manifest_path = package_dir / "package_manifest.json"

    assert weights_bin.exists()
    assert layout_path.exists()

    words = struct.unpack("<III", weights_bin.read_bytes())
    expected = tuple(
        struct.unpack("<I", struct.pack("<f", v))[0]
        for v in [1.0, -2.0, 0.5]
    )
    assert words == expected

    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    assert layout["format"] == "packed32"
    assert layout["total_words"] == 3
    assert layout["entries"] == [
        {"name": "W0", "kind": "weight", "offset_words": 0, "count_words": 2},
        {"name": "B0", "kind": "bias", "offset_words": 2, "count_words": 1},
    ]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime_weights"]["required"] is True
    assert manifest["runtime_weights"]["present"] is True
    assert manifest["runtime_weights"]["total_words"] == 3
    assert manifest["files"]["weights_bin"]["package_path"] == "weights/weights.bin"
    assert manifest["files"]["weight_layout"]["package_path"] == "weights/weight_layout.json"


def test_emit_runtime_package_does_not_create_weight_payload_for_embedded_bram(tmp_path: Path) -> None:
    out_dir = tmp_path / "compile"
    params_dir = out_dir / "hls" / "src"
    params_dir.mkdir(parents=True)

    (params_dir / "fpgai_params.cpp").write_text(
        """
#include "fpgai_params.h"

namespace fpgai {
const op0_wgt_t W0[2] = { 1.0, -2.0 };
const op0_bias_t B0[1] = { 0.5 };
} // namespace fpgai
""",
        encoding="utf-8",
    )

    result = emit_runtime_package(
        out_dir,
        board="kv260",
        pipeline_mode="inference",
        top_name="deeplearn",
        weights_mode="embedded",
    )

    assert result["runtime_weight_payload_required"] is False
    assert result["runtime_weight_payload_present"] is False
    assert result["runtime_weight_total_words"] == 0

    package_dir = out_dir / "runtime_package"
    assert not (package_dir / "weights" / "weights.bin").exists()
    assert not (package_dir / "weights" / "weight_layout.json").exists()

    manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_weights"]["required"] is False
    assert manifest["runtime_weights"]["present"] is False
    assert manifest["runtime_weights"]["total_words"] == 0
    assert "weights_bin" not in manifest["files"]
    assert "weight_layout" not in manifest["files"]
