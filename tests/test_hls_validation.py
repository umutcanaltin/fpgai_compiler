from __future__ import annotations

import json
from pathlib import Path

from fpgai.analysis.hls_validation import (
    attach_hls_validation_to_manifest,
    build_hls_validation_report,
    detect_tiled_kernels,
    hls_validation_manifest_entry,
    run_and_write_hls_validation,
)
from fpgai.backends.hls.runner import HLSRunResult


def test_detect_tiled_kernels_from_generated_top(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "deeplearn.cpp").write_text(
        """
        void f() {
            dense_out_in_tiled<1, 1, 1, 1>(a, b, W, B);
            conv2d_tiled<1, 1, 1, 1, 1, 1, 3, 1, 0, 1, 1, 1, 1>(a, b, W, B);
        }
        """,
        encoding="utf-8",
    )

    assert detect_tiled_kernels(tmp_path) == {
        "dense_out_in_tiled": True,
        "conv2d_tiled": True,
    }


def test_build_hls_validation_report_from_result(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "deeplearn.cpp").write_text(
        "dense_out_in_tiled<1, 1, 1, 1>(a, b, W, B);",
        encoding="utf-8",
    )
    report_path = tmp_path / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "csynth.rpt"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("csynth", encoding="utf-8")

    result = HLSRunResult(
        ok=True,
        returncode=0,
        command="vitis_hls -f run_hls.tcl",
        workdir=str(tmp_path),
        stdout_log=str(tmp_path / "stdout.log"),
        stderr_log=str(tmp_path / "stderr.log"),
        csynth_report=str(report_path),
    )

    report = build_hls_validation_report(
        hls_dir=tmp_path,
        result=result,
    )

    assert report["format"] == "fpgai.hls_validation.v1"
    assert report["ok"] is True
    assert report["returncode"] == 0
    assert report["csynth_report_present"] is True
    assert report["tiling_enabled"] is True
    assert report["tiled_kernels_detected"]["dense_out_in_tiled"] is True
    assert report["tiled_kernels_detected"]["conv2d_tiled"] is False


def test_hls_validation_manifest_entry_is_compact() -> None:
    entry = hls_validation_manifest_entry(
        {
            "format": "fpgai.hls_validation.v1",
            "requested": True,
            "ok": True,
            "returncode": 0,
            "csynth_report_present": True,
            "tiling_enabled": True,
            "tiled_kernels_detected": {
                "dense_out_in_tiled": True,
                "conv2d_tiled": False,
            },
        },
        path="reports/hls_validation.json",
    )

    assert entry == {
        "format": "fpgai.hls_validation.v1",
        "path": "reports/hls_validation.json",
        "requested": True,
        "ok": True,
        "returncode": 0,
        "csynth_report_present": True,
        "tiling_enabled": True,
        "dense_out_in_tiled": True,
        "conv2d_tiled": False,
    }


def test_run_and_write_hls_validation_writes_json_and_manifest(tmp_path) -> None:
    (tmp_path / "run_hls.tcl").write_text("exit\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "deeplearn.cpp").write_text(
        "conv2d_tiled<1, 1, 1, 1, 1, 1, 3, 1, 0, 1, 1, 1, 1>(a, b, W, B);",
        encoding="utf-8",
    )

    csynth = tmp_path / "csynth.rpt"
    csynth.write_text("report", encoding="utf-8")

    def fake_run(**kwargs):
        return HLSRunResult(
            ok=True,
            returncode=0,
            command="fake",
            workdir=str(kwargs["hls_dir"]),
            stdout_log=str(tmp_path / "stdout.log"),
            stderr_log=str(tmp_path / "stderr.log"),
            csynth_report=str(csynth),
        )

    manifest = {}
    report, updated = run_and_write_hls_validation(
        tmp_path,
        manifest=manifest,
        run_hls_fn=fake_run,
    )

    output = tmp_path / "reports" / "hls_validation.json"
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert updated is manifest
    assert manifest["hls_validation"]["path"] == "hls_validation.json"
    assert manifest["hls_validation"]["ok"] is True
    assert manifest["hls_validation"]["conv2d_tiled"] is True


def test_run_and_write_hls_validation_records_runner_errors(tmp_path) -> None:
    (tmp_path / "src").mkdir()

    def failing_run(**kwargs):
        raise RuntimeError("vitis unavailable")

    report, manifest = run_and_write_hls_validation(
        tmp_path,
        run_hls_fn=failing_run,
    )

    assert report["ok"] is False
    assert report["error"] == "vitis unavailable"
    assert manifest["hls_validation"]["ok"] is False
