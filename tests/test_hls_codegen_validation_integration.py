from __future__ import annotations

import json
from types import SimpleNamespace

from fpgai.backends.hls.codegen import emit_hls_stub


def test_emit_hls_stub_can_write_hls_validation_report(tmp_path, monkeypatch) -> None:
    def fake_validation(hls_dir, *, manifest, **kwargs):
        reports_dir = hls_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report = {
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
        }
        (reports_dir / "hls_validation.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        manifest["hls_validation"] = {
            "format": "fpgai.hls_validation.v1",
            "path": "hls_validation.json",
            "requested": True,
            "ok": True,
            "returncode": 0,
            "csynth_report_present": True,
            "tiling_enabled": True,
            "dense_out_in_tiled": True,
            "conv2d_tiled": False,
        }
        return report, manifest

    monkeypatch.setattr(
        "fpgai.backends.hls.codegen.run_and_write_hls_validation",
        fake_validation,
    )

    project = emit_hls_stub(
        graph=SimpleNamespace(),
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "pipeline_mode": "inference",
            "run_csim": False,
            "run_csynth": False,
            "run_hls_validation": True,
        },
    )

    assert (project.hls_dir / "reports" / "hls_validation.json").exists()

    meta = json.loads(
        (project.hls_dir / "codegen_meta.json").read_text(
            encoding="utf-8"
        )
    )

    assert meta["hls_validation"]["path"] == "reports/hls_validation.json"
    assert meta["hls_validation"]["ok"] is True
    assert meta["hls_validation"]["dense_out_in_tiled"] is True


def test_emit_hls_stub_hls_validation_is_opt_in(tmp_path) -> None:
    project = emit_hls_stub(
        graph=SimpleNamespace(),
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "pipeline_mode": "inference",
            "run_csim": False,
            "run_csynth": False,
        },
    )

    meta = json.loads(
        (project.hls_dir / "codegen_meta.json").read_text(
            encoding="utf-8"
        )
    )

    assert "hls_validation" not in meta
    assert "hls_validation_error" not in meta
