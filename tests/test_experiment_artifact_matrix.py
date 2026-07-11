from __future__ import annotations

import csv
import json
from pathlib import Path

from fpgai.paper.experiment_artifacts import emit_experiment_artifact_reports


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_experiment_artifact_matrix_records_bitstream_package_claim_level(tmp_path: Path) -> None:
    out = tmp_path / "compile"
    (out / "hls/src").mkdir(parents=True)
    (out / "hls/src/deeplearn.cpp").write_text("void deeplearn() {}", encoding="utf-8")
    _write_json(
        out / "manifest.json",
        {
            "pipeline_mode": "inference",
            "top_kernel_name": "deeplearn",
            "hls_ran": True,
            "hls_ok": True,
            "hls_returncode": 0,
            "build_stages": {"bitstream": True},
            "vivado_bridge": {
                "board": "kv260",
                "ok": True,
                "vivado_bridge_generated": True,
                "vivado_impl_requested": True,
                "bitstream_requested": True,
                "bitstream_exists": True,
                "xsa_exists": True,
                "failed_rows": [],
            },
        },
    )
    _write_json(
        out / "reports/paper_verification.json",
        {
            "verification_flags": {
                "vivado_implemented": True,
                "bitstream_generated": True,
            }
        },
    )
    _write_json(
        out / "runtime_package/runtime_package_validation.json",
        {"deployability_ready": True, "failed_count": 0},
    )
    _write_json(out / "runtime_package/package_manifest.json", {"files": {}, "board": "kv260"})

    result = emit_experiment_artifact_reports(out)

    assert result["summary"]["claim_level"] == "level_3_bitstream_package"
    matrix = json.loads((out / "reports/experiment_artifact_matrix.json").read_text(encoding="utf-8"))
    row = matrix["rows"][0]
    assert row["runtime_package_validated"] is True
    assert row["bitstream_generated"] is True
    assert row["board_execution_claimed"] is False
    assert row["claim_level"] == "level_3_bitstream_package"

    package_manifest = json.loads((out / "runtime_package/package_manifest.json").read_text(encoding="utf-8"))
    assert package_manifest["experiment_artifacts"]["claim_level"] == "level_3_bitstream_package"
    assert package_manifest["files"]["experiment_artifact_matrix_json"]["package_path"] == "reports/experiment_artifact_matrix.json"


def test_training_curve_contract_is_pending_until_board_runtime(tmp_path: Path) -> None:
    out = tmp_path / "compile"
    _write_json(
        out / "manifest.json",
        {
            "pipeline_mode": "training_on_device",
            "top_kernel_name": "deeplearn",
            "hls_ran": True,
            "hls_ok": True,
            "training_plan": {"optimizer_type": "sgd"},
            "training_reference": {
                "loss_before": 1.25,
                "loss_after": 0.75,
                "summary_json": "training_reference/summary.json",
            },
            "training_compare": {
                "grad_cosine": 0.99,
                "weight_after_cosine": 0.98,
                "weight_delta_cosine": 0.97,
            },
        },
    )
    _write_json(out / "runtime_package/package_manifest.json", {"files": {}, "pipeline_mode": "training_on_device"})

    result = emit_experiment_artifact_reports(out)

    assert result["summary"]["training_curve_contract_available"] is True
    assert result["summary"]["training_curve_available"] is False
    assert result["summary"]["training_curve_source"] == "pending_board_runtime"
    assert not (out / "training/training_curve.csv").exists()
    assert not (out / "reports/paper_training_curve.csv").exists()
    assert (out / "reports/training_curve_contract.json").exists()
    assert (out / "runtime_package/reports/training_curve_contract.json").exists()

    report = json.loads((out / "reports/training_curve_report.json").read_text(encoding="utf-8"))
    assert report["available"] is False
    assert report["contract_available"] is True
    assert report["source"] == "pending_board_runtime"
    assert "loss" in report["required_fields"]
    assert "runtime_seconds" in report["required_fields"]


def test_training_curve_artifacts_use_real_board_runtime_rows(tmp_path: Path) -> None:
    out = tmp_path / "compile"
    _write_json(
        out / "manifest.json",
        {
            "pipeline_mode": "training_on_device",
            "top_kernel_name": "deeplearn",
            "hls_ran": True,
            "hls_ok": True,
            "training_plan": {"optimizer_type": "sgd"},
        },
    )
    _write_json(out / "runtime_package/package_manifest.json", {"files": {}, "pipeline_mode": "training_on_device"})
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "reports/board_training_curve.csv").write_text(
        "step,epoch,batch,loss,accuracy,runtime_seconds,status\n"
        "0,0,0,1.25,0.50,0.010,ok\n"
        "1,0,1,0.75,0.70,0.020,ok\n",
        encoding="utf-8",
    )

    result = emit_experiment_artifact_reports(out)

    assert result["summary"]["training_curve_available"] is True
    assert result["summary"]["training_curve_source"] == "kv260_board_runtime"
    assert (out / "training/training_curve.csv").exists()
    assert (out / "reports/paper_training_curve.csv").exists()

    with (out / "training/training_curve.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["step"] for row in rows] == ["0", "1"]
    assert rows[0]["loss"] == "1.25"
    assert rows[1]["loss"] == "0.75"
    assert rows[1]["cumulative_runtime_seconds"] == "0.03"

    summary = json.loads((out / "reports/paper_training_curve_summary.json").read_text(encoding="utf-8"))
    assert summary["final_loss"] == 0.75
    assert summary["board_execution_claimed"] is False


def test_compiler_wires_experiment_artifact_reports_after_manifest_and_vivado_bridge() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "from fpgai.paper.experiment_artifacts import emit_experiment_artifact_reports" in source
    assert "emit_experiment_artifact_reports(out_dir)" in source
    assert source.count("emit_experiment_artifact_reports(out_dir)") >= 3


def test_failed_hls_does_not_upgrade_to_bitstream_claim_even_if_stale_bitstream_exists(tmp_path: Path) -> None:
    out = tmp_path / "compile_failed_hls"
    (out / "hls/src").mkdir(parents=True)
    (out / "hls/src/deeplearn.cpp").write_text("void deeplearn() {}", encoding="utf-8")
    _write_json(
        out / "manifest.json",
        {
            "pipeline_mode": "inference",
            "top_kernel_name": "deeplearn",
            "hls_ran": True,
            "hls_ok": False,
            "hls_returncode": 1,
            "build_stages": {"bitstream": True},
            "vivado_bridge": {
                "board": "kv260",
                "ok": True,
                "vivado_bridge_generated": True,
                "vivado_impl_requested": True,
                "bitstream_requested": True,
                "bitstream_exists": True,
                "xsa_exists": True,
                "failed_rows": [],
            },
        },
    )
    _write_json(out / "reports/paper_verification.json", {"verification_flags": {"vivado_implemented": True, "bitstream_generated": True}})
    _write_json(out / "runtime_package/runtime_package_validation.json", {"deployability_ready": True, "failed_count": 0})
    _write_json(out / "runtime_package/package_manifest.json", {"files": {}, "board": "kv260"})

    result = emit_experiment_artifact_reports(out)

    matrix = json.loads((out / "reports/experiment_artifact_matrix.json").read_text(encoding="utf-8"))
    row = matrix["rows"][0]
    assert row["hls_ok"] is False
    assert row["bitstream_generated"] is True
    assert result["summary"]["claim_level"] == "level_0_compiler_artifact"
    assert row["claim_level"] == "level_0_compiler_artifact"
