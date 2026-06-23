from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fpgai.reporting.artifacts import build_report
from fpgai.validation.results import validate_results


def _write_smoke_tree(root: Path) -> None:
    child = root / "training_convergence"
    child.mkdir(parents=True)
    (child / "results.json").write_text(
        json.dumps(
            {
                "failed_count": 0,
                "passed_count": 1,
                "skipped_count": 0,
                "results": [
                    {
                        "design_name": "d0",
                        "status": "passed",
                        "returncode": 0,
                        "parameters": {"policy": "balanced"},
                        "metrics": {"latency_ms": 1.25},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    status = {
        "kind": "paper_experiment_status",
        "failed_count": 0,
        "passed_count": 1,
        "skipped_count": 0,
        "dry_run_count": 0,
        "items": [
            {
                "name": "training_convergence",
                "status": "passed",
                "out_dir": str(child),
                "sweep_config": "configs/sweeps/training_multi_epoch_convergence.yml",
                "child_summary": {
                    "results_path": str(child / "results.json"),
                    "failed_count": 0,
                    "passed_count": 1,
                },
            }
        ],
    }
    (root / "experiment_status.json").write_text(json.dumps(status), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"items": status["items"]}), encoding="utf-8")


def test_build_report_writes_public_artifacts(tmp_path: Path):
    src = tmp_path / "paper"
    src.mkdir()
    _write_smoke_tree(src)

    out = tmp_path / "report"
    result = build_report(src, out)

    assert result.result_count == 1
    assert result.failed_count == 0
    assert (out / "summary.md").exists()
    assert (out / "results_table.csv").exists()
    assert (out / "claim_traceability.md").exists()


def test_validate_results_accepts_consistent_tree(tmp_path: Path):
    src = tmp_path / "paper"
    src.mkdir()
    _write_smoke_tree(src)

    result = validate_results(src)

    assert result.passed
    assert result.error_count == 0
    assert result.child_count == 1


def test_validate_results_rejects_false_pass(tmp_path: Path):
    src = tmp_path / "paper"
    src.mkdir()
    _write_smoke_tree(src)

    child_results = src / "training_convergence" / "results.json"
    payload = json.loads(child_results.read_text(encoding="utf-8"))
    payload["failed_count"] = 1
    payload["results"][0]["status"] = "failed"
    child_results.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_results(src)

    assert not result.passed
    assert result.error_count == 1
    assert "false pass" in result.issues[0].message

def _run_main_help(*args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]

    return subprocess.run(
        [sys.executable, str(root / "main.py"), *args, "--help"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_report_cli_exposes_existing_reporting_subcommands():
    result = _run_main_help("report")

    assert result.returncode == 0
    assert "paper-artifacts" in result.stdout
    assert "frontier" in result.stdout
    assert "estimator" in result.stdout


def test_report_paper_artifacts_help_is_public():
    result = _run_main_help("report", "paper-artifacts")

    assert result.returncode == 0
    assert "--csv" in result.stdout
    assert "--out" in result.stdout
    assert "generated paper artifacts" in result.stdout


def test_report_frontier_help_is_public():
    result = _run_main_help("report", "frontier")

    assert result.returncode == 0
    assert "--csv" in result.stdout
    assert "--out" in result.stdout
    assert "--require-pass" in result.stdout


def test_report_estimator_help_is_public():
    result = _run_main_help("report", "estimator")

    assert result.returncode == 0
    assert "--csv" in result.stdout
    assert "--out" in result.stdout
    assert "--inference-filter" in result.stdout
    assert "--training-filter" in result.stdout

def _run_main(*args: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]

    return subprocess.run(
        [sys.executable, str(root / "main.py"), *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_report_paper_artifacts_runs_on_tiny_csv(tmp_path: Path):
    csv_path = tmp_path / "paper.csv"
    out_dir = tmp_path / "paper_out"
    csv_path.write_text(
        "\n".join(
            [
                "model_name,precision_policy,parallel_policy,status,compile_ok,hls_ok,benchmark_passed,quant_cosine,quant_mse,quant_mae,quant_rmse,bench_cosine,bench_mse,bench_mae,bench_rmse,lut,ff,dsp,bram_18k,uram,latency_cycles_min,latency_cycles_max,latency_ms,estimated_clock_ns,ii,bottleneck",
                "toy,fx8_3,balanced,passed,True,True,True,0.99,0.01,0.01,0.1,0.98,0.02,0.02,0.14,100,200,4,2,0,1000,1200,0.01,5.0,1,compute",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_main(
        "report",
        "paper-artifacts",
        "--csv",
        str(csv_path),
        "--out",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "summary.txt").is_file()
    assert (out_dir / "tables" / "table_accuracy.csv").is_file()
    assert (out_dir / "tables" / "table_resource_latency.csv").is_file()


def test_report_frontier_runs_on_tiny_csv(tmp_path: Path):
    csv_path = tmp_path / "frontier.csv"
    out_dir = tmp_path / "frontier_out"
    csv_path.write_text(
        "\n".join(
            [
                "model_name,precision_policy,parallel_policy,benchmark_passed,dsp,lut,ff,bram_18k,uram,ii,estimated_clock_ns,latency_cycles_min,latency_cycles_max,latency_seconds_min,latency_seconds_max,latency_ms,quant_cosine,quant_mse,quant_mae,quant_rmse,bench_cosine,bench_mse,bench_mae,bench_rmse,out_dir",
                "toy,fx8_3,balanced,True,4,100,200,2,0,1,5.0,1000,1200,0.010,0.012,0.01,0.99,0.01,0.01,0.1,0.98,0.02,0.02,0.14,run0",
                "toy,fx16_6,latency_first,True,8,140,260,3,0,1,5.0,600,800,0.006,0.008,0.006,0.995,0.005,0.005,0.07,0.99,0.01,0.01,0.1,run1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_main(
        "report",
        "frontier",
        "--csv",
        str(csv_path),
        "--out",
        str(out_dir),
        "--require-pass",
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "frontier_points.csv").is_file()
    assert (out_dir / "frontier_knees.csv").is_file()
    assert (out_dir / "paper_discussion.txt").is_file()


def test_report_estimator_runs_on_tiny_csv(tmp_path: Path):
    csv_path = tmp_path / "estimator.csv"
    out_dir = tmp_path / "estimator_out"
    csv_path.write_text(
        "\n".join(
            [
                "model,mode,precision,policy,compile_ok,latency_cycles,pred_latency_cycles,lut,pred_lut,ff,pred_ff,dsp,pred_dsp,bram_18k,pred_bram_18k",
                "toy,inference,fx8_3,balanced,True,1000,1100,100,120,200,180,4,5,2,2",
                "toy,training,fx16_6,latency_first,True,2000,2100,180,170,260,250,8,9,3,4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_main(
        "report",
        "estimator",
        "--csv",
        str(csv_path),
        "--out",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    assert any(out_dir.iterdir())

