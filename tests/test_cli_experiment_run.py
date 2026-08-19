from pathlib import Path
import json
import sys
import pytest

from fpgai.cli import main as cli_main
from fpgai.experiments.benchmark_runner import run_experiment_from_config


def test_experiment_run_dry_run_writes_manifest(tmp_path, monkeypatch):
    config = Path("configs/experiments/benchmark_suite.yml")
    assert config.exists(), "configs/experiments/benchmark_suite.yml is missing"

    out = tmp_path / "benchmark"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fpgai",
            "experiment",
            "run",
            "--config",
            str(config),
            "--out",
            str(out),
            "--dry-run",
            "--max-design-points",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli_main()

    assert exc.value.code == 0

    manifest = out / "manifest.json"
    status = out / "experiment_status.json"

    assert manifest.exists()
    assert status.exists()

    data = json.loads(manifest.read_text())
    assert data["kind"] == "benchmark_run"
    assert data["dry_run"] is True
    assert "items" in data
    assert len(data["items"]) > 0
    assert all(item["status"] in {"dry_run", "skipped"} for item in data["items"])


def test_experiment_run_propagates_child_sweep_failed_count(tmp_path):
    sweep_cfg = tmp_path / "toy_sweep.yml"
    sweep_cfg.write_text("name: toy_sweep\n")

    experiment_cfg = tmp_path / "benchmark.yml"
    experiment_cfg.write_text(
        f"""
version: 1
benchmark:
  title: Toy
  stage: test
inputs:
  experiments:
    toy: {sweep_cfg}
""".strip()
        + "\n"
    )

    def fake_sweep_runner(config_path, *, out_dir, **kwargs):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(
            json.dumps(
                {
                    "kind": "sweep_results",
                    "failed_count": 1,
                    "passed_count": 0,
                    "records": [{"status": "failed"}],
                }
            )
        )
        return 0

    out_dir = tmp_path / "benchmark_out"
    rc = run_experiment_from_config(
        str(experiment_cfg),
        out_dir=str(out_dir),
        run_sweep_callable=fake_sweep_runner,
    )

    assert rc == 1
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["failed_count"] == 1
    assert manifest["passed_count"] == 0
    assert manifest["items"][0]["status"] == "failed"
    assert manifest["items"][0]["reason"] == "child sweep reported failed_count=1"


def test_experiment_run_supports_explicit_compile_items_and_master_results(tmp_path):
    compile_cfg = tmp_path / "compile.yml"
    compile_cfg.write_text("project:\n  name: toy\n", encoding="utf-8")
    experiment_cfg = tmp_path / "benchmark.yml"
    experiment_cfg.write_text(
        f"""
version: 1
benchmark:
  title: Explicit compile
inputs:
  experiments:
    toy:
      kind: compile
      config: {compile_cfg}
      required_artifacts: [manifest.json]
validation_levels:
  static_validation: generated
limitations: []
""".strip() + "\n",
        encoding="utf-8",
    )

    compiled = tmp_path / "compiled"
    compiled.mkdir()
    (compiled / "manifest.json").write_text(
        json.dumps({
            "pipeline_mode": "inference",
            "build_stages": {"cpp": True},
        }),
        encoding="utf-8",
    )

    def fake_sweep_runner(config_path, *, out_dir, **kwargs):
        generated = Path(config_path)
        assert generated.name == "single_compile.yml"
        text = generated.read_text(encoding="utf-8")
        assert "design_space_sweep: false" in text
        assert str(compile_cfg) in text
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(
            json.dumps({
                "failed_count": 0,
                "passed_count": 1,
                "results": [{
                    "design_name": "toy",
                    "status": "passed",
                    "out_dir": str(compiled),
                }],
            }),
            encoding="utf-8",
        )
        return 0

    out_dir = tmp_path / "campaign"
    rc = run_experiment_from_config(
        str(experiment_cfg),
        out_dir=str(out_dir),
        run_sweep_callable=fake_sweep_runner,
    )

    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["items"][0]["kind"] == "compile"
    assert manifest["items"][0]["status"] == "passed"
    assert manifest["master_results"]["status"] == "passed"
    assert (out_dir / "master_results.json").exists()
    assert (out_dir / "master_results.csv").exists()
    assert (out_dir / "master_results.md").exists()


def test_final_validation_campaign_is_inspectable_and_uses_explicit_compile_items():
    import yaml

    path = Path("configs/experiments/final_validation_campaign.yml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiments = data["inputs"]["experiments"]
    assert data["benchmark"]["compiler_feature_freeze"] is True
    assert data["benchmark"]["architecture_sweeps_are_compiler_feature"] is False
    for name in (
        "ir_inference_trace",
        "ir_training_trace",
        "mlp_mnist",
        "cnn_mnist",
        "yolo_like_detector",
        "transformer_training_layerwise",
        "transformer_training_phase_shared",
    ):
        assert experiments[name]["kind"] == "compile"
        assert Path(experiments[name]["config"]).exists()
