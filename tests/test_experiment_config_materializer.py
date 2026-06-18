from __future__ import annotations

import json
from pathlib import Path

import yaml

from fpgai.experiments.config_materializer import apply_parameter_overrides, materialize_design_config
from fpgai.experiments.design_matrix import expand_design_matrix
from fpgai.experiments.sweep_runner import SweepRunner


def test_apply_parameter_overrides_updates_existing_paths_strictly():
    base = {"notes": {"parallel_policy": "Balanced", "precision_mode": "fixed"}}
    out, report = apply_parameter_overrides(
        base,
        {"policy": "latency_first", "precision_mode": "fx8_3", "board": "kv260"},
    )
    assert out["notes"]["parallel_policy"] == "LatencyFirst"
    assert out["notes"]["precision_mode"] == "fx8_3"
    assert report["applied"]["policy"] == "notes.parallel_policy"
    assert report["applied"]["precision_mode"] == "notes.precision_mode"
    assert report["unapplied"]["board"] == "kv260"


def test_materialize_design_config_writes_strict_yaml_and_sidecar(tmp_path: Path):
    model = tmp_path / "model.onnx"
    model.write_text("dummy", encoding="utf-8")
    base = tmp_path / "fpgai.yml"
    base.write_text(
        "model:\n"
        "  path: old.onnx\n"
        "notes:\n"
        "  parallel_policy: Balanced\n"
        "  precision_mode: fixed\n",
        encoding="utf-8",
    )
    out = tmp_path / "experiments" / "configs" / "point.yml"
    report = materialize_design_config(
        base,
        out,
        {"policy": "resource_first", "precision_mode": "fx12_4", "model_path": str(model), "board": "kv260"},
        design_name="point",
        repo_root=tmp_path,
    )
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["notes"]["parallel_policy"] == "ResourceFirst"
    assert data["notes"]["precision_mode"] == "fx12_4"
    assert "experiment" not in data
    assert report.output_config_path.endswith("point.yml")
    meta = json.loads(Path(report.metadata_path).read_text(encoding="utf-8"))
    assert meta["design_name"] == "point"
    assert meta["unapplied"]["board"] == "kv260"


def test_sweep_runner_materializes_config_and_command_strictly(tmp_path: Path):
    repo = tmp_path
    (repo / "fpgai.yml").write_text("notes:\n  parallel_policy: Balanced\n", encoding="utf-8")
    config = {
        "name": "policy_test",
        "command_template": "python main.py benchmark --config {config_path}",
        "defaults": {"base_config_path": "fpgai.yml", "config_path": "fpgai.yml", "model_path": "m.onnx", "board": "kv260"},
        "parameters": {"policy": ["resource_first"]},
    }
    points = expand_design_matrix(config)
    runner = SweepRunner(
        repo / "experiments" / "policy_test",
        repo_root=repo,
        dry_run=True,
        materialize_configs={"enabled": True, "directory": "configs"},
        command_template=config["command_template"],
    )
    payload = runner.run_points(points)
    rec = payload["results"][0]
    assert rec["config_path"].endswith("experiments/policy_test/configs/policy_test_000.yml")
    assert "experiments/policy_test/configs/policy_test_000.yml" in rec["command"]
    generated = repo / rec["config_path"]
    assert generated.exists()
    assert (repo / rec["config_path"].replace(".yml", ".metadata.json")).exists()
    data = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert data["notes"]["parallel_policy"] == "ResourceFirst"
    assert "experiment" not in data
