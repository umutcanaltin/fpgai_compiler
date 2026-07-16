from __future__ import annotations

import json
from pathlib import Path

import yaml

from fpgai.experiments.config_materializer import (
    apply_parameter_overrides,
    materialize_design_config,
)
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

    base = tmp_path / "configs" / "examples" / "default_compile.yml"
    base.parent.mkdir(parents=True, exist_ok=True)
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
        {
            "policy": "resource_first",
            "precision_mode": "fx12_4",
            "model_path": str(model),
            "board": "kv260",
        },
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

    base = repo / "configs" / "examples" / "default_compile.yml"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(
        "notes:\n"
        "  parallel_policy: Balanced\n",
        encoding="utf-8",
    )

    config = {
        "name": "policy_test",
        "command_template": "python main.py benchmark --config {config_path}",
        "defaults": {
            "base_config_path": "configs/examples/default_compile.yml",
            "config_path": "configs/examples/default_compile.yml",
            "model_path": "m.onnx",
            "board": "kv260",
        },
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

    assert rec["config_path"].endswith(
        "experiments/policy_test/configs/policy_test_000.yml"
    )
    assert "experiments/policy_test/configs/policy_test_000.yml" in rec["command"]

    generated = repo / rec["config_path"]
    assert generated.exists()
    assert (repo / rec["config_path"].replace(".yml", ".metadata.json")).exists()

    data = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert data["notes"]["parallel_policy"] == "ResourceFirst"
    assert "experiment" not in data





def test_materializer_canonicalizes_memory_first_policy(tmp_path: Path):
    import inspect
    import yaml
    from fpgai.experiments.config_materializer import materialize_design_config

    base = tmp_path / "base.yml"
    base.write_text(
        """optimization:
  parallel_policy: Balanced
""",
        encoding="utf-8",
    )

    out = tmp_path / "out.yml"

    sig = inspect.signature(materialize_design_config)
    kwargs = {
        "base_config_path": base,
        "output_config_path": out,
        "design_name": "memory_first_test",
        "parameters": {"policy": "memory_first"},
        "parameter_mappings": {"policy": "optimization.parallel_policy"},
        "options": {"compiler_policy_names": True},
    }

    # Keep compatibility with the repo's current materializer API.
    if "metadata_path" in sig.parameters:
        kwargs["metadata_path"] = tmp_path / "out.metadata.json"

    report = materialize_design_config(**kwargs)

    assert report["applied"]["policy"] == "optimization.parallel_policy"

    data = yaml.safe_load(out.read_text())
    assert data["optimization"]["parallel_policy"] == "Memory-First"


def test_multi_epoch_training_sweep_uses_matching_real_mnist_base_contract() -> None:
    config_path = Path("configs/sweeps/training_multi_epoch_convergence.yml")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = data["defaults"]

    assert defaults["base_config_path"] == "examples/training/mnist_balanced10_dataset_training.yml"
    assert defaults["config_path"] == "examples/training/mnist_balanced10_dataset_training.yml"
    assert defaults["model_path"] == "models/suite/mlp_mnist.onnx"
    assert "dataset_path" not in defaults
    assert "validation_data/mnist_samples.npz" not in config_path.read_text(encoding="utf-8")

    base = yaml.safe_load(
        Path(defaults["base_config_path"]).read_text(encoding="utf-8")
    )
    assert base["model"]["path"] == "models/suite/mlp_mnist.onnx"
    assert base["validation"]["dataset"]["source"] == "torchvision"
    assert base["validation"]["dataset"]["name"] == "MNIST"
    assert base["validation"]["dataset"]["sample_selection"]["mode"] == "balanced_per_class"
    assert "execution" not in base["training"]
