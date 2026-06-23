from __future__ import annotations

import json
from pathlib import Path

from fpgai.experiments.config_materializer import materialize_design_config


def test_materializer_does_not_write_unknown_experiment_section(tmp_path: Path):
    model = tmp_path / "model.onnx"
    model.write_text("dummy")
    base = tmp_path / "configs/examples/default_compile.yml"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(
        "project: demo\n"
        "model:\n"
        "  path: old.onnx\n"
        "notes:\n"
        "  parallel_policy: Balanced\n"
        "  precision_mode: fixed\n"
    )

    out = tmp_path / "configs" / "design.yml"
    result = materialize_design_config(
        base,
        out,
        {"model_path": str(model), "policy": "resource_first", "precision_mode": "fixed"},
        design_name="design",
        repo_root=tmp_path,
    )

    text = out.read_text()
    assert "experiment:" not in text
    assert "design_parameters:" not in text
    assert "materialized_overrides:" not in text
    assert "ResourceFirst" in text
    assert result.metadata_path.endswith(".metadata.json")
    meta = json.loads(Path(result.metadata_path).read_text())
    assert meta["design_name"] == "design"
    assert meta["applied"]["model_path"] == "model.path"


def test_materializer_does_not_overwrite_model_path_when_requested_model_is_missing(tmp_path: Path):
    existing = tmp_path / "real.onnx"
    existing.write_text("dummy")
    base = tmp_path / "configs/examples/default_compile.yml"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(
        "model:\n"
        f"  path: {existing.name}\n"
        "notes:\n"
        "  parallel_policy: Balanced\n"
    )

    out = tmp_path / "design.yml"
    result = materialize_design_config(
        base,
        out,
        {"model_path": "examples/mnist.onnx", "policy": "balanced"},
        design_name="design",
        repo_root=tmp_path,
    )

    text = out.read_text()
    assert "examples/mnist.onnx" not in text
    assert f"path: {existing.name}" in text
    meta = json.loads(Path(result.metadata_path).read_text())
    assert meta["unapplied"]["model_path"] == "examples/mnist.onnx"
    assert "requested model file does not exist" in meta["skipped"]["model_path"]
