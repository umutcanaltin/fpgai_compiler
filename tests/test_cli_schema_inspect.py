from __future__ import annotations

from pathlib import Path
import json

import fpgai.cli as cli


def test_sweep_inspect_accepts_sweep_schema(tmp_path):
    cfg = tmp_path / "sweep.yml"
    cfg.write_text(
        """
name: smoke_sweep
defaults:
  model:
    path: models/mlp.onnx
parameters:
  precision:
    - fx8
    - fx16
command_template: "fpgai compile --config {config}"
design_name_template: "{precision}"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "sweep.json"

    rc = cli.inspect_sweep_config(str(cfg), json_output=str(out))

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert '"kind": "sweep"' in text
    assert '"valid": true' in text


def test_sweep_inspect_rejects_missing_parameters(tmp_path):
    cfg = tmp_path / "bad_sweep.yml"
    cfg.write_text(
        """
name: bad
defaults: {}
command_template: "fpgai compile --config {config}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rc = cli.inspect_sweep_config(str(cfg))

    assert rc == 1


def test_experiment_inspect_accepts_paper_schema(tmp_path):
    cfg = tmp_path / "paper.yml"
    cfg.write_text(
        """
version: 1
paper:
  title: "FPGAI: test"
inputs:
  vivado_summary: paper_experiments/vivado.csv
claim_levels:
  supported: "supported by artifacts"
limitations:
  - "No physical-board runtime claim."
""".strip()
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "paper.json"

    rc = cli.inspect_experiment_config(str(cfg), json_output=str(out))

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert '"kind": "paper_experiment"' in text
    assert '"valid": true' in text


def test_experiment_inspect_rejects_missing_sections(tmp_path):
    cfg = tmp_path / "bad_paper.yml"
    cfg.write_text(
        """
version: 1
paper:
  title: "Missing sections"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rc = cli.inspect_experiment_config(str(cfg))

    assert rc == 1


def test_sweep_inspect_accepts_design_points_schema(tmp_path):
    cfg = tmp_path / "design_points_sweep.yml"
    out = tmp_path / "inspection.json"
    cfg.write_text(
        """
name: design_points_sweep
defaults:
  board: kv260
design_points:
  - name: one
    command: "echo one"
  - name: two
    command: "echo two"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rc = cli.inspect_sweep_config(str(cfg), json_output=str(out))

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["valid"] is True
    assert payload["parameter_count"] == 0
    assert payload["design_point_count"] == 2
    assert payload["unknown_keys"] == []
