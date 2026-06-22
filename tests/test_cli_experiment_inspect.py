from pathlib import Path
import json
import sys
import pytest

from fpgai.cli import main as cli_main


def test_experiment_inspect_json_output(tmp_path, monkeypatch):
    config = Path("configs/experiments/arxiv_paper.yml")
    assert config.exists(), "configs/experiments/arxiv_paper.yml is missing"

    out = tmp_path / "experiment_inspection.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fpgai",
            "experiment",
            "inspect",
            "--config",
            str(config),
            "--json-output",
            str(out),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli_main()

    assert exc.value.code == 0
    assert out.exists()

    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert data.get("kind") == "paper_experiment"
    assert data.get("valid") is True
