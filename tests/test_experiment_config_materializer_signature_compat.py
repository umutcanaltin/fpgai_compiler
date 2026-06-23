from pathlib import Path

import yaml

from fpgai.experiments.config_materializer import materialize_design_config


def test_materialize_design_config_accepts_fourth_options_argument(tmp_path: Path):
    base = tmp_path / "configs/examples/default_compile.yml"
    base.parent.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "out.yml"
    base.write_text("notes:\n  parallel_policy: Balanced\n", encoding="utf-8")

    report = materialize_design_config(
        base,
        out,
        {"policy": "resource_first", "model_path": "missing.onnx"},
        {"enabled": True},
    )

    assert out.exists()
    assert Path(report.metadata_path).exists()
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "experiment" not in data
    assert data["notes"]["parallel_policy"] == "resource_first"
