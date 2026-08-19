from pathlib import Path

from fpgai.config.loader import load_config


def test_config_preserves_model_format_and_framework(tmp_path: Path):
    model = tmp_path / "model.mlir"
    model.write_text('module { }', encoding="utf-8")
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(f'''\nversion: 1\nmodel:\n  path: {model}\n  format: stablehlo\n  framework: jax\npipeline:\n  mode: inference\noperators:\n  supported: [MatMul]\n''', encoding="utf-8")
    cfg = load_config(str(cfg_path))
    assert cfg.model.format == "stablehlo"
    assert cfg.model.framework == "jax"
