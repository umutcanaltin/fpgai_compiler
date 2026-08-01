from pathlib import Path

from fpgai.config.loader import load_config


def test_external_ecosystem_sections_are_public_config(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")
    path = tmp_path / "config.yml"
    path.write_text(
        """
version: 1
model: {path: PLACEHOLDER}
pipeline: {mode: inference}
operators: {supported: [ScaleBias]}
ecosystem:
  enabled: true
  package_directories: [packages]
  operator_packages: {enable: [community.scale_bias_operator]}
  trust: {community.scale_bias_operator: approved_for_reference}
implementations:
  enable: [community.scale_bias_hls]
""",
        encoding="utf-8",
    )
    path.write_text(path.read_text(encoding="utf-8").replace("PLACEHOLDER", str(model)), encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg.raw["ecosystem"]["enabled"] is True
    assert cfg.raw["implementations"]["enable"] == ["community.scale_bias_hls"]
