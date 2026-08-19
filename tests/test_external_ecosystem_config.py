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


def test_ecosystem_model_package_resolves_to_normal_model_path(tmp_path: Path) -> None:
    import yaml

    package_root = tmp_path / "packages" / "demo_model"
    (package_root / "model").mkdir(parents=True)
    model_file = package_root / "model" / "demo.onnx"
    model_file.write_bytes(b"model-bytes")
    (package_root / "README.md").write_text("# Demo model\n", encoding="utf-8")
    manifest = {
        "schema": "fpgai.package/v1",
        "package": {"id": "example.demo_model", "name": "Demo", "version": "1.0.0", "asset_type": "model", "provider": "example"},
        "usage": {"platform_scope": "research", "permitted_uses": ["research", "experimentation", "validation", "benchmarking"], "production_path": "morfics"},
        "license": {"category": "open_source", "identifier": "Apache-2.0"},
        "compatibility": {"fpgai_contract": ">=1.0,<2.0"},
        "capabilities": {"inference": True, "training": {"forward": False}},
        "entrypoints": {"model": {"path": "model/demo.onnx", "format": "onnx"}},
        "validation": {"declared_level": "reference_tested"},
    }
    (package_root / "fpgai.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    config = tmp_path / "config.yml"
    config.write_text(
        f"""
version: 1
pipeline: {{mode: inference}}
operators: {{supported: [Dense]}}
ecosystem:
  enabled: true
  project_root: {tmp_path}
  package_directories: [{tmp_path / 'packages'}]
  model_package: example.demo_model
""",
        encoding="utf-8",
    )
    cfg = load_config(str(config))
    assert Path(cfg.model.path) == model_file.resolve()
    assert cfg.model.format == "onnx"
    assert cfg.raw["metadata"]["ecosystem_model"]["package_id"] == "example.demo_model"
    numeric = cfg.raw["validation"]["numeric"]
    assert numeric["enabled"] is True
    assert numeric["policy"] == "enforce"
    assert numeric["reference"]["source"] == "framework"
    assert numeric["reference"]["compare_ir"] is True
    assert set(numeric["levels"]) >= {"model", "layer", "intermediate", "state"}
