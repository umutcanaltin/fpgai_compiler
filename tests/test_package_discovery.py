from pathlib import Path

import yaml

from fpgai.discovery import DiscoveryRequest, discover_packages


def _write_model_package(root: Path, package_id: str = "community.model", version: str = "1.0.0") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Package\n", encoding="utf-8")
    (root / "model.onnx").write_bytes(b"onnx")
    manifest = {
        "schema": "fpgai.package/v1",
        "package": {"id": package_id, "name": package_id, "version": version, "asset_type": "model", "provider": package_id.split(".")[0]},
        "usage": {"platform_scope": "research", "permitted_uses": ["research", "experimentation", "validation", "benchmarking"], "production_path": "morfics"},
        "license": {"category": "research_only"},
        "compatibility": {"fpgai_contract": ">=1.0,<2.0"},
        "capabilities": {"inference": True, "training": {"forward": False}},
        "entrypoints": {"model": {"format": "onnx", "path": "model.onnx"}},
        "validation": {"declared_level": "unvalidated"},
    }
    (root / "fpgai.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return root


def test_project_local_package_is_discovered(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_model_package(project / "packages" / "example")

    result = discover_packages(DiscoveryRequest(project_root=project, include_builtin=False))

    assert result.ok
    assert [item.package_id for item in result.discovered] == ["community.model"]
    assert result.catalogue.models.resolve("community.model").ok


def test_absent_project_packages_directory_is_normal(tmp_path: Path) -> None:
    result = discover_packages(DiscoveryRequest(project_root=tmp_path, include_builtin=False))
    assert result.ok
    assert not result.warnings


def test_configured_missing_root_warns_or_fails_in_strict_mode(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    permissive = discover_packages(DiscoveryRequest(configured_directories=(missing,), include_builtin=False))
    strict = discover_packages(DiscoveryRequest(configured_directories=(missing,), include_builtin=False, strict=True))
    assert permissive.ok
    assert permissive.warnings[0].code == "PKGDISCW001"
    assert not strict.ok


def test_invalid_package_is_quarantined_without_blocking_valid_package(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    _write_model_package(root / "valid")
    broken = root / "broken"
    broken.mkdir(parents=True)
    (broken / "fpgai.yaml").write_text("schema: invalid\n", encoding="utf-8")

    result = discover_packages(DiscoveryRequest(configured_directories=(root,), include_builtin=False))

    assert result.ok
    assert len(result.discovered) == 1
    assert len(result.quarantined) == 1
