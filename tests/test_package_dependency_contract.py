from __future__ import annotations

import copy
from pathlib import Path

import yaml

from fpgai.contracts.package_dependency import parse_dependencies
from fpgai.contracts.package_validation import validate_package_manifest
from fpgai.contracts.package_version import VersionRange, normalize_version_range


BASE = {
    "schema": "fpgai.package/v1",
    "package": {
        "id": "community.root_package",
        "name": "Root package",
        "version": "1.0.0",
        "asset_type": "dataset",
        "provider": "community",
        "description": "Dependency contract fixture",
    },
    "usage": {
        "platform_scope": "research",
        "permitted_uses": ["research", "experimentation", "validation", "benchmarking"],
        "production_path": "morfics",
    },
    "license": {"category": "open_source", "identifier": "Apache-2.0"},
    "compatibility": {"fpgai_contract": ">=1.0,<2.0"},
    "capabilities": {"inference": False, "training": {"forward": False}},
    "validation": {"declared_level": "unvalidated"},
}


def _write(tmp_path: Path, raw: dict) -> Path:
    root = tmp_path / raw["package"]["id"].replace(".", "_")
    root.mkdir(parents=True)
    (root / "README.md").write_text("# Test package\n", encoding="utf-8")
    (root / "fpgai.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return root


def test_exact_caret_tilde_and_pep440_ranges() -> None:
    assert normalize_version_range("1.2.3") == "==1.2.3"
    assert VersionRange.parse("^1.2.3").contains("1.9.0")
    assert not VersionRange.parse("^1.2.3").contains("2.0.0")
    assert VersionRange.parse("^0.2.3").contains("0.2.9")
    assert not VersionRange.parse("^0.2.3").contains("0.3.0")
    assert VersionRange.parse("~1.2").contains("1.2.9")
    assert not VersionRange.parse("~1.2").contains("1.3.0")
    assert VersionRange.parse(">=1.0,<2.0").contains("1.5.0")
    assert VersionRange.parse("*").contains("99.0.0")


def test_dependency_manifest_parsing() -> None:
    raw = copy.deepcopy(BASE)
    raw["dependencies"] = [
        {"package": "fpgai.operator.conv2d", "version": ">=1.0,<2.0", "required": True},
        {"package": "community.axi_utils", "version": "^2.1", "required": False},
    ]
    dependencies = parse_dependencies(raw)
    assert [item.package_id for item in dependencies] == [
        "fpgai.operator.conv2d",
        "community.axi_utils",
    ]
    assert dependencies[1].version.contains("2.8.0")
    assert not dependencies[1].required


def test_invalid_dependency_has_stable_manifest_error(tmp_path: Path) -> None:
    raw = copy.deepcopy(BASE)
    raw["dependencies"] = [{"package": "Bad ID", "version": "not a range", "required": "yes"}]
    result = validate_package_manifest(_write(tmp_path, raw))
    assert "PKG015" in {issue.code for issue in result.errors}


def test_duplicate_dependency_declaration_is_rejected(tmp_path: Path) -> None:
    raw = copy.deepcopy(BASE)
    raw["dependencies"] = [
        {"package": "fpgai.operator.conv2d", "version": "^1.0"},
        {"package": "fpgai.operator.conv2d", "version": "^1.1"},
    ]
    result = validate_package_manifest(_write(tmp_path, raw))
    assert "PKG015" in {issue.code for issue in result.errors}
