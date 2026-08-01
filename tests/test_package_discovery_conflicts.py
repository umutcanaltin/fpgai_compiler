from pathlib import Path

from fpgai.discovery import DiscoveryRequest, discover_packages
from test_package_discovery import _write_model_package


def test_identical_duplicate_is_deduplicated_by_source_priority(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    project = tmp_path / "project"
    first = _write_model_package(configured / "model")
    second = _write_model_package(project / "packages" / "model")
    # Preserve byte-identical manifests for identical hashes.
    (second / "fpgai.yaml").write_bytes((first / "fpgai.yaml").read_bytes())

    result = discover_packages(DiscoveryRequest(project_root=project, configured_directories=(configured,), include_builtin=False))

    assert result.ok
    assert result.discovered[0].source.value == "project_local"
    assert len(result.deduplicated) == 1


def test_conflicting_duplicate_is_not_registered(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    project = tmp_path / "project"
    _write_model_package(configured / "model")
    package = _write_model_package(project / "packages" / "model")
    text = (package / "fpgai.yaml").read_text(encoding="utf-8")
    (package / "fpgai.yaml").write_text(text.replace("name: community.model", "name: Different model"), encoding="utf-8")

    result = discover_packages(DiscoveryRequest(project_root=project, configured_directories=(configured,), include_builtin=False))

    assert not result.ok
    assert result.errors[0].code == "PKGDISC009"
    assert len(result.conflicts) == 2
    assert not result.catalogue.models.list_entries()


def test_different_versions_are_both_registered(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    _write_model_package(root / "v1", version="1.0.0")
    _write_model_package(root / "v2", version="1.1.0")

    result = discover_packages(DiscoveryRequest(configured_directories=(root,), include_builtin=False))

    assert result.ok
    assert result.catalogue.models.list_versions("community.model") == ("1.1.0", "1.0.0")
