from pathlib import Path

from fpgai.discovery import DiscoveryRequest, discover_packages
from test_package_discovery import _write_model_package


def test_symlinked_search_root_is_not_scanned(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    result = discover_packages(DiscoveryRequest(configured_directories=(linked,), include_builtin=False))

    assert result.ok
    assert any(issue.code == "PKGDISC002" for issue in result.warnings)


def test_entrypoint_symlink_escape_is_quarantined(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    package = _write_model_package(root / "model")
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    (package / "model.onnx").unlink()
    (package / "model.onnx").symlink_to(outside)

    result = discover_packages(DiscoveryRequest(configured_directories=(root,), include_builtin=False))

    assert result.ok
    assert len(result.quarantined) == 1
    assert result.quarantined[0].manifest_errors[-1]["code"] == "PKGDISC004"


def test_discovery_does_not_import_or_execute_package_code(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "packages"
    _write_model_package(root / "model")

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess executed")))
    monkeypatch.setattr("importlib.import_module", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("plugin imported")))

    result = discover_packages(DiscoveryRequest(configured_directories=(root,), include_builtin=False))
    assert result.ok


def test_invalid_max_depth_is_rejected() -> None:
    result = discover_packages(DiscoveryRequest(max_depth=6, include_builtin=False))
    assert not result.ok
    assert result.errors[0].code == "PKGDISC001"
