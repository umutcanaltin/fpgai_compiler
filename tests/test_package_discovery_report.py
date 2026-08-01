import json
from pathlib import Path

from fpgai.discovery import DiscoveryRequest, discover_packages, write_discovery_report
from test_package_discovery import _write_model_package


def test_discovery_report_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    _write_model_package(root / "model")
    result = discover_packages(DiscoveryRequest(configured_directories=(root,), include_builtin=False))

    first_json, first_md = write_discovery_report(result, tmp_path / "first")
    second_json, second_md = write_discovery_report(result, tmp_path / "second")

    assert first_json.read_text(encoding="utf-8") == second_json.read_text(encoding="utf-8")
    assert first_md.read_text(encoding="utf-8") == second_md.read_text(encoding="utf-8")
    assert json.loads(first_json.read_text(encoding="utf-8"))["schema"] == "fpgai.package-discovery/v1"
