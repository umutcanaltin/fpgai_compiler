from pathlib import Path

from fpgai.contracts.package_validation import validate_package_manifest
from fpgai.ecosystem import scaffold_contribution, supported_contribution_types


def test_scaffold_model_operator_hls_vhdl_packages_validate(tmp_path: Path) -> None:
    cases = [
        ("model", None, "example.demo_model"),
        ("operator", None, "example.demo_operator"),
        ("implementation", "hls_cpp", "example.demo_hls"),
        ("implementation", "vhdl", "example.demo_vhdl"),
    ]
    for asset_type, language, package_id in cases:
        root = scaffold_contribution(
            tmp_path / package_id,
            asset_type=asset_type,
            package_id=package_id,
            language=language,
        )
        result = validate_package_manifest(root)
        assert result.ok, result.to_json()
        assert (root / "fpgai.yaml").is_file()
        assert (root / "README.md").is_file()
        assert "FPGAI Ecosystem" in (root / "README.md").read_text(encoding="utf-8")


def test_scaffold_refuses_nonempty_directory_without_force(tmp_path: Path) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    (root / "keep.txt").write_text("keep", encoding="utf-8")
    try:
        scaffold_contribution(root, asset_type="operator", package_id="example.operator.foo")
    except FileExistsError as exc:
        assert "ECO002" in str(exc)
    else:
        raise AssertionError("expected non-empty scaffold refusal")


def test_scaffold_contributor_friendly_hardware_aliases(tmp_path: Path) -> None:
    layer = scaffold_contribution(tmp_path / "layer", asset_type="layer", package_id="example.layer.foo")
    hls = scaffold_contribution(tmp_path / "hls", asset_type="hls", package_id="example.impl.foo_hls")
    vhdl = scaffold_contribution(tmp_path / "vhdl", asset_type="vhdl", package_id="example.impl.foo_vhdl")
    assert validate_package_manifest(layer).ok
    assert validate_package_manifest(hls).ok
    assert validate_package_manifest(vhdl).ok


def test_scaffold_generic_ecosystem_asset_types_validate(tmp_path: Path) -> None:
    for asset_type in ("board", "backend", "dataset", "optimizer", "loss", "reporter", "validation", "benchmark"):
        root = scaffold_contribution(
            tmp_path / asset_type,
            asset_type=asset_type,
            package_id=f"example.{asset_type}.demo",
        )
        result = validate_package_manifest(root)
        assert result.ok, result.to_json()


def test_supported_types_include_core_ecosystem_assets() -> None:
    types = set(supported_contribution_types())
    assert {"model", "operator", "implementation", "board", "backend", "optimizer", "loss", "dataset", "reporter"} <= types


def test_ecosystem_exports_existing_hls_and_vhdl_blocks_without_tool_execution(tmp_path: Path) -> None:
    import json
    from fpgai.ecosystem import export_implementation_artifact

    for package_id, expected_suffix, expected_language in (
        ("community.scale_bias_hls", "src/scale_bias.cpp", "hls_cpp"),
        ("community.scale_bias_vhdl", "rtl/scale_bias_vhdl.vhd", "vhdl"),
    ):
        out = export_implementation_artifact(
            package_id,
            tmp_path / package_id,
            project_root=Path(".").resolve(),
            directories=(Path("examples/packages").resolve(),),
        )
        manifest = json.loads((out / "artifact_manifest.json").read_text(encoding="utf-8"))
        assert manifest["language"] == expected_language
        assert manifest["tool_execution"] == {"vitis_hls": False, "vivado": False, "bitstream": False}
        assert (out / expected_suffix).is_file()
        assert (out / "implementation_contract.json").is_file()
        assert (out / "fpgai.yaml").is_file()


def test_scaffolded_hardware_implementation_has_loadable_contract_and_export_policy(tmp_path: Path) -> None:
    import yaml
    from fpgai.implementations import implementation_contract_from_manifest

    root = scaffold_contribution(tmp_path / "impl", asset_type="vhdl", package_id="example.impl.demo")
    contract = implementation_contract_from_manifest(root)
    raw = yaml.safe_load((root / "fpgai.yaml").read_text(encoding="utf-8"))
    assert contract.language == "vhdl"
    assert contract.backend == "vhdl"
    assert contract.operator_id == "example.impl.demo.operator"
    assert raw["export"]["requires_bitstream"] is False


def test_scaffold_declares_ecosystem_roles_and_numeric_validation(tmp_path: Path) -> None:
    import yaml
    expected = {"model": "model", "operator": "operator_semantics", "vhdl": "operator_implementation"}
    for kind, role in expected.items():
        root = scaffold_contribution(tmp_path / kind, asset_type=kind, package_id=f"example.{kind}.role")
        raw = yaml.safe_load((root / "fpgai.yaml").read_text())
        assert raw["ecosystem"]["role"] == role
        assert raw["validation"]["numeric"]["required"] is True
        if role == "operator_implementation":
            assert raw["implementation"]["implements"]["version"] == 1
