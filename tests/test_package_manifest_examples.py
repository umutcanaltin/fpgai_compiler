from pathlib import Path

from fpgai.contracts.package_validation import validate_package_manifest


EXAMPLES = (
    Path("examples/packages/hls_implementation"),
    Path("examples/packages/vhdl_implementation"),
    Path("examples/packages/onnx_operator"),
    Path("examples/packages/model"),
)


def test_all_maintained_package_examples_validate() -> None:
    for package_root in EXAMPLES:
        result = validate_package_manifest(package_root)
        assert result.ok, (package_root, result.to_dict())


def test_hls_and_vhdl_share_universal_manifest_contract() -> None:
    hls = validate_package_manifest(EXAMPLES[0])
    vhdl = validate_package_manifest(EXAMPLES[1])
    assert hls.schema == "fpgai.package/v1"
    assert vhdl.schema == "fpgai.package/v1"
