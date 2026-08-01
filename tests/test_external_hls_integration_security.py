from pathlib import Path

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.hls_integration import ExternalHLSProjectRequest, emit_external_hls_operator_project


def test_external_hls_integration_rejects_symlinked_source(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    (package / "src").mkdir(parents=True)
    (package / "include").mkdir()
    outside = tmp_path / "outside.cpp"
    outside.write_text("void bad() {}\n", encoding="utf-8")
    (package / "src" / "scale_bias.cpp").symlink_to(outside)
    (package / "include" / "scale_bias.hpp").write_text("#pragma once\n", encoding="utf-8")
    base = Path("examples/packages/scale_bias_hls/fpgai.yaml").read_text(encoding="utf-8")
    (package / "fpgai.yaml").write_text(base, encoding="utf-8")
    (package / "README.md").write_text("x", encoding="utf-8")
    (package / "LICENSE").write_text("x", encoding="utf-8")
    contract = implementation_contract_from_manifest(package)
    result = emit_external_hls_operator_project(
        ExternalHLSProjectRequest(tmp_path / "build", contract, "ScaleBias", {"scale": 1, "bias": 0}, 4, 4)
    )
    assert not result.ok
    assert result.issues[0].code == "HLSINT007"
