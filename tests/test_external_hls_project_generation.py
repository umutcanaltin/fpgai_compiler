import json
import subprocess
from pathlib import Path

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.hls_integration import ExternalHLSProjectRequest, emit_external_hls_operator_project


def _request(tmp_path: Path):
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_hls"))
    return ExternalHLSProjectRequest(
        out_dir=tmp_path / "build",
        contract=contract,
        operator_name="ScaleBias",
        operator_attributes={"scale": 2.0, "bias": 1.0},
        input_words=4,
        output_words=4,
    )


def test_external_hls_project_copies_sources_and_generates_wrapper(tmp_path: Path) -> None:
    package_source = Path("examples/packages/scale_bias_hls/src/scale_bias.cpp")
    before = package_source.read_bytes()
    result = emit_external_hls_operator_project(_request(tmp_path))
    assert result.ok, result.issues
    assert package_source.read_bytes() == before
    assert result.top_cpp and result.top_cpp.exists()
    text = result.top_cpp.read_text(encoding="utf-8")
    assert "scale_bias_hls(input, output, 4, 2.0f, 1.0f);" in text
    assert result.run_tcl and "external/community_scale_bias_hls/src/000_scale_bias.cpp" in result.run_tcl.read_text(encoding="utf-8")
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["ownership"]["package_root_modified"] is False
    assert payload["usage"]["production_path"] == "morfics"


def test_generated_external_hls_csim_harness_compiles_and_passes_with_host_compiler(tmp_path: Path) -> None:
    result = emit_external_hls_operator_project(_request(tmp_path))
    assert result.ok
    hls_dir = result.hls_dir
    source = result.copied_sources[0]
    header_dir = result.copied_headers[0].parent
    exe = tmp_path / "tb"
    completed = subprocess.run(
        [
            "g++",
            "-std=c++17",
            f"-I{hls_dir / 'include'}",
            f"-I{header_dir}",
            str(result.top_cpp),
            str(source),
            str(result.testbench_cpp),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "failures=0" in run.stdout


def test_external_hls_project_is_deterministic(tmp_path: Path) -> None:
    first = emit_external_hls_operator_project(_request(tmp_path))
    first_report = first.report_path.read_text(encoding="utf-8")
    second = emit_external_hls_operator_project(_request(tmp_path))
    assert second.report_path.read_text(encoding="utf-8") == first_report
