"""Synthetic vendor executables cannot automatically establish paper evidence."""
import stat
import json
from hashlib import sha256
from fpgai_research.experiments.results import summarize_case, _native_tool_identity
from fpgai_research.toolchains.vitis_hls.runner import run_vitis

def test_mock_process_reports_source_evidence_only(tmp_path):
    (tmp_path/"model.cpp").write_text("void fpgai_model() {}")
    (tmp_path/"tb.cpp").write_text("// mock")
    (tmp_path/"run_hls.tcl").write_text("exit\n")
    source=tmp_path/"model.cpp"
    bench=tmp_path/"tb.cpp"
    (tmp_path/"manifest.json").write_text(json.dumps({
        "cpp_sha256":sha256(source.read_bytes()).hexdigest(),
        "testbench_sha256":sha256(bench.read_bytes()).hexdigest(),
        "csim_sample_count":1,"csim_outputs_per_sample":1}))
    mock=tmp_path/"mock_vendor"
    mock.write_text("#!/bin/sh\necho 'FPGAI_METRICS samples=1 outputs=1 cosine=1 rmse=0 maxabs=0 failures=0'\necho 'FPGAI_CSIM_STATUS=completed'\n")
    mock.chmod(mock.stat().st_mode|stat.S_IXUSR)
    run_vitis(tmp_path,executable=str(mock))
    row=summarize_case(str(tmp_path))
    assert row["evidence"]=="E1-HLS-source"
    assert row["csim_passed"] is None
    assert any("identity/version" in msg for msg in row["warnings"])

def test_native_identity_requires_name_and_banner():
    sample={"executable":"/opt/Xilinx/Vitis/2023.2/bin/vitis_hls",
            "tool_version_output":"Vitis HLS v2023.2"}
    assert _native_tool_identity(sample,"vitis_hls")
    assert not _native_tool_identity({**sample,"executable":"/tmp/mock_vitis"},"vitis_hls")
    assert not _native_tool_identity({**sample,"tool_version_output":"Demo v1"},"vitis_hls")
    assert not _native_tool_identity({**sample,"tool_version_output":None},"vitis_hls")
    assert _native_tool_identity({"executable":"/opt/Xilinx/Vivado/2023.2/bin/vivado",
                                  "tool_version_output":"Vivado v2023.2"},"vivado")
