from pathlib import Path

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.mixed_backend.physical import (
    MixedBackendPhysicalRequest,
    _verilog_ports,
    emit_mixed_backend_physical_project,
)


def _fake_hls_rtl(root: Path) -> Path:
    rtl = root / "hls_rtl"
    rtl.mkdir()
    (rtl / "scale2_hls.v").write_text(
        """module scale2_hls(ap_clk, ap_rst, input_data, input_valid, output_data, output_valid);
input ap_clk;
input ap_rst;
input [15:0] input_data;
input input_valid;
output [15:0] output_data;
output output_valid;
assign output_data = input_data << 1;
assign output_valid = input_valid;
endmodule
""",
        encoding="utf-8",
    )
    return rtl


def test_verilog_port_parser_understands_hls_style_declarations(tmp_path):
    rtl = _fake_hls_rtl(tmp_path)
    ports = _verilog_ports(rtl / "scale2_hls.v", "scale2_hls")
    assert ports["input_data"] == ("input", 16)
    assert ports["output_valid"] == ("output", 1)


def test_emit_physical_hls_to_vhdl_project(tmp_path):
    rtl = _fake_hls_rtl(tmp_path)
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_vhdl"))
    result = emit_mixed_backend_physical_project(
        MixedBackendPhysicalRequest(out_dir=tmp_path / "out", hls_rtl_dir=rtl, hls_top="scale2_hls", vhdl_contract=contract)
    )
    assert result.ok, result.issues
    wrapper = result.wrapper.read_text(encoding="utf-8")
    tb = result.testbench.read_text(encoding="utf-8")
    tcl = result.run_tcl.read_text(encoding="utf-8")
    assert "scale2_hls u_hls" in wrapper
    assert f"{contract.top} u_vhdl" in wrapper
    assert "input_valid(hls_valid)" in wrapper
    assert "expected 14" in tb
    assert "read_vhdl" in tcl
    assert "read_verilog -sv" in tcl
    assert "synth_design" in tcl
