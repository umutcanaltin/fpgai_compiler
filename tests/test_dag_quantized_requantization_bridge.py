from pathlib import Path

from fpgai.implementations.mixed_backend import (
    DAGMixedBackendPhysicalRequest,
    HLSPhysicalBinding,
    RequantizationPhysicalBinding,
    emit_dag_mixed_backend_physical_project,
)
from fpgai.ir.graph import Graph
from fpgai.quantization import QuantizationParameters, QuantizationSpec


def _params(bits: int, scale: float):
    spec = QuantizationSpec(bits=bits, scheme="symmetric", granularity="per_tensor")
    return QuantizationParameters(spec, scale, 0, -1.0, 1.0)


def _graph():
    graph = Graph("requant_test")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1,), "int8", quantization=_params(8, 0.5).to_dict())
    graph.add_tensor("wide_in", (1,), "int16", quantization=_params(16, 0.25).to_dict())
    graph.add_tensor("wide_out", (1,), "int16", quantization=_params(16, 0.25).to_dict())
    graph.add_tensor("output", (1,), "int8", quantization=_params(8, 0.5).to_dict())
    graph.add_op("Requantize", ["input"], ["wide_in"], name="rq_up")
    graph.add_op("Add1", ["wide_in"], ["wide_out"], name="hls_add")
    graph.add_op("Requantize", ["wide_out"], ["output"], name="rq_down")
    return graph


def test_requantization_binding_is_public():
    assert RequantizationPhysicalBinding("rq").backend == "requantization"


def test_quantized_dag_rejects_hls_width_mismatch(tmp_path: Path):
    # A fake 8-bit HLS top is intentionally connected to a 16-bit tensor.
    rtl = tmp_path / "rtl"
    rtl.mkdir()
    (rtl / "add1_axis.v").write_text(
        """module add1_axis(ap_clk, ap_rst_n, input_r_TDATA, input_r_TVALID, input_r_TREADY, output_r_TDATA, output_r_TVALID, output_r_TREADY);
input ap_clk;
input ap_rst_n;
input [7:0] input_r_TDATA;
input input_r_TVALID;
output input_r_TREADY;
output [7:0] output_r_TDATA;
output output_r_TVALID;
input output_r_TREADY;
assign input_r_TREADY = output_r_TREADY;
assign output_r_TVALID = input_r_TVALID;
assign output_r_TDATA = input_r_TDATA + 1;
endmodule
""",
        encoding="utf-8",
    )
    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=_graph(),
            bindings={
                "rq_up": RequantizationPhysicalBinding("rq_up"),
                "hls_add": HLSPhysicalBinding("hls_add", rtl, "add1_axis", input_streams=("input",), output_streams=("output",)),
                "rq_down": RequantizationPhysicalBinding("rq_down"),
            },
            input_value=7,
            expected_output=8,
        )
    )
    assert result.ok is False
    assert any(issue.code == "MIXDAG012" for issue in result.issues)


def test_quantized_dag_emits_compiler_owned_requantization(tmp_path: Path):
    rtl = tmp_path / "rtl16"
    rtl.mkdir()
    (rtl / "add1_axis.v").write_text(
        """module add1_axis(ap_clk, ap_rst_n, input_r_TDATA, input_r_TVALID, input_r_TREADY, output_r_TDATA, output_r_TVALID, output_r_TREADY);
input ap_clk;
input ap_rst_n;
input [15:0] input_r_TDATA;
input input_r_TVALID;
output input_r_TREADY;
output [15:0] output_r_TDATA;
output output_r_TVALID;
input output_r_TREADY;
assign input_r_TREADY = output_r_TREADY;
assign output_r_TVALID = input_r_TVALID;
assign output_r_TDATA = input_r_TDATA + 1;
endmodule
""",
        encoding="utf-8",
    )
    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out_ok",
            graph=_graph(),
            bindings={
                "rq_up": RequantizationPhysicalBinding("rq_up"),
                "hls_add": HLSPhysicalBinding("hls_add", rtl, "add1_axis", input_streams=("input",), output_streams=("output",)),
                "rq_down": RequantizationPhysicalBinding("rq_down"),
            },
            input_value=7,
            expected_output=8,
        )
    )
    assert result.ok is True
    wrapper = result.wrapper.read_text(encoding="utf-8")
    report = result.report_path.read_text(encoding="utf-8")
    assert "compiler-owned requantization bridge: 8 -> 16 bits" in wrapper
    assert "compiler-owned requantization bridge: 16 -> 8 bits" in wrapper
    assert '"heterogeneous_tensor_widths": true' in report
    assert '"requantization_bridges": true' in report


def test_requantization_rtl_uses_valid_negative_signed_literals(tmp_path: Path):
    rtl = tmp_path / "rtl16_signed"
    rtl.mkdir()
    (rtl / "add1_axis.v").write_text(
        """module add1_axis(ap_clk, ap_rst_n, input_r_TDATA, input_r_TVALID, input_r_TREADY, output_r_TDATA, output_r_TVALID, output_r_TREADY);
input ap_clk;
input ap_rst_n;
input [15:0] input_r_TDATA;
input input_r_TVALID;
output input_r_TREADY;
output [15:0] output_r_TDATA;
output output_r_TVALID;
input output_r_TREADY;
assign input_r_TREADY = output_r_TREADY;
assign output_r_TVALID = input_r_TVALID;
assign output_r_TDATA = input_r_TDATA + 1;
endmodule
""",
        encoding="utf-8",
    )
    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out_signed",
            graph=_graph(),
            bindings={
                "rq_up": RequantizationPhysicalBinding("rq_up"),
                "hls_add": HLSPhysicalBinding("hls_add", rtl, "add1_axis", input_streams=("input",), output_streams=("output",)),
                "rq_down": RequantizationPhysicalBinding("rq_down"),
            },
            input_value=7,
            expected_output=8,
        )
    )
    assert result.ok is True
    wrapper = result.wrapper.read_text(encoding="utf-8")
    assert "128'sd-128" not in wrapper
    assert "128'sd-32768" not in wrapper
    assert "-128'sd128" in wrapper
    assert "-128'sd32768" in wrapper
