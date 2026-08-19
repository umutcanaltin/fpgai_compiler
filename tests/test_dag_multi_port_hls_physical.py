from pathlib import Path

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.mixed_backend import (
    DAGMixedBackendPhysicalRequest,
    HLSPhysicalBinding,
    VHDLPhysicalBinding,
    emit_dag_mixed_backend_physical_project,
)
from fpgai.ir.graph import Graph


def _write_multi_axis_rtl(path: Path, top: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{top}.v").write_text(
        f'''module {top}(ap_clk,
    input_r_TDATA, input_r_TVALID, input_r_TREADY,
    left_r_TDATA, left_r_TVALID, left_r_TREADY,
    right_r_TDATA, right_r_TVALID, right_r_TREADY);
input ap_clk;
input [15:0] input_r_TDATA;
input input_r_TVALID;
output input_r_TREADY;
output [15:0] left_r_TDATA;
output left_r_TVALID;
input left_r_TREADY;
output [15:0] right_r_TDATA;
output right_r_TVALID;
input right_r_TREADY;
assign input_r_TREADY = left_r_TREADY & right_r_TREADY;
assign left_r_TDATA = input_r_TDATA;
assign right_r_TDATA = input_r_TDATA;
assign left_r_TVALID = input_r_TVALID & right_r_TREADY;
assign right_r_TVALID = input_r_TVALID & left_r_TREADY;
endmodule
''',
        encoding="utf-8",
    )
    return path


def _write_add_axis_rtl(path: Path, top: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{top}.v").write_text(
        f'''module {top}(ap_clk,
    left_done_r_TDATA, left_done_r_TVALID, left_done_r_TREADY,
    right_done_r_TDATA, right_done_r_TVALID, right_done_r_TREADY,
    output_r_TDATA, output_r_TVALID, output_r_TREADY);
input ap_clk;
input [15:0] left_done_r_TDATA;
input left_done_r_TVALID;
output left_done_r_TREADY;
input [15:0] right_done_r_TDATA;
input right_done_r_TVALID;
output right_done_r_TREADY;
output [15:0] output_r_TDATA;
output output_r_TVALID;
input output_r_TREADY;
assign left_done_r_TREADY = output_r_TREADY & right_done_r_TVALID;
assign right_done_r_TREADY = output_r_TREADY & left_done_r_TVALID;
assign output_r_TDATA = left_done_r_TDATA + right_done_r_TDATA;
assign output_r_TVALID = left_done_r_TVALID & right_done_r_TVALID;
endmodule
''',
        encoding="utf-8",
    )
    return path


def _graph() -> Graph:
    graph = Graph("multi_port_hls_mixed_dag")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    for name in (
        "input", "left", "right", "left_done", "right_done", "merged",
        "vhdl_left", "vhdl_right", "output",
    ):
        graph.add_tensor(name, (1,), "int16")
    graph.add_op("Split2", ["input"], ["left", "right"], name="hls_split")
    graph.add_op("Left", ["left"], ["left_done"], name="hls_left")
    graph.add_op("Right", ["right"], ["right_done"], name="hls_right")
    graph.add_op("Add2", ["left_done", "right_done"], ["merged"], name="hls_merge")
    graph.add_op("Split", ["merged"], ["vhdl_left", "vhdl_right"], name="vhdl_split")
    graph.add_op("Add", ["vhdl_left", "vhdl_right"], ["output"], name="vhdl_merge")
    return graph


def test_dag_physical_supports_multi_port_hls_and_vhdl(tmp_path: Path) -> None:
    split_rtl = _write_multi_axis_rtl(tmp_path / "split", "split2_axis")
    add_rtl = _write_add_axis_rtl(tmp_path / "add", "add2_axis")
    unary_rtl = tmp_path / "unary"
    unary_rtl.mkdir()
    (unary_rtl / "unary.v").write_text(
        '''module unary(ap_clk, input_r_TDATA, input_r_TVALID, input_r_TREADY, output_r_TDATA, output_r_TVALID, output_r_TREADY);
input ap_clk; input [15:0] input_r_TDATA; input input_r_TVALID; output input_r_TREADY;
output [15:0] output_r_TDATA; output output_r_TVALID; input output_r_TREADY;
assign input_r_TREADY=output_r_TREADY; assign output_r_TDATA=input_r_TDATA;
assign output_r_TVALID=input_r_TVALID; endmodule
''', encoding="utf-8")
    split_contract = implementation_contract_from_manifest(Path("examples/packages/split_grouped_ready_valid_vhdl"))
    add_contract = implementation_contract_from_manifest(Path("examples/packages/add_grouped_ready_valid_vhdl"))

    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=_graph(),
            bindings={
                "hls_split": HLSPhysicalBinding(
                    "hls_split", split_rtl, "split2_axis",
                    input_streams=("input",), output_streams=("left", "right"),
                ),
                "hls_left": HLSPhysicalBinding("hls_left", unary_rtl, "unary"),
                "hls_right": HLSPhysicalBinding("hls_right", unary_rtl, "unary"),
                "hls_merge": HLSPhysicalBinding(
                    "hls_merge", add_rtl, "add2_axis",
                    input_streams=("left_done", "right_done"), output_streams=("output",),
                ),
                "vhdl_split": VHDLPhysicalBinding("vhdl_split", split_contract),
                "vhdl_merge": VHDLPhysicalBinding("vhdl_merge", add_contract),
            },
            expected_output=44,
        )
    )
    assert result.ok, [issue.to_dict() for issue in result.issues]
    wrapper = result.wrapper.read_text(encoding="utf-8")
    assert ".left_r_TDATA(tensor_left_data)" in wrapper
    assert ".right_r_TDATA(tensor_right_data)" in wrapper
    assert ".left_done_r_TDATA(tensor_left_done_data)" in wrapper
    assert ".right_done_r_TDATA(tensor_right_done_data)" in wrapper
    report = result.report_path.read_text(encoding="utf-8")
    assert '"multi_port_hls": true' in report
    assert '"hls_multi_port_handshake": "axis_independent_ports"' in report


def test_multi_port_hls_requires_explicit_stream_mapping(tmp_path: Path) -> None:
    graph = Graph("ambiguous_multi_hls")
    graph.inputs = ["input"]
    graph.outputs = ["left"]
    for name in ("input", "left", "right"):
        graph.add_tensor(name, (1,), "int16")
    graph.add_op("Split2", ["input"], ["left", "right"], name="split")
    rtl = _write_multi_axis_rtl(tmp_path / "rtl", "split2_axis")
    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=graph,
            bindings={"split": HLSPhysicalBinding("split", rtl, "split2_axis")},
        )
    )
    assert not result.ok
    assert result.issues[0].code == "MIXRTL013"


def test_grouped_vhdl_split_uses_elastic_fanout_not_combinational_peer_ready(tmp_path: Path) -> None:
    split_rtl = _write_multi_axis_rtl(tmp_path / "split", "split2_axis")
    add_rtl = _write_add_axis_rtl(tmp_path / "add", "add2_axis")
    unary_rtl = tmp_path / "unary"
    unary_rtl.mkdir()
    (unary_rtl / "unary.v").write_text(
        '''module unary(ap_clk, input_r_TDATA, input_r_TVALID, input_r_TREADY, output_r_TDATA, output_r_TVALID, output_r_TREADY);
input ap_clk; input [15:0] input_r_TDATA; input input_r_TVALID; output input_r_TREADY;
output [15:0] output_r_TDATA; output output_r_TVALID; input output_r_TREADY;
assign input_r_TREADY=output_r_TREADY; assign output_r_TDATA=input_r_TDATA;
assign output_r_TVALID=input_r_TVALID; endmodule
''', encoding="utf-8")
    split_contract = implementation_contract_from_manifest(Path("examples/packages/split_grouped_ready_valid_vhdl"))
    add_contract = implementation_contract_from_manifest(Path("examples/packages/add_grouped_ready_valid_vhdl"))

    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=_graph(),
            bindings={
                "hls_split": HLSPhysicalBinding(
                    "hls_split", split_rtl, "split2_axis",
                    input_streams=("input",), output_streams=("left", "right"),
                ),
                "hls_left": HLSPhysicalBinding("hls_left", unary_rtl, "unary"),
                "hls_right": HLSPhysicalBinding("hls_right", unary_rtl, "unary"),
                "hls_merge": HLSPhysicalBinding(
                    "hls_merge", add_rtl, "add2_axis",
                    input_streams=("left_done", "right_done"), output_streams=("output",),
                ),
                "vhdl_split": VHDLPhysicalBinding("vhdl_split", split_contract),
                "vhdl_merge": VHDLPhysicalBinding("vhdl_merge", add_contract),
            },
            expected_output=44,
        )
    )
    assert result.ok, [issue.to_dict() for issue in result.issues]
    wrapper = result.wrapper.read_text(encoding="utf-8")
    assert "node_4_fanout_fifo_data_0" in wrapper
    assert "node_4_fanout_fifo_data_1" in wrapper
    assert "node_4_fanout_fifo_count_0" in wrapper
    assert "node_4_fanout_fifo_count_1" in wrapper
    assert "node_4_fanout_fifo_can_accept_0" in wrapper
    assert "node_4_fanout_fifo_can_accept_1" in wrapper
    assert "assign node_4_output_ready = (node_4_fanout_fifo_can_accept_0) & (node_4_fanout_fifo_can_accept_1);" in wrapper
    assert "assign tensor_vhdl_left_valid = (node_4_fanout_fifo_count_0 != 0);" in wrapper
    assert "assign tensor_vhdl_right_valid = (node_4_fanout_fifo_count_1 != 0);" in wrapper
    # The previous combinational peer-ready gating could form a zero-valid loop
    # when this split fed the grouped VHDL join directly.
    assert "tensor_vhdl_left_valid = node_4_output_valid &" not in wrapper
    assert "tensor_vhdl_right_valid = node_4_output_valid &" not in wrapper
    report = result.report_path.read_text(encoding="utf-8")
    assert '"physical_bridge": "elastic_grouped_fanout"' in report
