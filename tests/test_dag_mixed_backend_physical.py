from pathlib import Path

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.mixed_backend import (
    DAGMixedBackendPhysicalRequest,
    HLSPhysicalBinding,
    VHDLPhysicalBinding,
    emit_dag_mixed_backend_physical_project,
)
from fpgai.ir.graph import Graph


def _write_axis_rtl(path: Path, top: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{top}.v").write_text(
        f'''module {top}(ap_clk, input_r_TDATA, input_r_TVALID, input_r_TREADY, output_r_TDATA, output_r_TVALID, output_r_TREADY);
input ap_clk;
input [15:0] input_r_TDATA;
input input_r_TVALID;
output input_r_TREADY;
output [15:0] output_r_TDATA;
output output_r_TVALID;
input output_r_TREADY;
assign input_r_TREADY = output_r_TREADY;
assign output_r_TDATA = input_r_TDATA;
assign output_r_TVALID = input_r_TVALID;
endmodule
''',
        encoding="utf-8",
    )
    return path


def _graph() -> Graph:
    graph = Graph("fork_join")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    for name in ("input", "pre", "left", "right", "left_done", "right_done", "output"):
        graph.add_tensor(name, (1,), "int16")
    graph.add_op("Pre", ["input"], ["pre"], name="pre")
    graph.add_op("Split", ["pre"], ["left", "right"], name="split")
    graph.add_op("Left", ["left"], ["left_done"], name="left")
    graph.add_op("Right", ["right"], ["right_done"], name="right")
    graph.add_op("Merge", ["left_done", "right_done"], ["output"], name="merge")
    return graph


def test_dag_project_generates_explicit_grouped_split_and_merge(tmp_path: Path) -> None:
    pre = _write_axis_rtl(tmp_path / "pre", "pre_hls")
    left = _write_axis_rtl(tmp_path / "left", "left_hls")
    right = _write_axis_rtl(tmp_path / "right", "right_hls")
    split_contract = implementation_contract_from_manifest(Path("examples/packages/split_grouped_ready_valid_vhdl"))
    add_contract = implementation_contract_from_manifest(Path("examples/packages/add_grouped_ready_valid_vhdl"))

    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=_graph(),
            bindings={
                "pre": HLSPhysicalBinding("pre", pre, "pre_hls"),
                "split": VHDLPhysicalBinding("split", split_contract),
                "left": HLSPhysicalBinding("left", left, "left_hls"),
                "right": HLSPhysicalBinding("right", right, "right_hls"),
                "merge": VHDLPhysicalBinding("merge", add_contract),
            },
            expected_output=43,
        )
    )

    assert result.ok, [issue.to_dict() for issue in result.issues]
    wrapper = result.wrapper.read_text(encoding="utf-8")
    assert "split_grouped_ready_valid_vhdl" in wrapper
    assert "add_grouped_ready_valid_vhdl" in wrapper
    assert "tensor_left_ready" in wrapper
    assert "tensor_right_ready" in wrapper
    assert "node_1_output_ready" in wrapper
    assert "node_4_input_valid" in wrapper

    report = result.report_path.read_text(encoding="utf-8")
    assert '"dag": true' in report
    assert '"multi_input_vhdl": true' in report
    assert '"multi_output_vhdl": true' in report
    assert '"implicit_fanout": false' in report


def test_dag_profile_rejects_implicit_fanout(tmp_path: Path) -> None:
    graph = Graph("implicit_fanout")
    graph.inputs = ["input"]
    graph.outputs = ["a_out"]
    for name in ("input", "shared", "a_out", "b_out"):
        graph.add_tensor(name, (1,), "int16")
    graph.add_op("Pre", ["input"], ["shared"], name="pre")
    graph.add_op("A", ["shared"], ["a_out"], name="a")
    graph.add_op("B", ["shared"], ["b_out"], name="b")

    rtl = _write_axis_rtl(tmp_path / "rtl", "hls")
    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=graph,
            bindings={
                "pre": HLSPhysicalBinding("pre", rtl, "hls"),
                "a": HLSPhysicalBinding("a", rtl, "hls"),
                "b": HLSPhysicalBinding("b", rtl, "hls"),
            },
        )
    )
    assert not result.ok
    assert result.issues[0].code == "MIXDAG007"


def test_dag_staging_deduplicates_reused_hls_rtl(tmp_path: Path) -> None:
    shared = _write_axis_rtl(tmp_path / "shared", "shared_hls")
    split_contract = implementation_contract_from_manifest(Path("examples/packages/split_grouped_ready_valid_vhdl"))
    add_contract = implementation_contract_from_manifest(Path("examples/packages/add_grouped_ready_valid_vhdl"))
    graph = _graph()

    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=graph,
            bindings={
                "pre": HLSPhysicalBinding("pre", shared, "shared_hls"),
                "split": VHDLPhysicalBinding("split", split_contract),
                "left": HLSPhysicalBinding("left", shared, "shared_hls"),
                "right": HLSPhysicalBinding("right", shared, "shared_hls"),
                "merge": VHDLPhysicalBinding("merge", add_contract),
            },
        )
    )
    assert result.ok, [issue.to_dict() for issue in result.issues]
    staged = list((result.project_dir / "rtl").glob("*shared_hls.v"))
    assert len(staged) == 1


def test_vector_testbench_emits_exact_word_sequences() -> None:
    from fpgai.implementations.mixed_backend.dag_physical import _testbench_source

    source = _testbench_source(
        "probe",
        32,
        32,
        0,
        0,
        input_values=(0x01020304, 0xFFFFFFFF),
        expected_outputs=(0x05060708, 0x80000000),
    )
    assert "INPUT_COUNT = 2" in source
    assert "OUTPUT_COUNT = 2" in source
    assert "32'h01020304" in source
    assert "32'hffffffff" in source
    assert "xsim_exact" not in source
    assert "numeric mismatch at word %0d" in source
    assert "output_ready = 1;" in source
    assert "while (!(output_valid && output_ready)" in source
    assert "output handshake timeout at word %0d" in source
    assert "@(posedge clk);\n      #1;" in source
    assert "@(negedge clk);\n      input_data = input_words[i];" in source
    assert "input_valid = 1;" in source
    assert "input handshake timeout at word %0d" in source
    assert "first_input_accept_cycle" in source
    assert "first_output_accept_cycle" in source
    assert "FPGAI_DAG_MIXED_BACKEND_SIM_METRICS" in source
    assert "@(posedge clk);\n      #1;\n      if (first_input_accept_cycle < 0)" in source
    assert "input_data <= input_words[i]" not in source


def test_dag_hls_axis_sidebands_are_connected_and_counted(tmp_path: Path) -> None:
    rtl = tmp_path / "axis_rtl"
    rtl.mkdir()
    (rtl / "axis_top.v").write_text(
        """
module axis_top(
  ap_clk, ap_rst_n,
  in_stream_TDATA, in_stream_TVALID, in_stream_TREADY,
  in_stream_TKEEP, in_stream_TSTRB, in_stream_TLAST,
  out_stream_TDATA, out_stream_TVALID, out_stream_TREADY,
  out_stream_TKEEP, out_stream_TSTRB, out_stream_TLAST
);
input ap_clk;
input ap_rst_n;
input [31:0] in_stream_TDATA;
input in_stream_TVALID;
output in_stream_TREADY;
input [3:0] in_stream_TKEEP;
input [3:0] in_stream_TSTRB;
input in_stream_TLAST;
output [31:0] out_stream_TDATA;
output out_stream_TVALID;
input out_stream_TREADY;
output [3:0] out_stream_TKEEP;
output [3:0] out_stream_TSTRB;
output out_stream_TLAST;
assign in_stream_TREADY = out_stream_TREADY;
assign out_stream_TDATA = in_stream_TDATA;
assign out_stream_TVALID = in_stream_TVALID;
assign out_stream_TKEEP = in_stream_TKEEP;
assign out_stream_TSTRB = in_stream_TSTRB;
assign out_stream_TLAST = in_stream_TLAST;
endmodule
""",
        encoding="utf-8",
    )
    graph = Graph("axis_sidebands")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_tensor("x", (1,), "uint32")
    graph.add_tensor("y", (1,), "uint32")
    graph.add_op("Identity", ["x"], ["y"], name="axis")

    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=graph,
            bindings={
                "axis": HLSPhysicalBinding(
                    "axis",
                    rtl,
                    "axis_top",
                    input_streams=("in_stream",),
                    output_streams=("out_stream",),
                    input_packet_words=(4,),
                    output_packet_words=(4,),
                )
            },
            input_values=(1, 2, 3, 4),
            expected_outputs=(1, 2, 3, 4),
        )
    )
    assert result.ok
    wrapper = result.wrapper.read_text(encoding="utf-8")
    assert ".in_stream_TKEEP(node_0_input_keep_0)" in wrapper
    assert ".in_stream_TSTRB(node_0_input_strb_0)" in wrapper
    assert ".in_stream_TLAST(node_0_input_last_0)" in wrapper
    assert "node_0_input_word_index_0" in wrapper
    assert ".out_stream_TKEEP(node_0_output_keep_0)" in wrapper
    assert ".out_stream_TSTRB(node_0_output_strb_0)" in wrapper
    assert ".out_stream_TLAST(node_0_output_last_0)" in wrapper
    assert "node_0_output_protocol_error_0" in wrapper


def test_dag_vivado_characterization_uses_out_of_context_implementation(tmp_path: Path) -> None:
    rtl = _write_axis_rtl(tmp_path / "rtl", "core_hls")
    graph = Graph("core_characterization")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1,), "int16")
    graph.add_tensor("output", (1,), "int16")
    graph.add_op("Identity", ["input"], ["output"], name="core")

    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=graph,
            bindings={"core": HLSPhysicalBinding("core", rtl, "core_hls")},
            run_implementation=True,
        )
    )
    assert result.ok, [issue.to_dict() for issue in result.issues]
    run_tcl = result.run_tcl.read_text(encoding="utf-8")
    assert "synth_design -mode out_of_context -top fpgai_dag_mixed_backend_top" in run_tcl
    assert "place_design" in run_tcl
    assert "route_design" in run_tcl

    import json
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["implementation_context"] == "out_of_context_accelerator_core"
    tcl_text = result.run_tcl.read_text(encoding="utf-8")
    assert "set_property xsim.simulate.runtime {100 us} [get_filesets sim_1]" in tcl_text
    assert "launch_simulation\nclose_sim" in tcl_text
    assert "run all" not in tcl_text
    assert "run 500 ns" not in tcl_text


def test_parse_simulation_metrics_reports_whole_dag_cycle_measurements() -> None:
    from fpgai.implementations.mixed_backend.dag_physical import _parse_simulation_metrics

    metrics = _parse_simulation_metrics(
        "FPGAI_DAG_MIXED_BACKEND_SIM_METRICS "
        "first_input_cycle=5 last_input_cycle=11 first_output_cycle=23 last_output_cycle=29 "
        "input_count=4 output_count=4\n"
    )
    assert metrics is not None
    assert metrics["first_output_latency_cycles"] == 18
    assert metrics["packet_completion_latency_cycles"] == 24
    assert metrics["post_input_drain_cycles"] == 18
    assert metrics["output_accept_span_cycles"] == 6
    assert metrics["mean_output_interbeat_cycles"] == 2.0
    assert metrics["initiation_interval"] is None
    assert metrics["initiation_interval_status"] == "not_measured_single_packet_testbench"
