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
