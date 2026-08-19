from scripts.run_quantized_residual_cnn_mixed_backend import _pack_int8x4, _physical_graph


def test_pack_int8x4_matches_hls_lane_order() -> None:
    assert _pack_int8x4([-1, 2, -3, 4]) == (0x04FD02FF,)
    assert _pack_int8x4([0, 6, 19, 32, 45, 58, 70, 83]) == (0x20130600, 0x53463A2D)


def test_quantized_residual_cnn_physical_graph_has_hls_to_vhdl_boundary() -> None:
    graph = _physical_graph()
    assert [op.name for op in graph.ops] == [
        "hls_quantized_residual_cnn",
        "vhdl_quantized_packet_bridge",
    ]
    assert graph.tensors["input_packet"].dtype == "uint32"
    assert graph.tensors["output_packet"].dtype == "uint32"


def test_quantized_relu_vhdl_package_is_real_operator() -> None:
    from pathlib import Path
    from fpgai.implementations import implementation_contract_from_manifest

    contract = implementation_contract_from_manifest(
        Path("examples/packages/quantized_relu_int8x4_vhdl").resolve()
    )
    assert contract.backend == "vhdl"
    assert contract.top == "quantized_relu_int8x4_vhdl"


def test_residual_add_partition_physical_graph_has_explicit_split_and_two_vhdl_ops() -> None:
    graph = _physical_graph(partition_type="residual_add_relu")
    assert [op.name for op in graph.ops] == [
        "vhdl_input_split",
        "hls_quantized_residual_cnn",
        "vhdl_quantized_add",
        "vhdl_quantized_relu",
    ]
    add = graph.ops[2]
    assert list(add.inputs) == ["main_packet", "skip_packet"]
    assert list(add.outputs) == ["sum_packet"]
    assert graph.tensors["main_packet"].dtype == "uint32"


def test_residual_add_partition_can_keep_terminal_relu_in_hls() -> None:
    graph = _physical_graph(partition_type="residual_add_relu", residual_relu_backend="hls")
    assert [op.name for op in graph.ops] == [
        "vhdl_input_split",
        "hls_quantized_residual_cnn",
        "vhdl_quantized_add",
        "hls_quantized_relu",
    ]
    assert graph.ops[-1].op_type == "Relu"
    assert list(graph.ops[-1].inputs) == ["sum_packet"]


def test_quantized_int8x4_split_vhdl_package_is_multi_output() -> None:
    from pathlib import Path
    from fpgai.implementations import implementation_contract_from_manifest
    from fpgai.implementations.vhdl_integration import parse_vhdl_tensor_ports_ready_valid_abi
    contract = implementation_contract_from_manifest(Path("examples/packages/quantized_int8x4_split_vhdl").resolve())
    abi = parse_vhdl_tensor_ports_ready_valid_abi(contract)
    assert len(abi.inputs) == 1
    assert len(abi.outputs) == 2
    assert abi.data_widths == (32, 32, 32)


def test_residual_add_partition_emits_physical_dag(tmp_path) -> None:
    from pathlib import Path
    from fpgai.implementations import implementation_contract_from_manifest
    from fpgai.implementations.mixed_backend import (
        DAGMixedBackendPhysicalRequest,
        HLSPhysicalBinding,
        VHDLPhysicalBinding,
        emit_dag_mixed_backend_physical_project,
    )
    from fpgai.quantization import emit_quantized_add_int8x4_vhdl_package

    rtl = tmp_path / "hls"
    rtl.mkdir()
    (rtl / "body.v").write_text('''module body(ap_clk, in_stream_TDATA, in_stream_TVALID, in_stream_TREADY, out_stream_TDATA, out_stream_TVALID, out_stream_TREADY);
input ap_clk; input [31:0] in_stream_TDATA; input in_stream_TVALID; output in_stream_TREADY;
output [31:0] out_stream_TDATA; output out_stream_TVALID; input out_stream_TREADY;
assign in_stream_TREADY=out_stream_TREADY; assign out_stream_TDATA=in_stream_TDATA; assign out_stream_TVALID=in_stream_TVALID;
endmodule
''')
    q = lambda scale: {"spec": {"bits": 8, "granularity": "per_tensor", "rounding": "nearest", "saturation": "saturate"}, "scale": scale, "zero_point": 0}
    partition = {
        "schema": "fpgai.quantized-residual-operator-partition/v1",
        "partition_type": "residual_add_relu",
        "add": {
            "left_quantization": q(0.125), "right_quantization": q(0.25), "output_quantization": q(0.5),
            "lowering": {"left_zero": 0, "left_multiplier": 1, "left_shift": 2, "right_zero": 0, "right_multiplier": 1, "right_shift": 1, "output_zero": 0, "qmin": -128, "qmax": 127, "rounding_mode": 0, "saturation_mode": 0},
        },
    }
    add_root = emit_quantized_add_int8x4_vhdl_package(tmp_path / "generated_add", partition)
    graph = _physical_graph(partition_type="residual_add_relu")
    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out",
            graph=graph,
            bindings={
                "vhdl_input_split": VHDLPhysicalBinding("vhdl_input_split", implementation_contract_from_manifest(Path("examples/packages/quantized_int8x4_split_vhdl"))),
                "hls_quantized_residual_cnn": HLSPhysicalBinding("hls_quantized_residual_cnn", rtl, "body", input_streams=("in_stream",), output_streams=("out_stream",)),
                "vhdl_quantized_add": VHDLPhysicalBinding("vhdl_quantized_add", implementation_contract_from_manifest(add_root)),
                "vhdl_quantized_relu": VHDLPhysicalBinding("vhdl_quantized_relu", implementation_contract_from_manifest(Path("examples/packages/quantized_relu_int8x4_vhdl"))),
            },
            input_values=(0x04030201,),
            expected_outputs=(0x04030201,),
        )
    )
    assert result.ok, [issue.to_dict() for issue in result.issues]
    wrapper = result.wrapper.read_text()
    assert "quantized_int8x4_split_vhdl" in wrapper
    assert "quantized_add_int8x4_vhdl" in wrapper
    assert "quantized_relu_int8x4_vhdl" in wrapper
    assert "node_0_fanout_fifo_count_0" in wrapper
    assert "node_0_fanout_fifo_count_1" in wrapper


def test_residual_add_partition_can_buffer_full_skip_packet(tmp_path) -> None:
    from pathlib import Path
    from fpgai.implementations import implementation_contract_from_manifest
    from fpgai.implementations.mixed_backend import (
        DAGMixedBackendPhysicalRequest,
        HLSPhysicalBinding,
        VHDLPhysicalBinding,
        emit_dag_mixed_backend_physical_project,
    )
    from fpgai.quantization import emit_quantized_add_int8x4_vhdl_package

    rtl = tmp_path / "hls_buffered"
    rtl.mkdir()
    (rtl / "body.v").write_text(
        """module body(ap_clk, in_stream_TDATA, in_stream_TVALID, in_stream_TREADY, out_stream_TDATA, out_stream_TVALID, out_stream_TREADY);
input ap_clk; input [31:0] in_stream_TDATA; input in_stream_TVALID; output in_stream_TREADY;
output [31:0] out_stream_TDATA; output out_stream_TVALID; input out_stream_TREADY;
assign in_stream_TREADY=out_stream_TREADY; assign out_stream_TDATA=in_stream_TDATA; assign out_stream_TVALID=in_stream_TVALID;
endmodule
""",
        encoding="utf-8",
    )
    q = lambda scale: {
        "spec": {"bits": 8, "granularity": "per_tensor", "rounding": "nearest", "saturation": "saturate"},
        "scale": scale,
        "zero_point": 0,
    }
    partition = {
        "schema": "fpgai.quantized-residual-operator-partition/v1",
        "partition_type": "residual_add_relu",
        "add": {
            "left_quantization": q(0.125),
            "right_quantization": q(0.25),
            "output_quantization": q(0.5),
            "lowering": {
                "left_zero": 0, "left_multiplier": 1, "left_shift": 2,
                "right_zero": 0, "right_multiplier": 1, "right_shift": 1,
                "output_zero": 0, "qmin": -128, "qmax": 127,
                "rounding_mode": 0, "saturation_mode": 0,
            },
        },
    }
    add_root = emit_quantized_add_int8x4_vhdl_package(tmp_path / "generated_add_buffered", partition)
    graph = _physical_graph(partition_type="residual_add_relu")
    result = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=tmp_path / "out_buffered",
            graph=graph,
            bindings={
                "vhdl_input_split": VHDLPhysicalBinding(
                    "vhdl_input_split",
                    implementation_contract_from_manifest(Path("examples/packages/quantized_int8x4_split_vhdl")),
                ),
                "hls_quantized_residual_cnn": HLSPhysicalBinding(
                    "hls_quantized_residual_cnn", rtl, "body",
                    input_streams=("in_stream",), output_streams=("out_stream",),
                    input_packet_words=(4,), output_packet_words=(4,),
                ),
                "vhdl_quantized_add": VHDLPhysicalBinding(
                    "vhdl_quantized_add", implementation_contract_from_manifest(add_root)
                ),
                "vhdl_quantized_relu": VHDLPhysicalBinding(
                    "vhdl_quantized_relu",
                    implementation_contract_from_manifest(Path("examples/packages/quantized_relu_int8x4_vhdl")),
                ),
            },
            input_values=(1, 2, 3, 4),
            expected_outputs=(1, 2, 3, 4),
            fanout_buffer_depths={"skip_packet": 4},
        )
    )
    assert result.ok, [issue.to_dict() for issue in result.issues]
    wrapper = result.wrapper.read_text(encoding="utf-8")
    assert "node_0_fanout_fifo_data_1 [0:3]" in wrapper
    assert "node_0_fanout_fifo_count_1 < 3'd4" in wrapper
    report = result.report_path.read_text(encoding="utf-8")
    assert '"elastic_buffer_depth_words": 4' in report
