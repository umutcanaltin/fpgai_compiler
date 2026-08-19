from __future__ import annotations

import numpy as np
import pytest


def test_ptq_residual_cnn_lowers_to_integer_hls(tmp_path):
    pytest.importorskip("onnx")
    from scripts.make_quantized_residual_cnn_example import write_model
    from fpgai.frontend.onnx import import_onnx
    from fpgai.quantization import QuantizationSpec, calibrate_model_ptq, apply_model_ptq_to_hls_graph
    from fpgai.validation.mixed_external_hls import execute_mixed_graph_trace
    from fpgai.backends.hls.emit.types_h import emit_types_h
    from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp
    from fpgai.ir.liveness import analyze_tensor_liveness
    from fpgai.backends.hls.buffer_allocation import build_hls_buffer_allocation

    model = write_model(tmp_path / "residual.onnx")
    graph = import_onnx(str(model))
    # ONNX initializers are canonical IR tensors as well as constant values, so
    # PTQ/QAT and type/reporting passes can attach metadata to weights/biases.
    for name, values in graph.constants.items():
        assert name in graph.tensors
        assert graph.tensors[name].shape == tuple(np.asarray(values).shape)
        assert graph.tensors[name].dtype == str(np.asarray(values).dtype)

    rng = np.random.default_rng(4)
    samples = [rng.uniform(-1.0, 1.0, size=(1, 1, 4, 4)).astype(np.float32) for _ in range(8)]
    result = calibrate_model_ptq(
        graph,
        samples,
        trace_fn=lambda g, x: execute_mixed_graph_trace(g, None, x),
        activation_spec=QuantizationSpec(bits=8, scheme="symmetric", granularity="per_tensor", signed=True),
        weight_spec=QuantizationSpec(bits=8, scheme="symmetric", granularity="per_channel", signed=True, axis=0),
    )
    lowering = apply_model_ptq_to_hls_graph(graph, result)
    assert lowering.quantized_conv_nodes == ("conv0", "conv1")
    assert lowering.quantized_add_nodes == ("add0",)
    assert lowering.quantized_relu_nodes == ("relu0", "relu1")
    assert np.asarray(graph.constants["w0"]).dtype.kind in {"i", "u"}

    raw = {
        "numerics": {
            "defaults": {
                "activation": {"type": "ap_int", "bits": 8},
                "weight": {"type": "ap_int", "bits": 8},
                "bias": {"type": "ap_int", "bits": 32},
                "accum": {"type": "ap_int", "bits": 32},
            }
        }
    }
    types = emit_types_h(graph, top_name="deeplearn", raw_cfg=raw)
    assert "typedef ap_int<8> act_t;" in types
    assert "typedef ap_int<32> acc_t;" in types

    live = analyze_tensor_liveness(graph)
    alloc = build_hls_buffer_allocation(graph, raw_cfg=raw, tensor_liveness=live)
    top = emit_dag_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        raw_cfg=raw,
        tensor_liveness=live,
        buffer_allocation=alloc,
    )
    assert "conv2d_quantized<" in top
    assert "add_vec_quantized<" in top
    assert "relu_quantized<" in top
    assert "fpgai_q_mult_0" in top


def test_quantized_integer_activation_width_drives_dag_axis_unpack():
    from types import SimpleNamespace
    from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp

    class Tensor:
        def __init__(self, shape):
            self.shape = shape

    class Graph:
        inputs = ["input"]
        outputs = ["output"]
        constants = {}
        ops = [SimpleNamespace(
            name="relu0",
            op_type="Relu",
            inputs=["input"],
            outputs=["output"],
            attrs={"precision": {"activation": {"type": "ap_int", "bits": 8}}},
        )]
        tensors = {"input": Tensor((1, 4)), "output": Tensor((1, 4))}

        def get_tensor(self, name):
            return self.tensors[name]

    allocation = {
        "tensor_to_buffer": {"input": "buffer0", "output": "buffer1"},
        "slots": [
            {"name": "buffer0", "cpp_type": "ap_int<8>", "words": 4, "tensors": ["input"]},
            {"name": "buffer1", "cpp_type": "ap_int<8>", "words": 4, "tensors": ["output"]},
        ],
    }
    raw = {"numerics": {"defaults": {"activation": {"type": "ap_int", "bits": 8}}}}
    cpp = emit_dag_top_cpp(Graph(), top_name="deeplearn", weights_mode="embedded", raw_cfg=raw, buffer_allocation=allocation)
    assert "static const int FPGAI_ACT_BITS = 8;" in cpp
    assert "static const int FPGAI_ACT_PER_AXIS = 4;" in cpp
    assert "fpgai_unpack_axis_value<ap_int<8>, FPGAI_ACT_BITS>" in cpp


def test_dag_hls_control_protocol_is_selectable_for_stream_composition():
    from types import SimpleNamespace
    from fpgai.backends.hls.emit.dag_top_cpp import emit_dag_top_cpp

    class Tensor:
        def __init__(self, shape):
            self.shape = shape

    class Graph:
        inputs = ["input"]
        outputs = ["output"]
        constants = {}
        ops = [SimpleNamespace(
            name="relu0",
            op_type="Relu",
            inputs=["input"],
            outputs=["output"],
            attrs={"precision": {"activation": {"type": "ap_int", "bits": 8}}},
        )]
        tensors = {"input": Tensor((1, 4)), "output": Tensor((1, 4))}

        def get_tensor(self, name):
            return self.tensors[name]

    allocation = {
        "tensor_to_buffer": {"input": "buffer0", "output": "buffer1"},
        "slots": [
            {"name": "buffer0", "cpp_type": "ap_int<8>", "words": 4, "tensors": ["input"]},
            {"name": "buffer1", "cpp_type": "ap_int<8>", "words": 4, "tensors": ["output"]},
        ],
    }
    raw = {
        "numerics": {"defaults": {"activation": {"type": "ap_int", "bits": 8}}},
        "targets": {"hls": {"control_protocol": "ap_ctrl_none"}},
    }
    cpp = emit_dag_top_cpp(Graph(), top_name="deeplearn", weights_mode="embedded", raw_cfg=raw, buffer_allocation=allocation)
    assert "#pragma HLS INTERFACE ap_ctrl_none port=return" in cpp
    assert "s_axilite port=return" not in cpp


def test_quantized_linear_graph_still_selects_dag_hls_codegen(tmp_path):
    """Backend partitioning may linearize a PTQ graph without changing its lowering ABI."""
    from types import SimpleNamespace
    from fpgai.backends.hls.codegen import emit_hls_stub

    class Tensor:
        def __init__(self, shape):
            self.shape = shape

    class Graph:
        name = "quantized_linear_after_partition"
        inputs = ["input"]
        outputs = ["output"]
        constants = {}
        ops = [SimpleNamespace(
            name="relu0",
            op_type="Relu",
            inputs=["input"],
            outputs=["output"],
            attrs={
                "precision": {
                    "activation": {"type": "ap_int", "bits": 8, "total_bits": 8, "int_bits": 8},
                    "weight": {"type": "ap_int", "bits": 8, "total_bits": 8, "int_bits": 8},
                    "bias": {"type": "ap_int", "bits": 32, "total_bits": 32, "int_bits": 32},
                    "accum": {"type": "ap_int", "bits": 32, "total_bits": 32, "int_bits": 32},
                },
                "quantized_relu": {
                    "input_zero": 0,
                    "multiplier": 1073741824,
                    "shift": 30,
                    "output_zero": 0,
                    "qmin": -128,
                    "qmax": 127,
                    "rounding_mode": 0,
                    "saturation_mode": 0,
                },
            },
        )]
        tensors = {"input": Tensor((1, 4)), "output": Tensor((1, 4))}

        def get_tensor(self, name):
            return self.tensors[name]

    raw = {
        "numerics": {"defaults": {
            "activation": {"type": "ap_int", "bits": 8},
            "weight": {"type": "ap_int", "bits": 8},
            "bias": {"type": "ap_int", "bits": 32},
            "accum": {"type": "ap_int", "bits": 32},
        }},
        "targets": {"hls": {"control_protocol": "ap_ctrl_none"}},
    }
    project = emit_hls_stub(
        graph=Graph(),
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "weights_mode": "embedded",
            "pipeline_mode": "inference",
            "raw_cfg": raw,
            "run_csim": False,
            "run_csynth": False,
            "export_ip": False,
        },
    )
    source = project.top_cpp.read_text()
    assert "relu_quantized<" in source
    assert "#pragma HLS INTERFACE ap_ctrl_none port=return" in source
