from __future__ import annotations

import numpy as np
import pytest

from fpgai.backends.hls.emit.layers_conv import (
    emit_conv_h,
)
from fpgai.backends.hls.emit.layers_dense import (
    emit_dense_h,
)
from fpgai.backends.hls.emit.params_cpp import (
    emit_params_cpp,
)
from fpgai.backends.hls.emit.params_h import (
    emit_params_h,
)
from fpgai.backends.hls.emit.top_cpp import (
    emit_top_cpp,
)
from fpgai.backends.hls.emit.types_h import (
    emit_types_h,
)
from fpgai.engine.layerwise_precision import (
    resolve_layerwise_precision,
)
from fpgai.ir import Graph


def _spec(
    total_bits: int,
    int_bits: int,
) -> dict[str, object]:
    return {
        "type": "ap_fixed",
        "total_bits": total_bits,
        "int_bits": int_bits,
    }


def _base_config() -> dict:
    return {
        "numerics": {
            "defaults": {
                "activation": _spec(16, 6),
                "weight": _spec(16, 6),
                "bias": _spec(24, 10),
                "accum": _spec(24, 10),
            },
            "layers": [],
        }
    }


def _dense_graph(
    *,
    include_bias: bool = True,
) -> Graph:
    graph = Graph("dense_mixed_precision")

    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor(
        "input",
        (1, 4),
    )
    graph.add_tensor(
        "hidden",
        (1, 3),
    )
    graph.add_tensor(
        "relu_output",
        (1, 3),
    )
    graph.add_tensor(
        "output",
        (1, 2),
    )

    dense0_weights = np.arange(
        12,
        dtype=np.float32,
    ).reshape(3, 4)

    dense1_weights = np.arange(
        6,
        dtype=np.float32,
    ).reshape(2, 3)

    graph.constants["dense0_weights"] = (
        dense0_weights
    )
    graph.constants["dense1_weights"] = (
        dense1_weights
    )

    graph.add_tensor(
        "dense0_weights",
        dense0_weights.shape,
    )
    graph.add_tensor(
        "dense1_weights",
        dense1_weights.shape,
    )

    dense0_inputs = [
        "input",
        "dense0_weights",
    ]
    dense1_inputs = [
        "relu_output",
        "dense1_weights",
    ]

    if include_bias:
        dense0_bias = np.array(
            [
                0.1,
                0.2,
                0.3,
            ],
            dtype=np.float32,
        )
        dense1_bias = np.array(
            [
                0.4,
                0.5,
            ],
            dtype=np.float32,
        )

        graph.constants["dense0_bias"] = (
            dense0_bias
        )
        graph.constants["dense1_bias"] = (
            dense1_bias
        )

        graph.add_tensor(
            "dense0_bias",
            dense0_bias.shape,
        )
        graph.add_tensor(
            "dense1_bias",
            dense1_bias.shape,
        )

        dense0_inputs.append(
            "dense0_bias"
        )
        dense1_inputs.append(
            "dense1_bias"
        )

    graph.add_op(
        "Dense",
        dense0_inputs,
        ["hidden"],
        name="dense0",
        attrs={
            "in_features": 4,
            "out_features": 3,
        },
    )

    graph.add_op(
        "Relu",
        ["hidden"],
        ["relu_output"],
        name="relu0",
    )

    graph.add_op(
        "Dense",
        dense1_inputs,
        ["output"],
        name="dense1",
        attrs={
            "in_features": 3,
            "out_features": 2,
        },
    )

    return graph


def _conv_graph(
    *,
    include_bias: bool = True,
) -> Graph:
    graph = Graph("conv_mixed_precision")

    graph.inputs = ["input"]
    graph.outputs = ["output"]

    weights = np.arange(
        18,
        dtype=np.float32,
    ).reshape(2, 1, 3, 3)

    graph.constants["conv_weights"] = weights

    graph.add_tensor(
        "input",
        (1, 1, 5, 5),
    )
    graph.add_tensor(
        "conv_weights",
        weights.shape,
    )
    graph.add_tensor(
        "output",
        (1, 2, 3, 3),
    )

    inputs = [
        "input",
        "conv_weights",
    ]

    if include_bias:
        bias = np.array(
            [
                0.25,
                -0.25,
            ],
            dtype=np.float32,
        )

        graph.constants["conv_bias"] = bias
        graph.add_tensor(
            "conv_bias",
            bias.shape,
        )
        inputs.append(
            "conv_bias"
        )

    graph.add_op(
        "Conv",
        inputs,
        ["output"],
        name="conv0",
        attrs={
            "strides": [1, 1],
            "pads": [0, 0, 0, 0],
            "dilations": [1, 1],
        },
    )

    return graph


def test_types_header_emits_distinct_layer_types() -> None:
    graph = _dense_graph()

    config = _base_config()
    config["numerics"]["layers"] = [
        {
            "match": {
                "name": "dense0",
            },
            "activation": _spec(12, 4),
            "weight": _spec(9, 3),
            "bias": _spec(15, 6),
            "accum": _spec(20, 8),
        },
        {
            "match": {
                "name": "dense1",
            },
            "activation": _spec(10, 3),
            "weight": _spec(8, 2),
            "bias": _spec(14, 5),
            "accum": _spec(18, 7),
        },
    ]

    resolve_layerwise_precision(
        graph,
        config,
    )

    header = emit_types_h(
        graph,
        top_name="deeplearn",
        raw_cfg=config,
    )

    assert (
        "typedef ap_fixed<12,4> "
        "op0_act_t;"
        in header
    )
    assert (
        "typedef ap_fixed<9,3> "
        "op0_wgt_t;"
        in header
    )
    assert (
        "typedef ap_fixed<15,6> "
        "op0_bias_t;"
        in header
    )
    assert (
        "typedef ap_fixed<20,8> "
        "op0_acc_t;"
        in header
    )

    assert (
        "typedef ap_fixed<10,3> "
        "op2_act_t;"
        in header
    )
    assert (
        "typedef ap_fixed<8,2> "
        "op2_wgt_t;"
        in header
    )
    assert (
        "typedef ap_fixed<14,5> "
        "op2_bias_t;"
        in header
    )
    assert (
        "typedef ap_fixed<18,7> "
        "op2_acc_t;"
        in header
    )


def test_parameter_declarations_use_layer_types() -> None:
    graph = _dense_graph()

    config = _base_config()
    config["numerics"]["layers"] = [
        {
            "match": {
                "name": "dense0",
            },
            "weight": _spec(9, 3),
            "bias": _spec(15, 6),
        },
        {
            "match": {
                "name": "dense1",
            },
            "weight": _spec(8, 2),
            "bias": _spec(14, 5),
        },
    ]

    resolve_layerwise_precision(
        graph,
        config,
    )

    header = emit_params_h(
        graph,
        weights_mode="embedded",
    )

    assert (
        "extern const op0_wgt_t W0[12];"
        in header
    )
    assert (
        "extern const op0_bias_t B0[3];"
        in header
    )
    assert (
        "extern const op2_wgt_t W1[6];"
        in header
    )
    assert (
        "extern const op2_bias_t B1[2];"
        in header
    )


def test_parameter_definitions_use_layer_types() -> None:
    graph = _dense_graph()

    config = _base_config()
    config["numerics"]["layers"] = [
        {
            "match": {
                "name": "dense0",
            },
            "weight": _spec(9, 3),
            "bias": _spec(15, 6),
        },
        {
            "match": {
                "name": "dense1",
            },
            "weight": _spec(8, 2),
            "bias": _spec(14, 5),
        },
    ]

    resolve_layerwise_precision(
        graph,
        config,
    )

    source = emit_params_cpp(
        graph,
        weights_mode="embedded",
    )

    assert (
        "const op0_wgt_t W0[12]"
        in source
    )
    assert (
        "const op0_bias_t B0[3]"
        in source
    )
    assert (
        "const op2_wgt_t W1[6]"
        in source
    )
    assert (
        "const op2_bias_t B1[2]"
        in source
    )


def test_bias_free_dense_emits_zero_bias() -> None:
    graph = _dense_graph(
        include_bias=False,
    )

    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    header = emit_params_h(
        graph,
        weights_mode="embedded",
    )
    source = emit_params_cpp(
        graph,
        weights_mode="embedded",
    )

    assert (
        "extern const op0_bias_t B0[3];"
        in header
    )
    assert (
        "extern const op2_bias_t B1[2];"
        in header
    )

    assert (
        "const op0_bias_t B0[3] = { "
        "0, 0, 0 };"
        in source
    )
    assert (
        "const op2_bias_t B1[2] = { "
        "0, 0 };"
        in source
    )


def test_dense_top_calls_use_mixed_types() -> None:
    graph = _dense_graph()

    config = _base_config()
    config["numerics"]["layers"] = [
        {
            "match": {
                "name": "dense0",
            },
            "activation": _spec(12, 4),
            "weight": _spec(9, 3),
            "bias": _spec(15, 6),
            "accum": _spec(20, 8),
        },
        {
            "match": {
                "name": "relu0",
            },
            "activation": _spec(11, 4),
        },
        {
            "match": {
                "name": "dense1",
            },
            "activation": _spec(10, 3),
            "weight": _spec(8, 2),
            "bias": _spec(14, 5),
            "accum": _spec(18, 7),
        },
    ]

    resolve_layerwise_precision(
        graph,
        config,
    )

    source = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
    )

    assert "op0_act_t layer_in[4];" in source
    assert "op0_act_t layer_0_out[3];" in source
    assert "op1_act_t layer_1_out[3];" in source
    assert "op2_act_t layer_2_out[2];" in source

    assert (
        "dense_out_in<"
        "4, 3, "
        "op0_act_t, "
        "op0_act_t, "
        "op0_wgt_t, "
        "op0_bias_t, "
        "op0_acc_t"
        ">"
        in source
    )

    assert (
        "relu_typed<"
        "3, "
        "op0_act_t, "
        "op1_act_t"
        ">"
        in source
    )

    assert (
        "dense_out_in<"
        "3, 2, "
        "op1_act_t, "
        "op2_act_t, "
        "op2_wgt_t, "
        "op2_bias_t, "
        "op2_acc_t"
        ">"
        in source
    )


def test_conv_top_call_uses_mixed_types() -> None:
    graph = _conv_graph()

    config = _base_config()
    config["numerics"]["layers"] = [
        {
            "match": {
                "op_type": "Conv",
            },
            "activation": _spec(13, 5),
            "weight": _spec(9, 3),
            "bias": _spec(17, 7),
            "accum": _spec(22, 9),
        }
    ]

    resolve_layerwise_precision(
        graph,
        config,
    )

    source = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
    )

    assert "op0_act_t layer_in[25];" in source
    assert "op0_act_t layer_0_out[18];" in source

    assert (
        "conv2d<"
        "5, 5, 1, "
        "3, 3, 2, "
        "3, 1, 0, "
        "op0_act_t, "
        "op0_act_t, "
        "op0_wgt_t, "
        "op0_bias_t, "
        "op0_acc_t"
        ">"
        in source
    )


def test_bias_free_conv_emits_zero_bias() -> None:
    graph = _conv_graph(
        include_bias=False,
    )

    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    header = emit_params_h(
        graph,
        weights_mode="embedded",
    )
    source = emit_params_cpp(
        graph,
        weights_mode="embedded",
    )

    assert (
        "extern const op0_bias_t B0[2];"
        in header
    )
    assert (
        "const op0_bias_t B0[2] = { "
        "0, 0 };"
        in source
    )


def test_dense_header_is_type_generic() -> None:
    header = emit_dense_h()

    assert "typename IN_T = act_t" in header
    assert "typename OUT_T = act_t" in header
    assert "typename WGT_T = wgt_t" in header
    assert "typename BIAS_T = bias_t" in header
    assert "typename ACC_T = acc_t" in header


def test_conv_header_is_type_generic() -> None:
    header = emit_conv_h()

    assert "typename IN_T = act_t" in header
    assert "typename OUT_T = act_t" in header
    assert "typename WGT_T = wgt_t" in header
    assert "typename BIAS_T = bias_t" in header
    assert "typename ACC_T = acc_t" in header


def test_mixed_precision_top_accepts_streamed_weights_as_planning_mode() -> None:
    graph = _dense_graph()

    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    source = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="stream",
    )

    assert "Requested weights mode: stream" in source
    assert "Non-embedded weight modes are represented in the memory/compile plan" in source
    assert "W0" in source
    assert "B0" in source


def test_mixed_precision_top_rejects_unresolved_add() -> None:
    graph = Graph("add_test")

    graph.inputs = [
        "left",
        "right",
    ]
    graph.outputs = [
        "output",
    ]

    graph.add_tensor(
        "left",
        (1, 4),
    )
    graph.add_tensor(
        "right",
        (1, 4),
    )
    graph.add_tensor(
        "output",
        (1, 4),
    )

    graph.add_op(
        "Add",
        [
            "left",
            "right",
        ],
        ["output"],
        name="add0",
    )

    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    with pytest.raises(
        RuntimeError,
        match="requires tensor liveness",
    ):
        emit_top_cpp(
            graph,
            top_name="deeplearn",
            weights_mode="embedded",
        )

def test_precision_mode_materialization_changes_generated_types() -> None:
    from fpgai.experiments.config_materializer import (
        apply_parameter_overrides,
    )

    base_config = _base_config()
    base_config["analysis"] = {
        "precision_sweep": {
            "candidates": [
                {
                    "name": "fx8_3",
                    "defaults": {
                        "activation": _spec(8, 3),
                        "weight": _spec(8, 3),
                        "bias": _spec(16, 6),
                        "accum": _spec(16, 6),
                    },
                },
                {
                    "name": "fx16_6",
                    "defaults": {
                        "activation": _spec(16, 6),
                        "weight": _spec(16, 6),
                        "bias": _spec(24, 10),
                        "accum": _spec(24, 10),
                    },
                },
            ]
        }
    }

    fx8_config, fx8_report = apply_parameter_overrides(
        base_config,
        {"precision_mode": "fx8_3"},
    )
    fx16_config, fx16_report = apply_parameter_overrides(
        base_config,
        {"precision_mode": "fx16_6"},
    )

    assert "numerics.defaults" in fx8_report.applied["precision_mode"]
    assert "numerics.precision_mode" in fx8_report.applied["precision_mode"]
    assert "analysis.precision_sweep.selected_candidate" in fx8_report.applied["precision_mode"]
    assert "numerics.defaults" in fx16_report.applied["precision_mode"]
    assert "numerics.precision_mode" in fx16_report.applied["precision_mode"]
    assert "analysis.precision_sweep.selected_candidate" in fx16_report.applied["precision_mode"]

    fx8_graph = _dense_graph()
    resolve_layerwise_precision(
        fx8_graph,
        fx8_config,
    )
    fx8_header = emit_types_h(
        fx8_graph,
        top_name="deeplearn",
        raw_cfg=fx8_config,
    )

    fx16_graph = _dense_graph()
    resolve_layerwise_precision(
        fx16_graph,
        fx16_config,
    )
    fx16_header = emit_types_h(
        fx16_graph,
        top_name="deeplearn",
        raw_cfg=fx16_config,
    )

    assert fx8_header != fx16_header

    assert "typedef ap_fixed<8,3> act_t;" in fx8_header
    assert "typedef ap_fixed<8,3> wgt_t;" in fx8_header
    assert "typedef ap_fixed<16,6> bias_t;" in fx8_header
    assert "typedef ap_fixed<16,6> acc_t;" in fx8_header

    assert "typedef ap_fixed<16,6> act_t;" in fx16_header
    assert "typedef ap_fixed<16,6> wgt_t;" in fx16_header
    assert "typedef ap_fixed<24,10> bias_t;" in fx16_header
    assert "typedef ap_fixed<24,10> acc_t;" in fx16_header



def test_weight_delivery_mode_changes_generated_hls_interfaces() -> None:
    graph = _dense_graph()
    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    embedded_h = emit_params_h(
        graph,
        weights_mode="embedded",
    )
    embedded_cpp = emit_params_cpp(
        graph,
        weights_mode="embedded",
    )
    embedded_top = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
    )

    stream_h = emit_params_h(
        graph,
        weights_mode="stream",
    )
    stream_cpp = emit_params_cpp(
        graph,
        weights_mode="stream",
    )
    stream_top = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="stream",
    )

    ddr_h = emit_params_h(
        graph,
        weights_mode="ddr",
    )
    ddr_cpp = emit_params_cpp(
        graph,
        weights_mode="ddr",
    )
    ddr_top = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="ddr",
    )

    assert "extern const op0_wgt_t W0[12];" in embedded_h
    assert "const op0_wgt_t W0[12]" in embedded_cpp
    assert "weight_stream" not in embedded_top
    assert "weights_mem" not in embedded_top

    assert "Runtime parameters are preloaded through the AXI weight stream." in stream_h
    assert "extern op0_wgt_t W0[12];" in stream_h
    assert "const op0_wgt_t W0_init[12]" in stream_cpp
    assert "hls::stream<axis_t>& weight_stream" in stream_top
    assert "#pragma HLS INTERFACE axis port=weight_stream" in stream_top
    assert "fpgai_load_stream_vector<op0_wgt_t, 12>" in stream_top

    assert "Runtime parameters are loaded from the external DDR/m_axi weight buffer." in ddr_h
    assert "extern op0_wgt_t W0[12];" in ddr_h
    assert "const op0_wgt_t W0_init[12]" in ddr_cpp
    assert "const ap_uint<32>* weights_mem" in ddr_top
    assert "#pragma HLS INTERFACE m_axi port=weights_mem" in ddr_top
    assert "fpgai_load_ddr_vector<op0_wgt_t, 12>" in ddr_top

    assert embedded_h != stream_h
    assert stream_h != ddr_h
    assert embedded_top != stream_top
    assert stream_top != ddr_top



def test_weight_storage_binding_changes_generated_hls_pragmas() -> None:
    graph = _dense_graph()
    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    bram_cpp = emit_params_cpp(
        graph,
        weights_mode="embedded",
        storage_impl="bram",
    )
    uram_cpp = emit_params_cpp(
        graph,
        weights_mode="embedded",
        storage_impl="uram",
    )
    lutram_cpp = emit_params_cpp(
        graph,
        weights_mode="embedded",
        storage_impl="lutram",
    )
    ddr_cpp = emit_params_cpp(
        graph,
        weights_mode="ddr",
        storage_impl="ddr",
    )

    assert "storage binding: bram requested for W0" in bram_cpp
    assert "storage binding: bram requested for B0" in bram_cpp

    assert "storage binding: uram requested for W0" in uram_cpp
    assert "storage binding: uram requested for B0" in uram_cpp

    assert "storage binding: lutram requested for W0" in lutram_cpp
    assert "storage binding: lutram requested for B0" in lutram_cpp

    assert "impl=bram" not in uram_cpp
    assert "impl=uram" not in bram_cpp
    assert "BIND_STORAGE variable=W0" not in ddr_cpp
    assert "BIND_STORAGE variable=B0" not in ddr_cpp
    assert "impl=bram" not in ddr_cpp
    assert "impl=uram" not in ddr_cpp
    assert "impl=lutram" not in ddr_cpp



def test_tensor_edge_communication_plan_models_input_weight_output_precision_and_compression() -> None:
    from types import SimpleNamespace

    from fpgai.engine.communication import make_communication_plan

    cfg = SimpleNamespace(
        raw={
            "communication": {
                "axi": {
                    "word_bits": 128,
                    "burst_len": 64,
                }
            },
            "data_movement": {
                "ps_pl": {
                    "input": {
                        "mode": "stream",
                        "size_bytes": 128,
                        "precision": {
                            "type": "ap_fixed",
                            "total_bits": 8,
                            "int_bits": 3,
                        },
                        "compression": {
                            "enabled": True,
                            "codec": "bitpack",
                        },
                    },
                    "weights": {
                        "mode": "ddr",
                        "compression": {
                            "enabled": True,
                            "codec": "rle",
                        },
                    },
                    "aux": {
                        "enabled": True,
                        "mode": "stream",
                        "size_bytes": 32,
                        "precision": {
                            "type": "ap_fixed",
                            "total_bits": 12,
                            "int_bits": 4,
                        },
                        "compression": {
                            "enabled": True,
                            "codec": "delta",
                        },
                    },
                },
                "pl_ps": {
                    "output": {
                        "mode": "stream",
                        "size_bytes": 64,
                        "precision": {
                            "type": "ap_fixed",
                            "total_bits": 16,
                            "int_bits": 6,
                        },
                        "compression": {
                            "enabled": False,
                        },
                    },
                },
            },
        }
    )

    memory_plan = SimpleNamespace(
        notes={
            "policy_name": "Latency-First",
        },
        placements=[
            SimpleNamespace(
                tensor_name="W0",
                kind="weight",
                region="DDR",
                size_bytes=256,
                double_buffer=True,
                notes={
                    "weight_bits": 8,
                    "reason": "unit-test",
                },
            )
        ],
    )

    plan = make_communication_plan(cfg, memory_plan)
    data = plan.to_dict()
    edges = {
        edge["tensor_name"]: edge
        for edge in data["edges"]
    }

    assert data["notes"]["planner"] == "tensor_edge_comm_v1"
    assert data["notes"]["scope"] == "input_weight_output_aux_tensor_edges"
    assert data["notes"]["contains_modeled_codecs"] is True
    assert data["notes"]["contains_hardware_codecs"] is False

    assert edges["input"]["direction"] == "PS_TO_PL"
    assert edges["input"]["precision_bits"] == 8
    assert edges["input"]["codec"] == "bitpack"
    assert edges["input"]["packed_bits"] == 8
    assert edges["input"]["transfer_bytes"] < edges["input"]["size_bytes"]
    assert edges["input"]["unpack_in_pl"] is True
    assert edges["input"]["implemented_in_hls"] is False

    assert edges["W0"]["direction"] == "PS_TO_PL"
    assert edges["W0"]["codec"] == "rle"
    assert edges["W0"]["transfer_bytes"] < edges["W0"]["size_bytes"]
    assert edges["W0"]["implemented_in_hls"] is False

    assert edges["output"]["direction"] == "PL_TO_PS"
    assert edges["output"]["precision_bits"] == 16
    assert edges["output"]["codec"] == "raw"
    assert edges["output"]["transfer_bytes"] == edges["output"]["size_bytes"]
    assert edges["output"]["implemented_in_hls"] is True

    assert edges["aux"]["direction"] == "PS_TO_PL"
    assert edges["aux"]["precision_bits"] == 12
    assert edges["aux"]["codec"] == "delta"
    assert edges["aux"]["transfer_bytes"] < edges["aux"]["size_bytes"]


def test_communication_plan_is_visible_in_generated_top_cpp_artifact() -> None:
    from types import SimpleNamespace

    from fpgai.backends.hls.emit.top_cpp import emit_top_cpp
    from fpgai.engine.communication import make_communication_plan

    graph = _dense_graph()
    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    cfg = SimpleNamespace(
        raw={
            "data_movement": {
                "ps_pl": {
                    "input": {
                        "size_bytes": 128,
                        "precision": {
                            "total_bits": 8,
                        },
                        "compression": {
                            "enabled": True,
                            "codec": "bitpack",
                        },
                    },
                    "weights": {
                        "mode": "embedded",
                        "compression": {
                            "enabled": False,
                        },
                    },
                },
                "pl_ps": {
                    "output": {
                        "size_bytes": 64,
                        "precision": {
                            "total_bits": 16,
                        },
                        "compression": {
                            "enabled": False,
                        },
                    }
                },
            }
        }
    )
    memory_plan = SimpleNamespace(notes={}, placements=[])
    communication_plan = make_communication_plan(cfg, memory_plan)

    source = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        communication_plan=communication_plan,
    )

    assert "FPGAI communication tensor-edge plan" in source
    assert "#define FPGAI_COMM_INPUT_PRECISION_BITS 8" in source
    assert "#define FPGAI_COMM_INPUT_TRANSFER_BYTES" in source
    assert "#define FPGAI_COMM_OUTPUT_PRECISION_BITS 16" in source
    assert "codec=bitpack" in source
    assert "implemented_in_hls=False" in source
