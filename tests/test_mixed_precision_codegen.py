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


def test_mixed_precision_top_rejects_streamed_weights() -> None:
    graph = _dense_graph()

    resolve_layerwise_precision(
        graph,
        _base_config(),
    )

    with pytest.raises(
        ValueError,
        match="requires weights mode 'embedded'",
    ):
        emit_top_cpp(
            graph,
            top_name="deeplearn",
            weights_mode="stream",
        )


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