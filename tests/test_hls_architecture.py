from __future__ import annotations

from fpgai.analysis.hls_architecture import build_hls_architecture
from fpgai.engine.models import CompilePlan, LayerDescriptor, LayerPlan


def _config() -> dict:
    return {
        "optimization": {
            "parallel_policy": "Balanced",
            "parallel": {
                "pe": 2,
                "simd": 2,
                "partition_factor": 2,
            },
        },
        "hls": {
            "pipeline_ii": 1,
        },
        "numerics": {
            "defaults": {
                "activation": {
                    "type": "ap_fixed",
                    "total_bits": 16,
                    "int_bits": 6,
                },
                "weight": {
                    "type": "ap_fixed",
                    "total_bits": 14,
                    "int_bits": 5,
                },
                "bias": {
                    "type": "ap_fixed",
                    "total_bits": 20,
                    "int_bits": 8,
                },
                "accum": {
                    "type": "ap_fixed",
                    "total_bits": 24,
                    "int_bits": 10,
                },
            },
        },
    }


def test_conv_architecture_models_pipeline_overlap() -> None:
    descriptor = LayerDescriptor(
        node_name="conv0",
        op_type="Conv",
        inputs=["input", "weights"],
        outputs=["output"],
        input_shapes=[
            (1, 4, 28, 28),
            (8, 4, 3, 3),
        ],
        output_shapes=[
            (1, 8, 26, 26),
        ],
        attrs={
            "kernel_shape": [3, 3],
            "group": 1,
        },
    )

    compile_plan = CompilePlan(
        clock_mhz=200.0,
        layer_plans=[
            LayerPlan(
                node_name="conv0",
                op_type="Conv",
                unroll={
                    "oc": 2,
                    "ic": 2,
                },
                pipeline_ii=1,
            ),
        ],
    )

    architecture = build_hls_architecture(
        [descriptor],
        _config(),
        compile_plan,
    )

    layer = architecture.layers[0]

    assert layer.op_type == "Conv"
    assert layer.pipeline_scope == "output_column"
    assert layer.pipeline_ii == 1

    # ceil(4 input channels / SIMD 2) * 3 * 3
    assert layer.reduction_iterations == 18

    # PE 2 * SIMD 2
    assert layer.explicit_lanes == 4

    # The output-column pipeline overlaps the reduction body.
    assert layer.pipeline_overlap == 18
    assert layer.effective_lanes == 72

    assert layer.arithmetic["activation_bits"] == 16
    assert layer.arithmetic["weight_bits"] == 14
    assert layer.arithmetic["accumulator_bits"] == 24

    # Generated Conv code casts operands to ACC_T before multiply.
    assert layer.arithmetic["multiply_left_bits"] == 24
    assert layer.arithmetic["multiply_right_bits"] == 24
    assert layer.arithmetic["effective_multiplier_units"] == 72


def test_conv_pipeline_ii_reduces_overlap() -> None:
    descriptor = LayerDescriptor(
        node_name="conv0",
        op_type="Conv",
        inputs=["input", "weights"],
        outputs=["output"],
        input_shapes=[
            (1, 4, 28, 28),
            (8, 4, 3, 3),
        ],
        output_shapes=[
            (1, 8, 26, 26),
        ],
        attrs={
            "kernel_shape": [3, 3],
        },
    )

    compile_plan = CompilePlan(
        layer_plans=[
            LayerPlan(
                node_name="conv0",
                op_type="Conv",
                unroll={
                    "oc": 1,
                    "ic": 1,
                },
                pipeline_ii=2,
            ),
        ],
    )

    architecture = build_hls_architecture(
        [descriptor],
        _config(),
        compile_plan,
    )

    layer = architecture.layers[0]

    assert layer.reduction_iterations == 36
    assert layer.pipeline_overlap == 18
    assert layer.explicit_lanes == 1
    assert layer.effective_lanes == 18


def test_dense_architecture_uses_explicit_unroll_only() -> None:
    descriptor = LayerDescriptor(
        node_name="dense0",
        op_type="Dense",
        inputs=["input", "weights", "bias"],
        outputs=["output"],
        input_shapes=[
            (1, 128),
            (10, 128),
            (10,),
        ],
        output_shapes=[
            (1, 10),
        ],
        attrs={
            "in_features": 128,
            "out_features": 10,
        },
    )

    compile_plan = CompilePlan(
        layer_plans=[
            LayerPlan(
                node_name="dense0",
                op_type="Dense",
                unroll={
                    "out": 2,
                    "in": 4,
                },
                pipeline_ii=1,
                weight_mode="embedded",
                activation_mode="buffer",
            ),
        ],
    )

    architecture = build_hls_architecture(
        [descriptor],
        _config(),
        compile_plan,
    )

    layer = architecture.layers[0]

    assert layer.pipeline_scope == "input_base"
    assert layer.reduction_iterations == 32
    assert layer.pipeline_overlap == 1
    assert layer.explicit_lanes == 8
    assert layer.effective_lanes == 8

    assert layer.memory["input_banks"] == 4
    assert layer.memory["output_banks"] == 2
    assert layer.memory["weight_banks"] == 8


def test_pool_architecture_models_window_overlap() -> None:
    descriptor = LayerDescriptor(
        node_name="pool0",
        op_type="MaxPool",
        inputs=["input"],
        outputs=["output"],
        input_shapes=[
            (1, 8, 26, 26),
        ],
        output_shapes=[
            (1, 8, 13, 13),
        ],
        attrs={
            "kernel_shape": [2, 2],
            "strides": [2, 2],
        },
    )

    compile_plan = CompilePlan(
        layer_plans=[
            LayerPlan(
                node_name="pool0",
                op_type="MaxPool",
                pipeline_ii=1,
            ),
        ],
    )

    architecture = build_hls_architecture(
        [descriptor],
        _config(),
        compile_plan,
    )

    layer = architecture.layers[0]

    assert layer.pipeline_scope == "output_element"
    assert layer.reduction_iterations == 4
    assert layer.pipeline_overlap == 4
    assert layer.explicit_lanes == 1
    assert layer.effective_lanes == 4


def test_architecture_serializes_to_dictionary() -> None:
    descriptor = LayerDescriptor(
        node_name="relu0",
        op_type="Relu",
        inputs=["input"],
        outputs=["output"],
        input_shapes=[
            (1, 16),
        ],
        output_shapes=[
            (1, 16),
        ],
    )

    architecture = build_hls_architecture(
        [descriptor],
        _config(),
    )

    payload = architecture.to_dict()

    assert payload["policy"] == "Balanced"
    assert payload["execution_mode"] == "sequential"
    assert payload["clock_mhz"] == 200.0
    assert payload["layers"][0]["name"] == "relu0"
    assert payload["layers"][0]["op_type"] == "Relu"