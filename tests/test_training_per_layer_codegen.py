from __future__ import annotations

import numpy as np

from fpgai.backends.hls.emit.layers_conv import emit_conv_h
from fpgai.backends.hls.emit.layers_dense import emit_dense_h
from fpgai.backends.hls.emit.top_train_cpp import emit_top_train_cpp
from fpgai.engine.layerwise_precision import resolve_layerwise_precision
from fpgai.engine.models import (
    ArchitecturePlan,
    BufferingPlan,
    CompilePlan,
    LayerMemoryPlan,
    LayerPlan,
    ParallelismPlan,
    PartitionPlan,
    PipelinePlan,
    PrecisionPlan,
    TilingPlan,
)
from fpgai.ir import Graph


def _precision() -> PrecisionPlan:
    return PrecisionPlan(
        mode="fixed",
        activation_bits=16,
        weight_bits=16,
        bias_bits=24,
        accumulator_bits=24,
    )


def _architecture(
    *,
    ii: int,
    input_unroll: int,
    output_unroll: int,
    partition: int,
) -> ArchitecturePlan:
    return ArchitecturePlan(
        precision=_precision(),
        pipeline=PipelinePlan(ii=ii),
        parallelism=ParallelismPlan(
            pe=output_unroll,
            simd=input_unroll,
            unroll={
                "out": output_unroll,
                "in": input_unroll,
            },
        ),
        partitioning=PartitionPlan(
            factor=partition,
            mode="cyclic",
            targets={
                "input": input_unroll,
                "output": output_unroll,
                "weight": partition,
            },
        ),
        tiling=TilingPlan(),
        buffering=BufferingPlan(),
        memory=LayerMemoryPlan(),
    )


def _dense_graph() -> Graph:
    graph = Graph("training_per_layer_codegen")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor("input", (1, 4))
    graph.add_tensor("output", (1, 3))

    graph.constants["w0"] = np.ones((3, 4), dtype=np.float32)
    graph.constants["b0"] = np.zeros((3,), dtype=np.float32)

    for name, value in graph.constants.items():
        graph.add_tensor(name, value.shape)

    graph.add_op(
        "Dense",
        ["input", "w0", "b0"],
        ["output"],
        name="dense0",
        attrs={"in_features": 4, "out_features": 3},
    )

    resolve_layerwise_precision(
        graph,
        {
            "numerics": {
                "defaults": {
                    "activation": {
                        "type": "ap_fixed",
                        "total_bits": 16,
                        "int_bits": 6,
                    },
                    "weight": {
                        "type": "ap_fixed",
                        "total_bits": 16,
                        "int_bits": 6,
                    },
                    "bias": {
                        "type": "ap_fixed",
                        "total_bits": 24,
                        "int_bits": 10,
                    },
                    "accum": {
                        "type": "ap_fixed",
                        "total_bits": 24,
                        "int_bits": 10,
                    },
                }
            }
        },
    )
    return graph


def test_dense_training_kernels_have_per_layer_controls() -> None:
    header = emit_dense_h()

    assert "void dense_weight_grad_typed" in header
    assert "void dense_backward_input_typed" in header
    assert "void sgd_update_wgt_typed" in header
    assert "void sgd_update_bias_typed" in header
    assert "int PIPELINE_II = FPGAI_PIPELINE_II" in header
    assert "factor=INPUT_PARTITION" in header
    assert "factor=OUTPUT_PARTITION" in header
    assert "factor=WEIGHT_PARTITION" in header
    assert "factor=PARTITION" in header


def test_conv_training_kernels_have_per_layer_controls() -> None:
    header = emit_conv_h()

    assert "void conv2d_weight_grad_typed" in header
    assert "void conv2d_backward_input_typed" in header
    assert "int PIPELINE_II = FPGAI_PIPELINE_II" in header
    assert "int IC_UNROLL = FPGAI_CONV_IC_UNROLL" in header
    assert "int OC_UNROLL = FPGAI_CONV_OC_UNROLL" in header
    assert "factor=INPUT_PARTITION" in header
    assert "factor=OUTPUT_PARTITION" in header
    assert "factor=WEIGHT_PARTITION" in header


def test_dense_training_path_uses_layer_specific_backward_and_update() -> None:
    graph = _dense_graph()
    plan = CompilePlan(
        layer_plans=[
            LayerPlan(
                node_name="dense0",
                op_type="Dense",
                architecture=_architecture(
                    ii=2,
                    input_unroll=2,
                    output_unroll=3,
                    partition=6,
                ),
            )
        ]
    )

    source = emit_top_train_cpp(
        graph=graph,
        top_name="deeplearn_train",
        weights_mode="embedded",
        training_cfg={
            "loss": {"type": "mse"},
            "optimizer": {"learning_rate": 0.01},
        },
        compile_plan=plan,
    )

    assert (
        "dense_weight_grad_typed<4, 3, act_t, grad_act_t, "
        "grad_wgt_t, acc_t, 2, 2, 3, 2, 3, 6>"
        in source
    )
    assert (
        "dense_bias_grad_typed<3, grad_act_t, grad_bias_t, 2, 3>"
        in source
    )
    assert (
        "dense_backward_input_typed<4, 3, grad_act_t, wgt_t, "
        "grad_act_t, acc_t, 2, 2, 3, 2, 3, 6>"
        in source
    )
    assert (
        "sgd_update_wgt_typed<12, wgt_t, grad_wgt_t, upd_t, "
        "acc_t, 2, 6>"
        in source
    )
    assert (
        "sgd_update_bias_typed<3, bias_t, grad_bias_t, upd_t, "
        "acc_t, 2, 3>"
        in source
    )
