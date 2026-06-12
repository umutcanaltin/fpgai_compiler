from __future__ import annotations

import numpy as np

from fpgai.backends.hls.emit.layers_conv import emit_conv_h
from fpgai.backends.hls.emit.layers_dense import emit_dense_h
from fpgai.backends.hls.emit.top_cpp import emit_top_cpp
from fpgai.engine.layerwise_precision import (
    resolve_layerwise_precision,
)
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


def _graph() -> Graph:
    graph = Graph("per_layer_codegen")
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor("input", (1, 4))
    graph.add_tensor("hidden", (1, 3))
    graph.add_tensor("output", (1, 2))

    graph.constants["w0"] = np.ones(
        (3, 4),
        dtype=np.float32,
    )
    graph.constants["b0"] = np.zeros(
        (3,),
        dtype=np.float32,
    )
    graph.constants["w1"] = np.ones(
        (2, 3),
        dtype=np.float32,
    )
    graph.constants["b1"] = np.zeros(
        (2,),
        dtype=np.float32,
    )

    for name, value in graph.constants.items():
        graph.add_tensor(name, value.shape)

    graph.add_op(
        "Dense",
        ["input", "w0", "b0"],
        ["hidden"],
        name="dense0",
        attrs={"in_features": 4, "out_features": 3},
    )
    graph.add_op(
        "Dense",
        ["hidden", "w1", "b1"],
        ["output"],
        name="dense1",
        attrs={"in_features": 3, "out_features": 2},
    )

    config = {
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
    }
    resolve_layerwise_precision(graph, config)
    return graph


def test_dense_kernel_has_per_layer_template_controls() -> None:
    header = emit_dense_h()

    assert "int PIPELINE_II = FPGAI_PIPELINE_II" in header
    assert "int IN_UNROLL = FPGAI_DENSE_IN_UNROLL" in header
    assert "int OUT_UNROLL = FPGAI_DENSE_OUT_UNROLL" in header
    assert "II=PIPELINE_II" in header
    assert "factor=INPUT_PARTITION" in header
    assert "factor=OUTPUT_PARTITION" in header
    assert "factor=WEIGHT_PARTITION" in header


def test_conv_kernel_has_per_layer_template_controls() -> None:
    header = emit_conv_h()

    assert "int PIPELINE_II = FPGAI_PIPELINE_II" in header
    assert "int OC_UNROLL = FPGAI_CONV_OC_UNROLL" in header
    assert "int IC_UNROLL = FPGAI_CONV_IC_UNROLL" in header
    assert "II=PIPELINE_II" in header
    assert "factor=INPUT_PARTITION" in header
    assert "factor=OUTPUT_PARTITION" in header
    assert "factor=WEIGHT_PARTITION" in header


def test_two_dense_layers_emit_distinct_architectures() -> None:
    graph = _graph()
    plan = CompilePlan(
        layer_plans=[
            LayerPlan(
                node_name="dense0",
                op_type="Dense",
                architecture=_architecture(
                    ii=1,
                    input_unroll=2,
                    output_unroll=3,
                    partition=6,
                ),
            ),
            LayerPlan(
                node_name="dense1",
                op_type="Dense",
                architecture=_architecture(
                    ii=2,
                    input_unroll=1,
                    output_unroll=2,
                    partition=2,
                ),
            ),
        ]
    )

    source = emit_top_cpp(
        graph,
        top_name="deeplearn",
        weights_mode="embedded",
        compile_plan=plan,
    )

    assert (
        "op0_acc_t, 1, 2, 3, 2, 3, 6"
        in source
    )
    assert (
        "op1_acc_t, 2, 1, 2, 1, 2, 2"
        in source
    )
