from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fpgai.validation.mixed_external_hls import execute_mixed_graph_reference


@dataclass
class Tensor:
    shape: tuple[int, ...]


@dataclass
class Op:
    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict = field(default_factory=dict)


@dataclass
class Graph:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    tensors: dict[str, Tensor]
    ops: tuple[Op, ...]
    constants: dict = field(default_factory=dict)


class Result:
    def __init__(self, output):
        self.outputs = (output,)


class ExternalContext:
    def reference_for(self, operator_id):
        assert operator_id == "community.operator.scale_bias"
        # Deliberately flatten, matching the maintained external package callback.
        return lambda ctx: Result(np.asarray(ctx.inputs[0], dtype=np.float32).reshape(-1))


def test_external_reference_output_is_restored_to_declared_tensor_shape_before_add():
    graph = Graph(
        inputs=("input",),
        outputs=("sum",),
        tensors={
            "input": Tensor((1, 4)),
            "relu": Tensor((1, 4)),
            "scaled": Tensor((1, 4)),
            "sum": Tensor((1, 4)),
        },
        ops=(
            Op("relu_0", "Relu", ("input",), ("relu",)),
            Op(
                "scale_bias_0",
                "ScaleBias",
                ("input",),
                ("scaled",),
                {"_fpgai_external_operator": {"operator_id": "community.operator.scale_bias"}},
            ),
            Op("add_0", "Add", ("relu", "scaled"), ("sum",)),
        ),
    )

    actual = execute_mixed_graph_reference(
        graph,
        ExternalContext(),
        np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(actual, [-1.0, 0.0, 2.0, 4.0])
