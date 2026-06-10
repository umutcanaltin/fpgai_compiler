from __future__ import annotations

import numpy as np

from fpgai.engine.training_graph_utils import (
    as_chw,
    flat_size,
    get_tensor_shape,
    infer_conv_output_shape,
    infer_pool_output_shape,
    remap_dense_weights_chw_to_hwc,
    resolve_batchnorm_arrays,
    resolve_conv_arrays,
    resolve_dense_arrays,
    shape_without_batch,
)
from fpgai.ir.graph import Graph


def test_shape_helpers() -> None:
    assert shape_without_batch((1, 3, 8, 8)) == (3, 8, 8)
    assert shape_without_batch((4,)) == (4,)
    assert shape_without_batch(None) == ()

    assert flat_size((3, 8, 8)) == 192
    assert flat_size(()) == 1
    assert as_chw((3, 8, 8)) == (3, 8, 8)

    assert infer_conv_output_shape(
        (3, 8, 8),
        (4, 3, 3, 3),
        stride=1,
        pad=1,
    ) == (4, 8, 8)

    assert infer_pool_output_shape(
        (4, 8, 8),
        kernel_size=2,
        stride=2,
    ) == (4, 4, 4)


def test_resolve_dense_arrays_from_constants() -> None:
    graph = Graph()
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor("input", (1, 4))
    graph.add_tensor("weights", (3, 4))
    graph.add_tensor("bias", (3,))
    graph.add_tensor("output", (1, 3))

    graph.constants["weights"] = np.arange(
        12,
        dtype=np.float32,
    ).reshape(3, 4)

    graph.constants["bias"] = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float32,
    )

    op = graph.add_op(
        "Dense",
        ["input", "weights", "bias"],
        ["output"],
        name="dense0",
        attrs={
            "in_features": 4,
            "out_features": 3,
        },
    )

    weights, bias, input_features, output_features = (
        resolve_dense_arrays(graph, op)
    )

    assert weights.shape == (3, 4)
    assert bias.shape == (3,)
    assert input_features == 4
    assert output_features == 3

    np.testing.assert_array_equal(
        weights,
        graph.constants["weights"],
    )
    np.testing.assert_allclose(
        bias,
        graph.constants["bias"],
    )


def test_dense_resolution_remaps_chw_to_hwc() -> None:
    graph = Graph()
    graph.inputs = ["feature_map"]
    graph.outputs = ["output"]

    graph.add_tensor("feature_map", (1, 2, 1, 2))
    graph.add_tensor("flat", (1, 4))
    graph.add_tensor("weights", (1, 4))
    graph.add_tensor("output", (1, 1))

    graph.constants["weights"] = np.array(
        [[10.0, 20.0, 30.0, 40.0]],
        dtype=np.float32,
    )

    graph.add_op(
        "Flatten",
        ["feature_map"],
        ["flat"],
        name="flatten0",
    )

    dense = graph.add_op(
        "Dense",
        ["flat", "weights"],
        ["output"],
        name="dense0",
        attrs={
            "in_features": 4,
            "out_features": 1,
        },
    )

    weights, bias, _, _ = resolve_dense_arrays(
        graph,
        dense,
    )

    np.testing.assert_array_equal(
        weights,
        np.array(
            [[10.0, 30.0, 20.0, 40.0]],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        bias,
        np.zeros((1,), dtype=np.float32),
    )


def test_remapping_ignores_non_spatial_input() -> None:
    weights = np.arange(
        8,
        dtype=np.float32,
    ).reshape(2, 4)

    result = remap_dense_weights_chw_to_hwc(
        weights,
        (4,),
    )

    assert result is weights


def test_resolve_conv_arrays_with_default_bias() -> None:
    graph = Graph()
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor("input", (1, 1, 5, 5))
    graph.add_tensor("weights", (2, 1, 3, 3))
    graph.add_tensor("output", (1, 2, 3, 3))

    graph.constants["weights"] = np.arange(
        18,
        dtype=np.float32,
    ).reshape(2, 1, 3, 3)

    op = graph.add_op(
        "Conv",
        ["input", "weights"],
        ["output"],
        name="conv0",
        attrs={
            "kernel_shape": [3, 3],
        },
    )

    weights, bias, weight_shape = resolve_conv_arrays(
        graph,
        op,
    )

    assert weight_shape == (2, 1, 3, 3)
    assert weights.shape == weight_shape
    assert bias.shape == (2,)

    np.testing.assert_array_equal(
        weights,
        graph.constants["weights"],
    )

    np.testing.assert_array_equal(
        bias,
        np.zeros((2,), dtype=np.float32),
    )


def test_resolve_batchnorm_arrays_uses_defaults() -> None:
    graph = Graph()
    graph.inputs = ["input"]
    graph.outputs = ["output"]

    graph.add_tensor("input", (1, 3, 4, 4))
    graph.add_tensor("output", (1, 3, 4, 4))

    op = graph.add_op(
        "BatchNormalization",
        ["input"],
        ["output"],
        name="batchnorm0",
    )

    gamma, beta, mean, variance = (
        resolve_batchnorm_arrays(
            graph,
            op,
            channels=3,
        )
    )

    np.testing.assert_array_equal(
        gamma,
        np.ones((3,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        beta,
        np.zeros((3,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        mean,
        np.zeros((3,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        variance,
        np.ones((3,), dtype=np.float32),
    )

    assert get_tensor_shape(
        graph,
        "input",
    ) == (3, 4, 4)