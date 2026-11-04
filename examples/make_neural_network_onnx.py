"""Generate three reproducible, synthetic convolution/dense ONNX examples.

These models are deliberately small and untrained. They exercise the same
supported static neural-chain topologies as the compiler's existing source
validation. No private weights, datasets or native FPGA tools are required.

Usage:
    python examples/make_neural_network_onnx.py cnn_pool_dense build/gallery/pool.onnx
"""
from __future__ import annotations

import argparse
from math import prod
from pathlib import Path

NETWORKS = ("cnn_pool_dense", "cnn_strided_dense", "cnn_two_conv_softmax")
EXPECTED_INPUTS = {
    "cnn_pool_dense": (1, 1, 6, 6),
    "cnn_strided_dense": (2, 1, 7, 8),
    "cnn_two_conv_softmax": (1, 2, 5, 7),
}
EXPECTED_OUTPUTS = {
    "cnn_pool_dense": (1, 2),
    "cnn_strided_dense": (2, 3),
    "cnn_two_conv_softmax": (1, 3),
}


def _weight(name: str, shape: tuple[int, ...], seed: int):
    import numpy as np

    n = prod(shape)
    values = (((np.arange(n, dtype=np.int32) * 7 + seed) % 13) - 6)
    return name, (values.astype(np.float32) * np.float32(0.00625)).reshape(shape)


def _bias(name: str, count: int, seed: int):
    import numpy as np

    values = ((np.arange(count, dtype=np.int32) + seed) % 7) - 3
    return name, values.astype(np.float32) * np.float32(0.0125)


def make_model(network: str):
    """Return a checked ONNX ModelProto, without writing any file.

    Uses the same explicit shapes, weights, operator ordering and attributes
    as three previously validated static neural-network research fixtures,
    but is independently constructed with the public ONNX helper API.
    """
    if network not in NETWORKS:
        raise ValueError(f"Unknown network {network!r}; choose from {NETWORKS}")

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def node(kind, inputs, output, name, **attrs):
        return helper.make_node(kind, inputs, [output], name=name, **attrs)

    if network == "cnn_pool_dense":
        weights = (
            _weight("w0", (2, 1, 3, 3), 1),
            _bias("b0", 2, 2),
            _weight("w1", (3, 2, 1, 1), 3),
            _weight("w2", (27, 5), 4),
            _weight("w3", (2, 5), 5),
            _bias("b3", 2, 6),
        )
        nodes = [
            node("Conv", ["x", "w0", "b0"], "c0", "conv0", pads=[1, 1, 1, 1]),
            node("Relu", ["c0"], "r0", "relu0"),
            node("MaxPool", ["r0"], "p0", "pool0",
                 kernel_shape=[2, 2], strides=[2, 2]),
            node("Conv", ["p0", "w1"], "c1", "conv1"),
            node("Flatten", ["c1"], "flat", "flatten", axis=1),
            node("MatMul", ["flat", "w2"], "d0", "dense0"),
            node("Relu", ["d0"], "r1", "relu1"),
            node("Gemm", ["r1", "w3", "b3"], "out", "dense1", transB=1),
        ]
    elif network == "cnn_strided_dense":
        weights = (
            _weight("w0", (3, 1, 3, 3), 4),
            _weight("w1", (48, 4), 6),
            _bias("b1", 4, 2),
            _weight("w2", (4, 3), 8),
        )
        nodes = [
            node("Conv", ["x", "w0"], "c0", "conv0",
                 strides=[2, 2], pads=[1, 1, 1, 1]),
            node("Relu", ["c0"], "r0", "relu0"),
            node("Flatten", ["r0"], "flat", "flatten", axis=1),
            node("Gemm", ["flat", "w1", "b1"], "d0", "dense0"),
            node("Relu", ["d0"], "r1", "relu1"),
            node("MatMul", ["r1", "w2"], "out", "dense1"),
        ]
    else:
        weights = (
            _weight("w0", (3, 2, 3, 2), 1),
            _bias("b0", 3, 3),
            _weight("w1", (2, 3, 2, 3), 4),
            _weight("w2", (3, 48), 6),
            _bias("b2", 3, 7),
        )
        nodes = [
            node("Conv", ["x", "w0", "b0"], "c0", "conv0",
                 pads=[1, 0, 1, 0]),
            node("Conv", ["c0", "w1"], "c1", "conv1",
                 pads=[0, 1, 0, 1]),
            node("Relu", ["c1"], "r0", "relu0"),
            node("Flatten", ["r0"], "flat", "flatten", axis=1),
            node("Gemm", ["flat", "w2", "b2"], "d0", "dense0", transB=1),
            node("Softmax", ["d0"], "out", "softmax", axis=1),
        ]

    graph = helper.make_graph(
        nodes,
        f"fpgai_example_{network}",
        [helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, list(EXPECTED_INPUTS[network]))],
        [helper.make_tensor_value_info(
            "out", TensorProto.FLOAT, list(EXPECTED_OUTPUTS[network]))],
        initializer=[numpy_helper.from_array(array, name=name)
                     for name, array in weights],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)],
        producer_name="fpgai-research-synthetic-examples",
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    actual = tuple(int(dim.dim_value)
                   for dim in model.graph.output[0].type.tensor_type.shape.dim)
    if actual != EXPECTED_OUTPUTS[network]:
        raise ValueError(
            f"Unexpected ONNX output geometry for {network}: {actual}"
        )
    return model


def save_model(network: str, output: str | Path) -> Path:
    """Write a checked ONNX model, refusing to overwrite existing files."""
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite example ONNX model: {path}")
    model = make_model(network)
    data = model.SerializeToString()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", choices=NETWORKS)
    parser.add_argument("output", help="New ONNX path; will not overwrite")
    args = parser.parse_args(argv)
    path = save_model(args.network, args.output)
    print(
        f"Wrote synthetic {args.network}: "
        f"{EXPECTED_INPUTS[args.network]} -> {EXPECTED_OUTPUTS[args.network]} "
        f"to {path} (no synthesis or vendor tools)"
    )


if __name__ == "__main__":
    main()
