"""Standalone synthetic two-layer dense ONNX example (no convolution).

Use the general FPGAI compiler CLI rather than the strict NCHW CNN-chain
example entrypoint. No downloads, private weights or vendor tools.

    python examples/make_mlp_onnx.py build/gallery/tiny_mlp.onnx
    fpgai-research build/gallery/tiny_mlp.onnx -o build/gallery/tiny_mlp_hls
"""
from __future__ import annotations

import argparse
from pathlib import Path


def make_model():
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def values(shape, seed):
        n = int(np.prod(shape))
        return ((((np.arange(n) * 3 + seed) % 17) - 8).astype(np.float32)
                * np.float32(0.0078125)).reshape(shape)

    initializer = [
        numpy_helper.from_array(values((4, 6), 1), name="w0"),
        numpy_helper.from_array(values((6,), 2), name="b0"),
        numpy_helper.from_array(values((6, 3), 3), name="w1"),
        numpy_helper.from_array(values((3,), 4), name="b1"),
    ]
    nodes = [
        helper.make_node("Gemm", ["x", "w0", "b0"], ["hidden"], name="dense0"),
        helper.make_node("Relu", ["hidden"], ["activated"], name="relu0"),
        helper.make_node("Gemm", ["activated", "w1", "b1"],
                         ["logits"], name="dense1"),
        helper.make_node("Softmax", ["logits"], ["out"],
                         name="softmax", axis=1),
    ]
    graph = helper.make_graph(
        nodes, "fpgai_synthetic_two_layer_mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])],
        initializer=initializer,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="fpgai-research-synthetic-examples",
    )
    onnx.checker.check_model(model)
    return onnx.shape_inference.infer_shapes(model)


def save_model(output: str | Path) -> Path:
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite synthetic MLP model: {path}")
    data = make_model().SerializeToString()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="New ONNX path, never overwritten")
    args = parser.parse_args(argv)
    print(f"Wrote untrained two-layer MLP: {save_model(args.output)}")


if __name__ == "__main__":
    main()
