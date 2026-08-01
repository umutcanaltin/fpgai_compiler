from __future__ import annotations
import argparse
from pathlib import Path


def write_model(path: Path) -> Path:
    import onnx
    from onnx import TensorProto, helper
    nodes = [
        helper.make_node("Relu", ["input"], ["relu_out"], name="relu_0"),
        helper.make_node("ScaleBias", ["relu_out"], ["scaled"], name="scale_bias_0", domain="community.fpgai", scale=2.0, bias=1.0),
        helper.make_node("Sigmoid", ["scaled"], ["output"], name="sigmoid_0"),
    ]
    graph = helper.make_graph(
        nodes, "mixed_external_relu_scale_bias_sigmoid",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("community.fpgai", 1)])
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/mixed_external_relu_scale_bias_sigmoid.onnx")
    args = parser.parse_args()
    print(write_model(Path(args.out)))

if __name__ == "__main__":
    main()
