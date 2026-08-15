from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build_model(path: Path) -> Path:
    split = helper.make_node(
        "SplitScale",
        ["input"],
        ["identity", "scaled"],
        name="split_scale_0",
        domain="community.fpgai",
        scale=2.0,
    )
    add = helper.make_node("Add", ["identity", "scaled"], ["summed"], name="add_0")
    relu = helper.make_node("Relu", ["summed"], ["output"], name="relu_0")
    graph = helper.make_graph(
        [split, add, relu],
        "mixed_external_multi_output",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        value_info=[
            helper.make_tensor_value_info("identity", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("scaled", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("summed", TensorProto.FLOAT, [1, 4]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="fpgai-maintained-example",
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("community.fpgai", 1)],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    print(build_model(Path("models/mixed_external_multi_output.onnx")))
