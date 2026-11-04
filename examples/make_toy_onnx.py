"""Create a small MatMul->Relu ONNX graph for research CLI demonstration."""
import argparse


def main():
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    p = argparse.ArgumentParser()
    p.add_argument("output", nargs="?", default="toy.onnx")
    args = p.parse_args()
    graph = helper.make_graph([
        helper.make_node("MatMul", ["x", "w"], ["z"], name="mm"),
        helper.make_node("Relu", ["z"], ["y"], name="relu")],
        "toy", [helper.make_tensor_value_info("x",TensorProto.FLOAT,[2,3])],
        [helper.make_tensor_value_info("y",TensorProto.FLOAT,[2,2])],
        initializer=[numpy_helper.from_array(np.asarray([[1,2],[3,4],[5,6]],dtype=np.float32),"w")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model,args.output)


if __name__ == "__main__":
    main()
