from fpgai.ir.graph import Graph


def test_tensor_quantization_metadata_is_ir_visible():
    graph = Graph("quantized")
    graph.add_tensor("x", (1, 4), quantization={"bits": 8, "scale": 0.1, "zero_point": 0})
    assert graph.tensors["x"].quantization["bits"] == 8
    graph.set_tensor_quantization("x", {"bits": 4, "scale": 0.2, "zero_point": 0})
    assert graph.tensors["x"].quantization["bits"] == 4
