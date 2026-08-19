from pathlib import Path

from fpgai.engine.result import CompileResult
from fpgai.ir import Graph


def test_compile_result_summary_uses_canonical_graph_constants(tmp_path: Path):
    g = Graph("mlir_graph")
    g.add_tensor("x", (1, 4), "float32")
    g.add_tensor("w", (4, 4), "float32")
    g.add_tensor("y", (1, 4), "float32")
    g.inputs = ["x"]
    g.outputs = ["y"]

    import numpy as np
    g.constants["w"] = np.eye(4, dtype=np.float32)
    g.add_op("MatMul", ["x", "w"], ["y"], name="mm")

    # Canonical source-driven graphs do not require the historic ONNX .params alias.
    assert not hasattr(g, "params")

    result = CompileResult(out_dir=tmp_path, graph=g)
    text = result.summary()
    assert "Params               : 1" in text
    assert "Ops                  : 1" in text
