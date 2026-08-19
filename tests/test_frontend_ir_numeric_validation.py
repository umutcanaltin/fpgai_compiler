from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from fpgai.benchmark.graph_reference import execute_graph_reference_trace
from fpgai.engine.compiler import Compiler
from fpgai.ir import Graph


def _dense_relu_graph() -> Graph:
    g = Graph("dense_relu")
    g.add_tensor("x", (2,), "float32")
    g.add_tensor("w", (2, 2), "float32")
    g.add_tensor("b", (2,), "float32")
    g.add_tensor("dense_y", (2,), "float32")
    g.add_tensor("y", (2,), "float32")
    g.inputs = ["x"]
    g.outputs = ["y"]
    g.constants["w"] = np.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
    g.constants["b"] = np.asarray([0.25, -0.5], dtype=np.float32)
    g.add_op("Dense", ["x", "w", "b"], ["dense_y"], name="dense0")
    g.add_op("Relu", ["dense_y"], ["y"], name="relu0")
    return g


def test_functional_ir_reference_trace_exposes_layer_and_model_values() -> None:
    graph = _dense_relu_graph()
    trace = execute_graph_reference_trace(graph, {"x": np.asarray([1.0, -2.0], dtype=np.float32)})
    np.testing.assert_allclose(trace["dense_y"], np.asarray([-2.75, -2.5], dtype=np.float32))
    np.testing.assert_allclose(trace["y"], np.asarray([0.0, 0.0], dtype=np.float32))


def test_numeric_enforce_requires_frontend_ir_when_compare_ir_enabled(tmp_path) -> None:
    report = tmp_path / "reports" / "numeric_validation.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        "status": "passed",
        "frontend_to_fpgai_ir": {"status": "runtime_unavailable", "passed": None},
    }))
    compiler = object.__new__(Compiler)
    compiler.cfg = SimpleNamespace(raw={
        "validation": {"numeric": {
            "enabled": True,
            "policy": "enforce",
            "reference": {"compare_ir": True},
        }}
    })
    with pytest.raises(RuntimeError, match="NUMVAL003"):
        compiler._enforce_numeric_validation_policy(
            out_dir=tmp_path,
            numeric_validation_artifacts={"numeric_validation_json": report},
        )
