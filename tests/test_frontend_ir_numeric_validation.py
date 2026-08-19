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


def test_mlir_frontend_reference_bundle_validates_model_and_intermediate(tmp_path):
    import json
    import numpy as np
    from fpgai.ir.graph import Graph
    from fpgai.validation.frontend_ir import validate_frontend_to_fpgai_ir

    model = tmp_path / "model.mlir"
    model.write_text("module {}\n", encoding="utf-8")
    bundle_dir = tmp_path / "reference"
    bundle_dir.mkdir()
    x = np.asarray([[-1.0, 2.0, -3.0, 4.0]], dtype=np.float32)
    a = np.maximum(x, 0.0)
    y = np.maximum(a, 0.0)
    np.save(bundle_dir / "x.npy", x)
    np.save(bundle_dir / "a.npy", a)
    np.save(bundle_dir / "y.npy", y)
    manifest = {
        "schema": "fpgai.frontend-reference/v1",
        "source_framework": "jax",
        "inputs": {"jax_arg0": {"path": "x.npy", "fpgai_tensor": "x"}},
        "outputs": {"jax_result": {"path": "y.npy", "fpgai_tensor": "y"}},
        "intermediates": {"jax_relu0": {"path": "a.npy", "fpgai_tensor": "a"}},
    }
    manifest_path = bundle_dir / "reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    graph = Graph("jax_demo")
    for name in ("x", "a", "y"):
        graph.add_tensor(name, (1, 4))
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_op("Relu", ["x"], ["a"], name="relu0")
    graph.add_op("Relu", ["a"], ["y"], name="relu1")

    raw = {"validation": {"numeric": {
        "enabled": True,
        "levels": ["model", "layer", "intermediate"],
        "reference": {"source": "framework", "compare_ir": True, "bundle": str(manifest_path)},
    }}}
    result = validate_frontend_to_fpgai_ir(
        graph=graph, model_path=model, model_format="stablehlo", raw_config=raw, out_dir=tmp_path / "out"
    )
    assert result["passed"] is True
    assert result["source_framework"] == "jax"
    assert result["outputs"]["y"]["passed"] is True
    assert result["intermediates"]["a"]["passed"] is True


def test_reference_bundle_enforces_requested_layer_coverage(tmp_path):
    import json
    import numpy as np
    from fpgai.ir.graph import Graph
    from fpgai.validation.frontend_ir import validate_frontend_to_fpgai_ir

    model = tmp_path / "model.mlir"
    model.write_text("module {}\n")
    x = np.asarray([[-1.0, 2.0]], dtype=np.float32)
    y = np.maximum(x, 0.0)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", y)
    manifest = {
        "schema": "fpgai.frontend-reference/v1",
        "inputs": {"x": {"path": "x.npy", "fpgai_tensor": "x"}},
        "outputs": {"y": {"path": "y.npy", "fpgai_tensor": "y"}},
    }
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(manifest))
    graph = Graph("coverage")
    for name in ("x", "a", "y"):
        graph.add_tensor(name, (1, 2))
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_op("Relu", ["x"], ["a"], name="relu0")
    graph.add_op("Identity", ["a"], ["y"], name="out")
    raw = {"validation": {"numeric": {"enabled": True, "levels": ["model", "layer"], "reference": {"bundle": str(bundle)}}}}
    result = validate_frontend_to_fpgai_ir(graph=graph, model_path=model, model_format="stablehlo", raw_config=raw, out_dir=tmp_path / "out")
    assert result["passed"] is False
    assert result["status"] == "insufficient_reference_coverage"
    assert result["missing_intermediates"] == ["a"]


def test_functional_reference_executes_stablehlo_broadcast_in_dim_semantics():
    graph = Graph("broadcast")
    graph.add_tensor("x", (3,))
    graph.add_tensor("y", (1, 3))
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.add_op("Broadcast", ["x"], ["y"], name="broadcast0", attrs={"broadcast_dimensions": [1]})
    trace = execute_graph_reference_trace(graph, {"x": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)})
    np.testing.assert_allclose(trace["y"], np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
