from pathlib import Path

import pytest

from fpgai.frontend import FrontendSpec, frontend_registry, import_model_source, register_frontend, source_framework_route
from fpgai.ir import Graph


def test_builtin_frontends_are_source_driven_and_model_agnostic():
    names = frontend_registry()
    assert {"onnx", "mlir", "stablehlo"}.issubset(names)
    assert "yolo" not in names
    assert "transformer" not in names
    assert "attention" not in names


def test_external_frontend_can_register_without_core_model_handler(tmp_path: Path):
    model = tmp_path / "model.foo"
    model.write_text("external", encoding="utf-8")

    def importer(source, **kwargs):
        g = Graph("external")
        g.add_tensor("x", (1, 4))
        g.inputs = ["x"]
        g.outputs = ["x"]
        return g

    register_frontend(FrontendSpec("testfoo", importer, (".foo",), provider="test"), replace=True)
    graph = import_model_source(model, format_hint="testfoo", source_framework="custom")
    assert graph.metadata["source"]["format"] == "testfoo"
    assert graph.metadata["source"]["framework"] == "custom"
    assert graph.metadata["source"]["frontend_provider"] == "test"


def test_unknown_source_requires_format_hint(tmp_path: Path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="set model.format"):
        import_model_source(path)


def test_framework_routes_are_explicit_about_interchange_and_legalization():
    jax = source_framework_route("jax", "stablehlo")
    assert jax["selected_path_accepted"] is True
    tf = source_framework_route("tensorflow", "stablehlo")
    assert tf["selected_path_accepted"] is True
    assert tf["legalization_applied_upstream"] is True
    torch = source_framework_route("pytorch", "onnx")
    assert torch["selected_path_accepted"] is True
    onnx = source_framework_route("onnx", "onnx")
    assert onnx["accepted_by_fpgai"] is True
