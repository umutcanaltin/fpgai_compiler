from pathlib import Path
import py_compile

ROOT = Path("examples/frontends")


def test_framework_python_examples_exist_and_are_syntax_valid():
    expected = {
        "jax_linear.py",
        "pytorch_linear.py",
        "tensorflow_linear.py",
        "onnx_linear.py",
    }
    assert expected.issubset({p.name for p in ROOT.glob("*.py")})
    for name in expected:
        py_compile.compile(str(ROOT / name), doraise=True)


def test_framework_examples_use_equivalent_parameters_and_supported_routes():
    sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in (
        "jax_linear.py", "pytorch_linear.py", "tensorflow_linear.py", "onnx_linear.py"
    )}
    for source in sources.values():
        assert "(0.10, 0.20, 0.30)" in source
        assert "BIAS = (0.01, 0.02, 0.03)" in source
    assert "jax.export.export" in sources["jax_linear.py"]
    assert "torch.onnx.export" in sources["pytorch_linear.py"]
    assert "tf2onnx.convert.from_function" in sources["tensorflow_linear.py"]
    assert 'helper.make_node("MatMul"' in sources["onnx_linear.py"]


def test_framework_examples_document_claim_boundaries():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "JAX -> StableHLO -> FPGAI IR" in text
    assert "PyTorch -> ONNX -> FPGAI IR" in text
    assert "TensorFlow -> ONNX -> FPGAI IR" in text
    assert "Direct ONNX -> FPGAI IR" in text
    assert "not claimed as a native PyTorch dialect frontend" in text
    assert "does not claim native TensorFlow-dialect import" in text
