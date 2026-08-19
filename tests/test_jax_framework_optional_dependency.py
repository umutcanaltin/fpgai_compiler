from pathlib import Path


def test_framework_mlir_extra_declares_jax():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "framework-mlir = [" in text
    assert "jax>=0.4.30" in text


def test_jax_exporter_dependency_error_has_exact_install_command():
    text = Path("scripts/export_jax_stablehlo.py").read_text(encoding="utf-8")
    assert "python -m pip install -e '.[framework-mlir]'" in text


def test_jax_exporter_uses_portable_jax_export_api():
    text = Path("scripts/export_jax_stablehlo.py").read_text(encoding="utf-8")
    assert "jax.export.export" in text
    assert ".mlir_module()" in text
    assert ".compiler_ir(" not in text
