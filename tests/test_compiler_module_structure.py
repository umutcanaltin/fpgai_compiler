"""Regression tests for the compiler module decomposition."""

from pathlib import Path


def test_compiler_helpers_are_split_into_focused_modules() -> None:
    expected_modules = {
        "compiler_reports.py",
        "inference_reference.py",
        "training_contracts.py",
        "vivado_pipeline.py",
        "memory_semantics.py",
    }

    engine_dir = Path("fpgai/engine")
    existing = {path.name for path in engine_dir.glob("*.py")}

    assert expected_modules <= existing


def test_compiler_module_is_smaller_after_helper_extraction() -> None:
    compiler_lines = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8").splitlines()
    assert len(compiler_lines) < 4500


def test_private_compatibility_imports_remain_available() -> None:
    from fpgai.engine.compiler import (
        _emit_inference_reference_artifacts,
        _resolve_codegen_readability,
        _resolve_runtime_sequence,
        _resolve_training_optimizer_loss_contract,
        _resolved_toolchain_summary,
    )

    assert callable(_emit_inference_reference_artifacts)
    assert callable(_resolve_codegen_readability)
    assert callable(_resolve_runtime_sequence)
    assert callable(_resolve_training_optimizer_loss_contract)
    assert callable(_resolved_toolchain_summary)

def test_compiler_constructor_accepts_config() -> None:
    import inspect

    from fpgai.engine.compiler import Compiler

    signature = inspect.signature(Compiler)
    assert "cfg" in signature.parameters



def test_memory_semantics_are_owned_by_a_focused_module() -> None:
    from fpgai.engine.compiler import Compiler
    from fpgai.engine.memory_semantics import MemorySemanticsMixin

    assert issubclass(Compiler, MemorySemanticsMixin)
    assert callable(Compiler._resolve_weight_movement_semantics)
    assert callable(Compiler._annotate_memory_movement_semantics)


def test_training_testbench_postprocess_staticmethod_signature() -> None:
    import inspect

    from fpgai.engine.compiler import Compiler

    signature = inspect.signature(
        Compiler._postprocess_training_tb_cpp_for_requested_export_capture
    )
    assert list(signature.parameters) == ["tb_cpp", "raw"]
