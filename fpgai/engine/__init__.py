"""FPGAI engine package exports with lazy compiler imports."""

__all__ = ["Compiler", "CompileResult"]


def __getattr__(name):
    if name == "Compiler":
        from .compiler import Compiler

        return Compiler
    if name == "CompileResult":
        from .result import CompileResult

        return CompileResult
    raise AttributeError(name)
