"""Top-level FPGAI package exports.

Keep heavy compiler imports lazy so utility modules such as reports/runtime can be
imported in lightweight environments without ONNX/Vitis dependencies installed.
"""

__all__ = ["Compiler", "CompileResult"]


def __getattr__(name):
    if name == "Compiler":
        from .engine.compiler import Compiler

        return Compiler
    if name == "CompileResult":
        from .engine.result import CompileResult

        return CompileResult
    raise AttributeError(name)
