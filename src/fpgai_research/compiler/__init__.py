"""Stable research compiler API: Model IR -> legal architecture -> emitted HLS.

Vendor programs are never launched during compilation. Full codegen and
numerical behavior remain in the existing model_ir/architecture/codegen
packages; this is the shared orchestration boundary for contributors.
"""
from .pipeline import CompilationArtifact, compile_architecture
from .neural_chain import (
    NeuralChainCompilation, compile_neural_chain, compile_onnx_neural_chain,
)
__all__ = ["CompilationArtifact", "compile_architecture",
           "NeuralChainCompilation", "compile_neural_chain",
           "compile_onnx_neural_chain"]
