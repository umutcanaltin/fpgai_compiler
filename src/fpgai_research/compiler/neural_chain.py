"""R2z end-to-end CNN -> Dense source compiler, no new kernel generators.

Reuses the canonical semantic-preserving compile_architecture path and
Vitis HLS C++ emitter for Conv2D, MaxPool, Relu, Flatten and Gemm/MatMul.
All layers of the supported sequential graph are materialized; no
fictitious in-place/FIFO/BRAM optimisation or new native tool calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from fpgai_research.compiler.pipeline import CompilationArtifact, compile_architecture
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.neural_chain import (
    NeuralChainIR, resolve_neural_chain, neural_chain_receipt,
)


@dataclass(frozen=True)
class NeuralChainCompilation:
    ir: NeuralChainIR
    compilation: CompilationArtifact
    evidence: dict

    @property
    def emission(self):
        return self.compilation.emission


def compile_onnx_neural_chain(
    path: str,
    *, default=None, overrides=None,
    max_operators: int=24,
) -> NeuralChainCompilation:
    """Import a real statically shaped ONNX CNN/Dense model, no vendor calls.

    The existing strict ONNX frontend, architecture resolver and canonical
    HLS emitter remain the single implementation path. Unsupported ONNX
    operators/attributes fail closed; there is no silent fallback.
    """
    from fpgai_research.architecture.resolver import resolve
    from fpgai_research.frontend.onnx import load_onnx
    from fpgai_research.model_ir.contracts import ArchitectureRequest
    base = ArchitectureRequest() if default is None else default
    graph = load_onnx(path)
    architecture = resolve(graph,base,overrides)
    return compile_neural_chain(
        architecture,max_operators=max_operators,
    )


def compile_neural_chain(
    architecture: ArchitecturalModel,
    *, max_operators: int=24,
    max_model_elements: int=250_000,
    max_constant_elements: int=150_000,
) -> NeuralChainCompilation:
    ir=resolve_neural_chain(
        architecture,max_operators=max_operators,
        max_model_elements=max_model_elements,
        max_constant_elements=max_constant_elements,
    )
    compiled=compile_architecture(architecture)
    emission=compiled.emission
    model=architecture.functional
    actual=[(x["operator"],x["kind"])
            for x in emission.manifest.get("operators",[])]
    expected=[(x.name,x.kind) for x in model.operators]
    if actual!=expected:
        raise ValueError("R2z canonical HLS generator dropped/reordered NN stages")
    if emission.manifest.get("functional_sha256")!=compiled.refinement_receipt[
            "functional_sha256"]:
        raise ValueError("R2z HLS source changed Functional Model identity")
    if emission.manifest.get("loop_codegen_version")!="static-conv-trip-v3":
        raise ValueError("R2z requires current bounded-loop Conv2D source lowering")
    if sha256(emission.cpp.encode()).hexdigest()!=emission.manifest.get(
            "cpp_sha256"):
        raise ValueError("R2z canonical HLS C++ digest mismatch")
    receipt=neural_chain_receipt(ir)
    return NeuralChainCompilation(
        ir=ir,compilation=compiled,
        evidence={
            **receipt,
            "canonical_hls_source_sha256":emission.manifest["cpp_sha256"],
            "original_existing_conv_and_dense_emitters_reused":True,
            "per_operator_lowering_preserved":True,
            "source_compilation":True,
            "actual_GCC_execution":False,
            "native_HLS_CSim_or_CSynth":False,
            "physical_FPGA_resources":None,
            "evidence_level":"E1-generated-source-only-no-vendor",
        },
    )
