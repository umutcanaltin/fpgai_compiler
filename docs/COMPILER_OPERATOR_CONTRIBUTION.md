# Extending FPGAI: operators and neural-network layers

This guide explains how to extend the **research compiler** without adding a fake operator declaration or silently changing semantics. It complements [Contributing](../CONTRIBUTING.md), [Architecture](ARCHITECTURE.md) and [Supported features](SUPPORTED_FEATURES.md).

## Choose the right scope

A named ONNX operator is not equivalent to end-to-end compiler support. A complete contribution may require changes at multiple boundaries:

| Boundary | Location | What must be established |
|---|---|---|
| Import | `src/fpgai_research/frontend/` | Valid graph attributes, tensor ranks, shapes, type constraints |
| Functional semantics | `src/fpgai_research/model_ir/` | What the operator computes, tensor dependencies, numeric interpretation |
| Independent reference | `src/fpgai_research/validation/` | Deterministic expected values and tolerance policy |
| Architecture resolution | `src/fpgai_research/architecture/` | Which precision, tiling, PE, SIMD, serialization and pipeline requests are legal |
| IR lowering | `src/fpgai_research/compiler/`, `model_ir/lowering.py` | Refinement preserves value, shape and request provenance |
| HLS C++ emission | `src/fpgai_research/codegen/hls/` | Generated loop structures and source contracts match the resolved architecture |
| Native hardware (optional) | `src/fpgai_research/toolchains/` | Independently qualified original CSim, synthesis and routing |

The authoritative compiler must own legality and device restrictions; do **not** implement missing operator semantics in a research agent, numerical report, test fixture or copied second backend.

## Minimum implementation sequence

1. **Specify mathematical behavior.** Record input and output tensors, indexing, broadcasting, attributes, supported shapes, numeric accumulation, zero/padding behavior and invalid cases.
2. **Write an independent numeric reference.** Include a tiny reproducible input, a nontrivial shape, a corner case and an invalid case. Define rounding/error bounds before collecting results.
3. **Extend static import and validation.** Keep the functional graph independent of hardware decisions. Reject attributes that are not implemented.
4. **Add supported architecture mapping.** Explicitly declare operator-specific PE/replica/SIMD/tiling semantics, including tail iterations, source dependencies and any capacity restrictions. Do not silently implement a different mapping.
5. **Lower through the canonical Model IR.** Preserve operator/tensor identities and typed decisions. Reject unknown source ownership and unsupported temporary storage assumptions.
6. **Emit HLS C++ using existing facilities.** Inspect generated loops, local memory accesses, array shapes and source manifest identity. Support actual source-level numerical/reference comparison when possible.
7. **Add tests and documentation.** Demonstrate deterministic compilation, strict rejection, valid outputs, architecture variation and no licensed tool invocation at import/test time.
8. **Consider native HLS only separately.** Use a new isolated, create-only study directory and original source/report hashes. Synthesis estimates are not routed implementation results.

## Acceptance examples

- **MatMul or Gemm:** correct 2D shape inference, supported transpose/bias semantics, a reference output and stable operator-local mapping.
- **Conv2D:** explicit NCHW shape, stride and pad behavior, correct output dimensions, supported kernel layout and valid tail handling. Grouped/dilated forms are not silently accepted if unsupported.
- **Nonlinear layers:** source arithmetic has an independent numeric reference; no unjustified spatial mapping is assigned to a scalar-only operator.
- **Multi-operator networks:** tensor dependencies and intermediate storage remain valid across the whole supported chain. Native support for individual layers does not automatically imply support for arbitrary graph composition.

## Minimum test matrix

| Test | Requirement |
|---|---|
| Valid tiny graph | Generates the documented output shape and reference values |
| Malformed graph or unsupported attr | Explicit, relevant failure |
| Changed hardware request | Deterministic legality response and emitted-source difference or documented equivalence |
| Non-divisible dimension | Defined tail handling or an explicit rejection |
| Numeric variants | Reproducible acceptance under a stated tolerance |
| Provenance | Complete effective architecture and source identity recorded |
| Optional vendor phase | Reproducible original native tool invocation and strict timing/resource qualification |

## Research integrity

No compiled C++ or source certificate alone establishes successful Vitis simulation, synthesis, routed WNS, FPGA BRAM utilization or runtime. Existing study receipts remain immutable. New feature claims should distinguish **semantic support**, **source materialization**, **numerical evidence**, **synthesis evidence**, and **physical evidence**. Preserve negative results, unsupported cases and source provenance.

All contributions are subject to the repository's currently unresolved source-distribution and intellectual-property approval requirements.
