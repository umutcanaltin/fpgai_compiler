# IR-01 — source-grounded compiler capability audit (2026-11-01)

**Evidence status:** manual read-only inspection of the connected `main` source and named existing tests. This report records *implemented source pathways* and omissions; it is not a freshly executed CPU suite, CSim/CSynth result or proof of physical behavior. User previously reported 767/767 CPU tests at baseline `5221094af585bd501e842deca94d19c2289bb628`. No code has been removed for this compiler-paper track.

## Confirmed source-level capabilities

| Boundary | Source evidence | Current supported interpretation | Scope/limitation |
| --- | --- | --- | --- |
| Functional IR | `model_ir/contracts.py::FunctionalModel.validate` | Static named tensors; single-output, topologically ordered operators; explicit input/output/constant identities; checks missing dependencies and duplicate producers. | Functional tensor dtype is restricted to `float32`; one output per operator. |
| Architectural IR | `model_ir/contracts.py::ArchitecturePoint`, `architecture/resolver.py::resolve`, `architecture/scopes.py::resolve_scoped` | Operator-local requested and effective precision, parallelism, PE, SIMD, serialization, tile, pipeline II with model/region/operator/supported loop scopes. | No separate typed inter-operator *architectural* edge/interface contract in `ArchitecturalModel`; only `functional` and `points` fields. |
| Architectural legality | `architecture/resolver.py`, `architecture/mapping.py` | Rejects unsupported attributes/requests, checks `Phi` mapping and bounded collision/coverage. | Resolver can reject unsupported; does not prove HLS schedule, resource mapping or clock. |
| Lowered IR | `model_ir/lowering.py::lower`, `model_ir/semantics.py::verify_refinement` | Concrete loop structure/PE/SIMD/fold and HLS numeric type per op with preserving source scopes and effective-request identity. | `LoweredModel` only contains `architectural` and `operators`; no explicit graph-edge architecture interface/adapters. |
| Generated HLS | `codegen/hls/emitter.py::emit_hls` | Executable ordered operator computation using static, flattened C++ arrays per tensor; one whole-model HLS C++ function; source manifest and per-operator mechanisms. | **Exactly one graph input and one output** in current top. All operator points must share one `(precision_bits,integer_bits)` pair; emitter **rejects mixed numerical formats** with “Mixed per-operator numerical formats require explicit cast lowering”. Per-operator PE/SIMD differences do **not** alone require an adapter in this array-based source flow. |
| Whole-graph source examples | `tests/test_scopes.py`, `experiments/branch_ir_study.py` | MatMul→Relu per-operator scope; existing branched two Conv→Add→Relu→GlobalAveragePool test/study and independent per-op specialization. | **Do not claim graph composition is absent or newly invented by FPGAI**. Graph structural dependencies are already supported in the existing research compiler. |
| Independent mapping/source checks | `model_ir/phi_refinement.py`, `architecture/materialization_oracle.py`, `model_ir/output_ownership.py` | Bounded iteration ownership, spatial/temporal mapping and emitted-loop model checks. | Bounded/model-level. Algebraic large-domain division derivations are not machine-verified source/RTL proof. |
| Executed source conformance | `validation/cpp_phi_trace.py::trace_cpp_phi`, `tests/test_cpp_phi_trace.py` | Host-GCC executes generated C++ instrumented mapping and compares the resulting indices/PE/SIMD/fold with `Phi`. | `FPGAI_HOST_FLOAT` is not ap_fixed-bit-exact, Vitis timing, RTL identity or P&R evidence. |
| HLS tools / experiments | `toolchains/vitis_hls/`, paper/RL native protocol | Has a real source-attributed Vitis HLS integration. | Do not reuse the running source-pinned L0 RL data as a completed new compiler-IR graph-interface experiment. |

## Important correction to the earlier paper plan

The paper must not claim FPGAI newly introduces multi-operator graphs, operator-local architecture settings, scoped IR, explicit `Phi`, the current host trace, or graph branching: those are already implemented baseline capabilities. **The proposed next contribution** is explicit *typed architectural interface and composition semantics*, separate from existing functional tensor dependencies, including where typed boundary properties are compatible, where declared transport/layout/schedule contracts require a real implementation, and why unsupported combinations must fail closed. This must be substantiated by new implemented features, a restricted same-compiler baseline and measurable graph-level boundary cases.

Do **not** invent incompatibility between a producer's PE count and consumer's PE count under current array-based materialization. Likewise mixed precision is currently unsupported; it is a source-backed blocker, not a shipped conversion feature. Hardware streaming, FIFOs, conversion logic, per-edge clock domains, dynamic tensor shapes, or physical buffer topology require future implementations, not just metadata fields.

## Near-term engineering decisions (no RL edits)

1. **E0 source-only implemented; workstation testing pending:** `architecture/interfaces.py::inspect_array_interfaces` records the current source-array producer/consumer boundaries and explicit format/top-arity blockers without touching the emitter; `tests/test_architecture_interfaces.py` covers chain, fan-out/merge, mixed numeric and top arity. This is baseline evidence preparation, NOT the proposed novel typed graph-level interface. Include explicit mixed-format rejection and multi-consumer/source provenance.
2. Add typed graph-edge architectural contracts only after tests define their semantics and an implementation in HLS lowering. Prefer input/output type and transport granularity compatibility over fabricated dataflow assumptions.
3. Implement and independently verify a concrete interface requirement not already covered by functional `Tensor` and per-op `Phi`. If no substantive capability can be shown, change thesis rather than claiming novelty.
4. Benchmark against a deliberately restricted uniform-mapping configuration using the **same** compiler code; distinguish configuration expressiveness from QoR, and source HLS claims separately.
5. Do a small original-Vitis native campaign only after CPU and host source tests are green; never use a long RL study as a substitute.

## Frozen RL source and backup

- Original running/source-verified L0 research revision: `5221094af585bd501e842deca94d19c2289bb628`.
- Backup branch of the *entire repository at that revision*: `backup/rl-environment-frozen-2026-11-01`.
- RL modules `rl_environment/` and `paper/rl_experiments/` are untouched by this audit. The active workstation RL checkout must not pull remote `main` until native experiment and source audit finish.
