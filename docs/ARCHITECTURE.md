# FPGAI architecture

FPGAI separates **what a model computes** from **which hardware organization is requested** and **which HLS mechanisms can be emitted**. This document describes the research implementation's boundaries, not a claim of a complete heterogeneous system compiler.

## Compiler stages

```text
Static ONNX graph
    |
    v
Functional Model IR -- typed operators, tensors, dependencies, shapes
    |
    v
Architectural Model IR -- complete per-operator requests and source scopes
    |                      precision, parallelism, PE, SIMD, serialization,
    |                      tiling and pipeline intentions
    v
Legality and refinement -- explicit supported features and source constraints
    |
    v
Lowered Model IR -- implementation-ready loops, assignments and provenance
    |
    v
HLS C++ emission + manifest
    |
    +--> optional independent source and numeric validation
    |
    +--> optional Vitis CSim / HLS synthesis (user-authorized)
    |
    +--> optional Vivado implementation (independently qualified)
```

Functional operators and tensor identities remain stable across requested and resolved mapping decisions. A target-dependent or unsupported request is **rejected explicitly** instead of quietly being rewritten as a success.

## Architecture dimensions

The research representation exposes numerical precision, coarse parallelism, PE count, SIMD lanes, serialization/folding, tiling and pipeline requests. These dimensions interact: more PE/SIMD units may require more memory bandwidth and change timing feasibility; temporal reuse does not imply a known physical schedule.

For a logical iteration `i` of operator `v`, the source architecture coordinate is a typed assignment of tile, replica, PE, lane, reuse step and pipeline relation. This is **not equivalent** to a guaranteed cycle-level RTL schedule. Source legality and physical qualification are deliberately distinct.

A `TargetLimits.max_parallel_units` restriction is a **source-side upper bound** on the requested parallel structure, not a calibrated FPGA utilization or power predictor.

## Compositional tensor contracts

The current research supplies **bounded static** source-event checking for selected MatMul producer/consumer structures. It distinguishes:

- reduction contributions and producer element commitment;
- legal reads only after a tensor element is ready;
- last required consumer read before source scratch reuse;
- value safety independently from the source issue order prescribed by an operator-local mapping;
- actual generated-source address witnesses for supported forms.

The bounded verifier does **not** prove arbitrary ONNX DAGs, native port arbitration, physical BRAM allocation, multi-clock hardware timing, or equivalent behavior for every possible input.

## Policy-neutral compiler actions

The compiler also exposes complete-design source states and typed actions through a policy-neutral transition interface. A proposed action can be **source certified**, **source rejected**, or **not established** for the bounded contract. The latter is not silently promoted to legal hardware.

An optional masked candidate catalog calls the real source compiler for each candidate. A cached certified source state may be selected without recompilation, but the upfront compiler cost remains. The separate RL environment is not part of this compiler's correctness argument.

## Code ownership

| Responsibility | Path |
|---|---|
| Import | `src/fpgai_research/frontend` |
| Data types, IR and semantic constraints | `src/fpgai_research/model_ir` |
| Scoped mapping and target restrictions | `src/fpgai_research/architecture` |
| Canonical compile orchestration | `src/fpgai_research/compiler` |
| C++ generation | `src/fpgai_research/codegen/hls` |
| Reference and generated-source validation | `src/fpgai_research/validation` |
| Native vendor boundaries and original receipts | `src/fpgai_research/toolchains` |
| Separate optimization research | `rl_environment` |

For extensions, [the contributor guide](../CONTRIBUTING.md) and [operator-specific guide](COMPILER_OPERATOR_CONTRIBUTION.md) identify the canonical points to update. Do not implement operator semantics inside an agent, vendor wrapper or experiment-specific independent emitter.
