# FPGAI Community Extension and Implementation Ecosystem

## Architectural objective

FPGAI must allow external contributors to extend the compiler without modifying central compiler internals. Extensions remain explicit, versioned, YAML-selectable, compatibility-checked, numerically validated, and traceable through generated artifacts.

The compiler separates:

1. **Model semantics** — what graph/model is represented.
2. **Implementation** — C++/HLS, VHDL/RTL, or model-specific hardware.
3. **Memory architecture** — residency, allocation, buffering, checkpointing, recomputation, and optimizer-state placement.
4. **Streaming and transport** — local streams, AXI streams, DMA, packing, compression, prefetching, and overlap.
5. **Training architecture** — gradient computation/materialization, optimizer update, accumulation, and alternative learning mechanisms.
6. **Backend/platform** — toolchain, FPGA vendor, board, runtime, and report parsers.

## Extension classes

FPGAI will support independently registered extension kinds:

1. `model_family`
2. `operator_definition`
3. `operator_implementation`
4. `model_implementation`
5. `ir_pass`
6. `memory_policy`
7. `streaming_policy`
8. `transport_policy`
9. `training_mechanism`
10. `numerical_policy`
11. `backend`
12. `toolchain`
13. `board`
14. `runtime_adapter`
15. `validation_adapter`
16. `report_parser`

## Contribution levels

Contributors may add:

- new model families and import/pattern mappings;
- reusable implementations of existing operators in HLS C++, VHDL, or later Verilog/SystemVerilog;
- complete model-specific accelerators;
- memory planners and residency/buffering mechanisms;
- streaming/dataflow and transport policies;
- training and optimizer mechanisms;
- precision/numerical policies;
- compiler IR passes;
- vendor backends, boards, toolchains, runtimes, simulators, and report parsers.

## Standing requirements

Every extension must declare:

- stable extension ID, kind, version, author, and license;
- FPGAI ABI/API compatibility;
- inference and training capabilities;
- supported operators/models;
- precision, interface, memory, board, and toolchain assumptions;
- configuration schema and YAML selection path;
- implementation language and source ownership;
- validation artifacts and current support status;
- unsupported combinations and diagnostic messages.

Source presence alone never means support. Status promotion is explicit:

`registered -> schema_validated -> simulation_validated -> numeric_validated -> synthesized -> implemented -> board_validated`

## Sprint program

### F5C — Generic numerical-equivalence engine

- Reuse existing Python/reference, HLS CSim, weight, gradient, loss, and optimizer-state captures.
- Define implementation-agnostic tensor/result capture schemas.
- Compare pre/post loss, outputs, gradients, weights, biases, Adam `m/v`, optimizer step, norms, max/mean absolute error, RMSE, and relative error.
- Emit machine-readable and human-readable equivalence artifacts.
- Keep workload identity separate from implementation identity.
- Allow future VHDL/simulator adapters to produce the same capture schema.

### F5D — Matched architecture evaluation

- Require identical workload fingerprints.
- Compare `full_buffer`, `tiled_accumulate`, and `fused_update` after F5C passes.
- Collect HLS latency/II/resources, Vivado timing/Fmax/slack/resources/power, and board runtime/energy.
- Promote claims only according to validation status.

### E1 — Extension ABI and manifest schema

- Define common extension metadata and per-kind schemas.
- Add semantic versioning and FPGAI compatibility ranges.
- Add discovery, registration, conflict handling, deterministic resolution, and version locks.
- Add `fpgai extension validate/list/inspect/install` CLI contracts.
- Begin declaratively; Python hooks are reserved for extensions requiring compiler logic.

### E2 — Model and operator extension registries

- Model-family importer/pattern registry.
- Operator-definition registry.
- Alternative operator-implementation registry.
- Whole-model implementation matching and replacement scope.
- Explicit YAML selection and policy-based selection.
- Preserve built-in implementations as selectable modes.

### E3 — Multi-language implementation adapters

- Built-in HLS C++ adapter contract.
- VHDL/RTL entity, generics, ports, source lists, clock/reset, constraints, and simulation contract.
- Later Verilog/SystemVerilog adapters.
- Mixed HLS/RTL Vivado project assembly.
- Typed tensor/interface adapters between implementation languages.

### E4 — Memory-policy extension ABI

- Stable liveness/access/mutation analysis input contract.
- Allocation/residency plan output contract.
- BRAM/URAM/DDR placement, aliasing, ping-pong/double buffering, caching, paging, prefetch, checkpointing, recomputation, compression, and sparse storage.
- Training-aware targets, gradients, optimizer state, and checkpoint semantics.

### E5 — Streaming and transport extension ABI

- Graph stream partitioning and producer-consumer fusion.
- FIFO/channel planning and depth selection.
- Fork/join, multi-rate, backpressure, and deadlock validation.
- AXI Stream, m_axi/DMA, burst, scatter-gather, packing, compression, multi-bank, and compute-transfer overlap.
- Inference and training parity.

### E6 — Training and numerical extension ABI

- Gradient, accumulation, optimizer, checkpointing, compression, and alternative-learning mechanisms.
- Fixed point, block floating point, FP8, posit/logarithmic experiments, scaling, rounding, saturation, and QAT policies.
- Common tensor/memory/transport/execution semantics shared with inference.

### E7 — Backend, board, runtime, and toolchain ecosystem

- Vendor/toolchain adapters.
- Board resource/interface/clock/memory/deployment contracts.
- Runtime and telemetry adapters.
- Simulator and synthesis/implementation report parsers.
- Reproducible tool/version compatibility reporting.

### E8 — Community validation and support-status promotion

- Reference-vector generation.
- Simulator adapters for HLS CSim and RTL simulation.
- Numeric equivalence through F5C capture schema.
- Synthesis, implementation, timing, power, and board-runtime validation.
- Automated support-status promotion and regression requirements.

### E9 — Mixed-implementation composition

- Compose built-in and community operators in one graph.
- Compose HLS C++ and VHDL/RTL blocks.
- Validate clocks, resets, tensor layouts, rates, precisions, storage, and transport boundaries.
- Reject incompatible combinations with actionable diagnostics.

### E10 — Catalog, benchmarks, and reproducibility

- Local/remote extension catalog metadata.
- Authorship, license, version lock, supported models/boards/toolchains, examples, and validation level.
- Benchmark comparable implementations using common IR, workload fingerprints, implementation fingerprints, and artifact schemas.
- Do not make automatic online code installation a requirement for the core compiler.

## Immediate integration rule

All new compiler artifacts must carry two different identities:

- **workload fingerprint**: semantic/numeric execution contract used to determine fair comparability;
- **implementation-stack fingerprint**: exact model/operator/memory/streaming/transport/training/numeric/backend/toolchain/board selections used to reproduce the artifact.

Changing an implementation must not change the workload fingerprint when semantics remain matched. It must change the implementation-stack fingerprint.
