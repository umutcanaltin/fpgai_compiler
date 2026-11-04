<!-- Preserved historical compiler/operator and separate product research notes. -->

# Adding a new compiler operator or AI layer

This guide is for researchers who want to extend FPGAI's **compiler**, not just add an operator label to a Gym observation.

## Source ownership

The authoritative compiler code is organized by responsibility:

| Responsibility | Location |
|---|---|
| Graph import | `src/fpgai_research/frontend/` |
| Typed tensors/operators and shape semantics | `src/fpgai_research/model_ir/` |
| Reference execution and samples | `src/fpgai_research/validation/` |
| Legal architecture requests and restrictions | `src/fpgai_research/architecture/` |
| Lowering, refinement proofs, composition | `src/fpgai_research/compiler/`, `model_ir/lowering.py` |
| HLS C++ emission | `src/fpgai_research/codegen/hls/` |
| Vendor invocation and source attestation | `src/fpgai_research/toolchains/vitis_hls/` |
| Searchable legal knobs | `src/fpgai_research/experiments/architecture_knobs_v2.py` |
| Optional native Gym benchmark | `rl_environment/cohort.py`, `curriculum.py`, `environments/` |

The current research backend supports particular MatMul/Gemm paths; this does not mean all future operators have already been implemented or validated.

## Contribution process

1. **Define a semantic contract**: operator name, input/output ranks and shapes, attributes, broadcasting, indexing, numeric precision/overflow behavior, whether computation is stateful or temporal, and a deliberately small reference example.
2. **Implement reference correctness first**: deterministic samples and independent expected outputs, with tests covering corner cases, invalid shapes, numerical errors and rounding.
3. **Support typed import and model verification**: adapt the frontend and typed Model IR without leaking compiler-specific hardware tiling choices into importer semantics. Check model graph connections and output dimensions explicitly.
4. **Lower and legalize architecture**: choose per-operator hardware knobs via the architecture library. Vendor hardware and tool limits must remain in compiler-side abstractions, not in Gym policy code.
5. **Emit real hardware source**: generate VHDL or the currently qualified HLS C++ research pathway as appropriate, preserving the model's semantics; reject unimplemented variants clearly.
6. **Test all layers**: at least import, shape and type checks, numerical reference, legal architecture enumeration, deterministic code generation, source digests and no-Vitis package integration. Run a small original HLS CSim + CSynth smoke **only after CPU tests** and independently re-audit the output.
7. **Extend Gym only after compiler qualification**: add a separately sourced experiment with an appropriate operator encoding and heldout cohort. The agent should see the operator's graph and legal hardware knobs, never be responsible for implementing missing compiler semantics.

## Suggested minimum test matrix

| Test | Expected evidence |
|---|---|
| Invalid shapes/types | Rejected before source generation |
| Baseline numeric result | Matches independent reference within documented tolerance |
| Multiple shapes | Correct tensor shape and output semantics |
| Changed precision | Explicit legal numeric format or clear rejection |
| Different architecture knobs | Compiler-legal, deterministic emitted source |
| Round-trip serialization | Identical typed model fingerprint |
| HLS packaging | Stable C++/testbench/Tcl/manifest fingerprints |
| Native HLS (opt-in) | Original Vitis CSim, fresh CSynth, resource/clock evidence, independent audit |
| Gym stage (separate) | Source-pinned action mask, real-HLS-only reward and unseen heldout tasks |

For a new DRL algorithm instead, follow [Research Extension Architecture](RESEARCH_EXTENSION_ARCHITECTURE.md) and [Contributing](../CONTRIBUTING.md).

## Hardware and scientific constraints

The full heterogeneous, multi-device, VHDL-first five-layer Morfics product is broader than this Vitis research backend. Respect separate System IR, Model IR, Math IR, Circuit Architecture and Physical Implementation responsibilities, preserving clocks, temporal T/T+1 semantics, typed interconnect, CDC, PS integration and user-editable native constraints when applicable. Do not redesign the full compiler through an RL experiment-specific shortcut.

Never claim physical latency from HLS cycle counts or change historical source-pinned evidence. Add a new experiment and preserve earlier original Vitis receipts exactly as recorded.
