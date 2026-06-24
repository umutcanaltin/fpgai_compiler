# FPGAI Feature Matrix

This document defines what FPGAI features are allowed to claim publicly.

A feature is `supported` only when all of these are true:

1. A YAML knob exists.
2. The compiler reads the knob.
3. Generated HLS/C++ artifacts materially change.
4. The compile manifest or report records the applied setting.
5. A regression test proves the generated artifact changed.

## Status labels

| Status | Meaning |
|---|---|
| `supported` | Implemented, materialized in generated artifacts, recorded, and tested. |
| `supported_estimate_only` | Implemented as a pre-HLS analytical estimate, but not a measured HLS/Vivado result by itself. |
| `partial` | Some implementation exists, but at least one required proof point is missing. |
| `experimental` | Research/development path exists, but it is not a stable user-facing feature. |
| `not_yet_supported` | Public docs/configs should not claim support. |

## Core compiler features

| Feature | Required public meaning | Inference scope | Training scope | Current status | Required work before public support |
|---|---|---|---|---|---|
| YAML-first inspect/compile | User can inspect and compile from YAML through `fpgai` CLI. | supported | supported | supported | Keep tests and docs current. |
| Precision selection | YAML precision changes generated HLS/C++ data types and reports the applied precision. | supported | supported | supported | Keep materialization and generated-type regression tests current. |
| Memory/storage selection | YAML weight delivery and storage choices change generated HLS interfaces and parameter storage pragmas. | supported | supported | supported | Keep generated-interface and BIND_STORAGE regression tests current. |
| Tiling selection | YAML tiling choices change generated HLS dense/conv tiled helpers, tile template arguments, training backward/update call-site materialization, and architecture metadata. | supported | supported | supported | Training dense/conv forward and backward/update tiling are materialized in generated HLS call sites and metadata. |
| Pipeline selection | YAML pipeline style/II changes planner metadata and generated HLS layer call pipeline-II arguments. | supported | supported | supported | Keep planner-to-codegen pipeline regression tests current for inference and training. |
| Parallelization selection | YAML PE/SIMD/unroll/partition knobs change planner metadata, generated HLS call arguments, and array partition mode pragmas. | supported | supported | supported | Keep planner-to-codegen parallel regression tests current for inference and training. |
| Resource estimation | Compile/inspect emits pre-HLS LUT/FF/DSP/BRAM estimates. | supported_estimate_only | supported_estimate_only | supported_estimate_only | Keep clearly labeled as estimate unless compared to HLS/Vivado. |
| Timing estimation | Compile/inspect emits pre-HLS cycle/latency/timing estimates. | supported_estimate_only | supported_estimate_only | supported_estimate_only | Keep clearly labeled as estimate unless compared to HLS/Vivado. |
| Design-space exploration | DSE evaluates configured candidates and emits estimate-based recommendations with compile-ready/materialized-knob metadata. | supported_estimate_only | supported_estimate_only | supported_estimate_only | No exhaustive search is claimed; recommendations are pre-HLS estimates from configured candidates only. |
| HLS generation | Compiler emits HLS C++/headers/testbench/project artifacts. | supported | supported | supported | Keep generated artifact tests. |
| HLS run/artifact collection | When enabled and tools exist, compiler runs/collects HLS logs/reports. | supported | supported | supported | Keep optional-tool boundary documented. |
| Vivado bridge generation | Board-aware Vivado bridge scripts are generated for supported boards. | supported | supported | supported | Keep Vivado implementation clearly separate from compile. |
| Runtime package | Compile emits runtime package metadata and copies existing runtime-facing files. | supported | supported | supported | Keep hardware presence flags truthful. |
| Inference correctness benchmark | Supported inference flows compare outputs against reference/ONNX Runtime. | supported | not_applicable | supported | Keep benchmark limitations documented; training validation is separate. |
| Training code generation | Training configs generate HLS artifacts for forward, loss, gradient, update, optimizer, and training-specific data movement paths. | not_applicable | supported | supported | This is a generated-artifact/codegen claim, not a physical-board convergence claim. |
| Training single-step reference reports | Compile reports can record CPU/reference single-step training summaries such as loss_before, loss_after, and training estimate metadata. | not_applicable | supported | supported | This covers report generation and reference-summary traceability; numerical tolerance scope must remain in tests. |
| Training multi-step convergence | Multi-step training/convergence sweeps exist for research validation. | not_applicable | experimental | experimental | Do not claim stable convergence until reproducible loss/accuracy trends are validated and summarized as paper artifacts. |
| Training on-board runtime | Physical FPGA training runs and board-measured loss/timing artifacts. | not_applicable | experimental | experimental | Not stable until board runtime artifacts record bitstream, board, dataset, loss/timing, and reference comparison. |
| Communication optimization | Compiler models input/weight/output/aux tensor-edge data movement, per-edge precision/compression, transfer estimates, and generated HLS communication annotations. | supported | supported | supported | Compression codecs are modeled unless implemented_in_hls=true; measured board DMA speedup remains outside this claim. |

## Implementation order

1. Precision selection
2. Memory/storage selection
3. Tiling selection
4. Pipeline selection
5. Parallelization selection
6. DSE knob truth
7. Training code generation
8. Training single-step reference reports
9. Training multi-step convergence
10. Training on-board runtime
11. Communication optimization
