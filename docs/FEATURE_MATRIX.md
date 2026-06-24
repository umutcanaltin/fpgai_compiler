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

| Feature | Required public meaning | Current status | Required work before public support |
|---|---|---|---|
| YAML-first inspect/compile | User can inspect and compile from YAML through `fpgai` CLI. | supported | Keep tests and docs current. |
| Precision selection | YAML precision changes generated HLS/C++ data types and reports the applied precision. | partial | Prove generated code changes for each supported precision mode. |
| Memory/storage selection | YAML memory strategy changes generated storage interfaces/pragmas/weight handling and reports the applied strategy. | partial | Prove BRAM/URAM/stream/DDR-style selections change generated artifacts or downgrade unsupported modes. |
| Tiling selection | YAML/DSE tile choices change generated loop/helper code and local buffers. | partial | Prove tile sizes are emitted and connected to top/layer code, not only reports. |
| Pipeline selection | YAML pipeline style changes generated HLS pipeline pragmas/target II. | partial | Prove conservative/balanced/aggressive styles produce different generated pragmas or II metadata. |
| Parallelization selection | YAML PE/SIMD/unroll/partition knobs change generated unroll/partition pragmas and reports. | partial | Prove unroll and partition factors are materialized per supported layer. |
| Resource estimation | Compile/inspect emits pre-HLS LUT/FF/DSP/BRAM estimates. | supported_estimate_only | Keep clearly labeled as estimate unless compared to HLS/Vivado. |
| Timing estimation | Compile/inspect emits pre-HLS cycle/latency/timing estimates. | supported_estimate_only | Keep clearly labeled as estimate unless compared to HLS/Vivado. |
| Design-space exploration | DSE enumerates candidates, estimates them, and recommends candidates. | partial | Ensure DSE only recommends knobs that are materialized or marks them estimate-only. |
| HLS generation | Compiler emits HLS C++/headers/testbench/project artifacts. | supported | Keep generated artifact tests. |
| HLS run/artifact collection | When enabled and tools exist, compiler runs/collects HLS logs/reports. | supported | Keep optional-tool boundary documented. |
| Vivado bridge generation | Board-aware Vivado bridge scripts are generated for supported boards. | supported | Keep Vivado implementation clearly separate from compile. |
| Runtime package | Compile emits runtime package metadata and copies existing runtime-facing files. | supported | Keep hardware presence flags truthful. |
| Inference correctness benchmark | Supported inference flows compare outputs against reference/ONNX Runtime. | supported | Keep benchmark limitations documented. |
| Training support | Training config generates and validates training artifacts including forward/loss/gradient/update behavior. | experimental | Audit and either finish correctness/convergence support or downgrade all public claims. |
| Communication optimization | Compiler models data movement and records communication plans. | partial | Separate modeled transfer planning from measured board-level DMA runtime. |

## Implementation order

1. Precision selection
2. Memory/storage selection
3. Tiling selection
4. Pipeline selection
5. Parallelization selection
6. DSE knob truth
7. Training support
8. Communication optimization
