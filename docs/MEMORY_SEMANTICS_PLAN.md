# FPGAI memory semantics plan

## Purpose

This document defines the compiler contract for FPGAI memory semantics. It is intentionally artifact-driven: a memory mode is considered supported only when the YAML request reaches the compiler resolver, the memory plan, generated HLS source, runtime package metadata, reports, and tests.

## Required terminology

Use these terms instead of ambiguous `ddr`:

- `embedded_weights`: weights compiled as constants.
- `bram_weights`: weights stored in local BRAM-backed buffers.
- `uram_weights`: weights stored in local URAM-backed buffers.
- `ddr_preload_weights`: weights loaded from external DDR into full local W/B buffers before compute.
- `ddr_tiled_weights`: weights remain external and only tiles are loaded into local tile buffers during compute.
- `ddr_tiled_mutable`: training/update mode where weights and updated weights are imported/exported through tiled DDR buffers.

## Compiler contract

### embedded_weights

- No `weights_mem` top argument.
- No runtime weight payload required.
- Generated compute consumes constant W/B arrays.

### bram_weights

- Weights are local on-chip W/B buffers.
- Weight buffers are BRAM-bound when legal in Vitis HLS.
- No full external DDR ownership is required.

### uram_weights

- Weights are local on-chip W/B buffers.
- Weight buffers are URAM-bound when the target board supports URAM.
- BRAM may still be used for activations, FIFOs, and scratch buffers.
- BRAM must not be interpreted as weight BRAM unless source/audit proves it.

### ddr_preload_weights

- `weights_mem` top argument exists.
- `m_axi port=weights_mem` exists.
- Runtime weight payload exists.
- Full model W/B arrays are allocated locally.
- This mode is only valid when the full weight set fits local memory.
- This mode is not a scalable large-network DDR-resident mode.

### ddr_tiled_weights

- `weights_mem` top argument exists.
- `m_axi port=weights_mem` exists.
- Runtime weight payload exists.
- No full local W/B model replica is allocated.
- Only tile-sized local buffers are allocated.
- Local storage scales with tile size, not model size.
- Dense and Conv inference generated-HLS paths are structurally implemented in the existing HLS top emitter.

### ddr_tiled_mutable

- Training top exposes runtime weight memory.
- Weights are imported as tiles before local compute/update.
- Updated weights can be exported back through tiled DDR buffers.
- Testbench/runtime preload paths treat this as a runtime weight-buffer mode.
- Optimizer-state and gradient movement must remain separately reported.

## Current implementation status after P2H inspection

The earlier Sprint 28A boundary was correct at the time: generic `ddr` should not be claimed as scalable DDR-resident execution unless generated HLS proves tiled behavior. The current repository is further along now.

Validated focused memory tests now cover structural generated-source and packaging behavior for:

- Dense inference `ddr_tiled` HLS rewriting.
- Conv inference `ddr_tiled` HLS rewriting.
- Mixed Conv/Dense DDR-tiled offsets.
- Memory semantics classification for `ddr_tiled`.
- Runtime package requirements for `ddr_tiled` payloads.
- HLS testbench runtime weight-buffer handling.
- Training `ddr_tiled_mutable` generated-HLS markers and preload/export behavior.
- Optimizer-state DDR-tiled reporting.

Passing focused validation command:

```bash
python -m pytest -q   tests/test_memory_storage_effect.py   tests/test_memory_semantics_classifier.py   tests/test_training_memory_storage_contract.py   tests/test_runtime_package.py   tests/test_hls_testbench_runtime_weights.py   tests/test_hls_training_testbench_runtime_preload.py   tests/test_training_optimizer_loss_contract.py
```

Expected/current result:

```text
114 passed
```

## Current claim boundary

Safe current claim:

FPGAI has structural generated-HLS, testbench, runtime-package, and memory-semantics support for embedded, BRAM/URAM local, DDR-preload, DDR-tiled inference, and DDR-tiled mutable training paths, with focused tests validating the generated artifact markers and package/report behavior.

Not yet safe claim:

FPGAI has fully validated scalable DDR-resident large-network execution across the full paper suite and real board runtime. That requires selected compile-level HLS/Vivado validation and later physical-board runtime validation.

## Required next validation

1. Keep `ddr_preload_weights` and `ddr_tiled_weights` distinct in reports.
2. Run selected compile-level examples for DDR-tiled inference/training.
3. Confirm generated HLS source, memory semantics report, runtime package metadata, and HLS/Vivado artifacts for those selected examples.
4. Only after physical-board execution should runtime result counts move above zero.
