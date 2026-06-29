# FPGAI memory semantics plan

## Problem found in Sprint 28A

The Sprint 27J paper-validation matrix compiles and passes structural artifact checks, but the memory rows are not yet architecturally clean enough for strong paper claims.

Current inference memory modes are:

- embedded/local constants
- runtime DDR preload into full local W/B buffers
- runtime DDR load into URAM-bound full local W/B buffers

The current DDR path is not a scalable large-network DDR-resident design because generated HLS loads the full model weights from `weights_mem` into local W/B arrays before compute.

Therefore current results must not be presented as a pure BRAM-vs-DDR-vs-URAM storage-only comparison.

## Required terminology

Use these terms instead of ambiguous `ddr`:

- `embedded_weights`: weights compiled as constants.
- `bram_weights`: weights stored in local BRAM-backed buffers.
- `uram_weights`: weights stored in local URAM-backed buffers.
- `ddr_preload_weights`: weights loaded from external DDR into full local W/B buffers before compute.
- `ddr_tiled_weights`: weights remain external and only tiles are loaded into local tile buffers during compute.

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
- Weight buffers are URAM-bound.
- BRAM may still be used for activations, FIFOs, and scratch buffers.
- BRAM must not be interpreted as weight BRAM unless source/audit proves it.

### ddr_preload_weights

- `weights_mem` top argument exists.
- `m_axi port=weights_mem` exists.
- Runtime weight payload exists.
- Full model W/B arrays are allocated locally.
- This mode is only valid when the full weight set fits local memory.
- This mode is not a scalable large-network DDR mode.

### ddr_tiled_weights

- `weights_mem` top argument exists.
- `m_axi port=weights_mem` exists.
- Runtime weight payload exists.
- No full local W/B model replica is allocated.
- Only tile-sized local buffers are allocated.
- Local storage scales with tile size, not model size.
- This is the required mode for large-network external DDR execution.

## Current paper claim boundary

Safe current claim:

FPGAI materializes embedded weights, DDR-preloaded weights, and URAM-cached/preloaded weights for inference, and validates generated HLS source, runtime packages, and Vitis HLS reports.

Not safe current claim:

FPGAI already supports scalable DDR-resident large-network weight execution.

## Required next implementation

1. Add generated-code classifier:
   - `embedded_constants`
   - `bram_local`
   - `uram_local`
   - `ddr_preload_full`
   - `uram_preload_full`
   - `ddr_tiled`
   - `invalid_or_ambiguous`

2. Rename/report current DDR mode as `ddr_preload_full`.

3. Implement `ddr_tiled_weights` later for Dense first:
   - remove full local W/B arrays for DDR-tiled mode
   - load only `weight_tile` from `weights_mem`
   - compute using tile buffers
   - keep activation storage independently controlled

4. Add tests proving:
   - current DDR inference is classified as `ddr_preload_full`
   - current URAM inference is classified as `uram_preload_full`
   - embedded/BRAM rows are not mislabeled as scalable DDR
   - training DDR/URAM remains rejected until implemented

5. Rerun paper memory experiments on a larger model only after `ddr_tiled_weights` exists.
