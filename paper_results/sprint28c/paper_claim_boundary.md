# Paper claim boundary after Sprint 28C

## Safe claims

FPGAI generates HLS designs, prediction artifacts, runtime packages, and Vitis HLS reports for the validated paper matrix.

For inference memory/data movement, the current validated modes are:

- `embedded_constants`: weights are compiled into the generated HLS design.
- `ddr_preload_full`: weights are loaded from external AXI memory into full local W/B arrays before compute.
- `uram_preload_full`: weights are loaded from external AXI memory into full local URAM-bound W/B arrays before compute.

For precision, FPGAI changes generated fixed-point HLS types and affects LUT, FF, DSP, and latency. In the current small model, BRAM18 remains constant because memory allocation is dominated by buffer granularity and non-weight buffers.

For training, the validated path is embedded/BRAM-style training. Training DDR/URAM runtime weight storage is explicitly rejected until a real mutable load-update-store backend is implemented.

## Unsafe claims

Do not claim that current `ddr_preload_full` is scalable DDR-resident large-network execution.

Do not describe the current memory table as a pure BRAM-vs-DDR-vs-URAM storage-only comparison.

Do not claim measured board runtime, board power, board energy, or real-board accuracy unless separate board artifacts are generated.

## Future work needed for stronger memory results

To support large-network DDR claims, FPGAI needs a real `ddr_tiled` backend where only tile-sized weight buffers are local and the full model is not replicated into BRAM/URAM/local arrays.
