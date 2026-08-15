# Advanced multi-backend validation

This batch extends the FPGAI research compiler in four directions while preserving existing validated flows.

## `tensor_ports_v1` heterogeneous ports

`tensor_ports_v1` now supports optional per-port `scalar_type`, static `shape`, `layout`, and `count_mode`.

- `count_mode: shared` preserves the original equal-size ABI.
- `count_mode: per_port` allows heterogeneous flattened sizes and emits an explicit count for every input and output port.

This keeps package interfaces deterministic and avoids inferring tensor sizes inside contributed kernels.

## Residual CNN example

`mixed_external_residual_cnn` is the maintained next-step DAG model. It combines two built-in Conv nodes, Relu nodes, an external ScaleBias skip branch, and Add merge. It is intended to validate parameterized operators, residual liveness, feature-map buffering, and external package composition in the same graph.

## External VHDL validation

The VHDL integration project now generates a behavioral VHDL testbench as well as synthesis Tcl. The maintained `scalar_stream_v1` profile uses an explicitly declared `reference_behavior` so numeric simulation behavior is part of the package contract rather than inferred from the operator name.

`run_external_vhdl_project()` executes Vivado batch mode and records simulation/synthesis status separately from project generation.

## Vivado clock sweep

`scripts/run_vivado_clock_sweep.py` performs repeated full FPGAI compile/implementation runs at requested clocks. The resulting report gives the highest passing implementation frequency and, when available, the first failing frequency above it. This is an implementation-tested Fmax bracket and replaces unsupported WNS-to-Fmax claims.

## Mixed HLS/VHDL planning

The mixed-backend planner records HLS and VHDL segments and explicit backend boundaries. Direct RTL stitching remains intentionally marked unsupported until a validated bridge implementation exists.
