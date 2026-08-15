# Graph-driven mixed-backend physical composition

`fpgai.graph-mixed-backend-physical/v1` turns an FPGAI IR node chain plus explicit physical implementation bindings into one mixed-language Vivado project.

The first maintained physical profile is deliberately strict: one graph input, one graph output, one runtime tensor per node, one output tensor per node, equal scalar data width, and `valid + data` transport. Both `vitis_hls -> vhdl` and `vhdl -> vitis_hls` boundaries are emitted and reported. Unsupported graph shapes, widths, or binding gaps fail with `MIXGRAPH###` diagnostics; FPGAI does not silently fall back to another backend.

The maintained tool validation uses `HLS(scale x2) -> VHDL(identity) -> HLS(add 1)`. For input `7`, XSim must observe output `15`, after which the same project is synthesized and emits utilization and timing reports.

This profile is a physical-composition foundation, not an AXI-stream claim. Ready/backpressure, multi-input, multi-output, arbitrary DAG stitching, and training boundaries remain separate capability stages.
