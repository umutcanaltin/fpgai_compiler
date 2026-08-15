# Mixed-backend ready/valid physical profile

`linear_scalar_ready_valid_v1` extends FPGAI's graph-driven HLS/VHDL physical
composition with explicit backpressure. Both sides of every physical boundary
carry `data`, `valid`, and `ready`; valid/data must remain stable while the
consumer deasserts ready.

The maintained validation chain is HLS AXI-stream `scale2_axis` -> VHDL
one-entry elastic identity -> HLS AXI-stream `add1_axis`. The XSim testbench
holds the graph output not-ready before accepting the result and checks the
numeric result after backpressure is released.

This profile is intentionally scalar and linear. Multi-port/DAG physical
composition is a separate capability and must not be inferred from this
validation level.
