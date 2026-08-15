# Multi-port HLS physical composition

FPGAI's DAG mixed-backend composer supports HLS nodes with multiple AXI-stream inputs or outputs. Each stream has independent `TDATA`, `TVALID`, and `TREADY` semantics. Multi-port HLS bindings must explicitly name the generated AXI-stream argument prefixes so graph tensor order is never inferred from RTL declaration order.

The maintained validation graph combines multi-port HLS and multi-port VHDL in one design:

`input -> HLS split -> parallel HLS branches -> HLS add -> VHDL split -> VHDL add -> output`

The HLS policy is `axis_independent_ports`; the VHDL multi-port policy remains `grouped_transaction`. The physical report records both policies and whether multi-port HLS is present.

This profile intentionally keeps implicit fanout disabled. Replication remains an explicit graph operation so buffering and backpressure are visible to compilation and validation.
