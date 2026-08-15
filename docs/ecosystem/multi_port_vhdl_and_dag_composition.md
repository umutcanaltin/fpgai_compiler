# Multi-port VHDL and mixed-backend DAG composition

FPGAI supports a grouped multi-port VHDL contract for hardware nodes that consume or produce more than one tensor in one logical transaction.

## ABI

The ABI is named:

```text
tensor_ports_ready_valid_v1
```

Each package declares ordered logical input and output ports with explicit RTL data-port names, widths, signedness, clock/reset names, and one grouped ready/valid handshake.

```yaml
integration:
  vhdl:
    abi: tensor_ports_ready_valid_v1
    handshake_policy: grouped_transaction
    data_width: 16
    inputs:
      - name: left
        data: left_data
      - name: right
        data: right_data
    outputs:
      - name: output
        data: output_data
```

`grouped_transaction` means all declared inputs are accepted atomically and all declared outputs are released atomically. This avoids guessing how unrelated independent handshakes should synchronize.

## Physical DAG profile

The maintained physical profile is:

```text
dag_grouped_ready_valid_v1
```

The initial maintained graph is an explicit fork/join topology:

```text
Input
  ↓
HLS scale×2
  ↓
VHDL split
  ├───────────────┐
  ↓               ↓
HLS +1         HLS ×2
  ↓               ↓
  └─────→ VHDL add ←─────┘
             ↓
           Output
```

For input `7`, the expected output is `43`:

```text
7 → 14
split → 14, 14
left branch → 15
right branch → 28
merge → 43
```

## Handshake semantics

The compiler-owned wrapper uses explicit grouped synchronization:

- multi-input join: a node sees `input_valid` only when all input tensors are valid;
- each upstream input is acknowledged only when the grouped node can accept the transaction and all peer inputs are valid;
- multi-output split: the node sees `output_ready` only when all output paths are ready;
- each output path sees a valid transaction only when the grouped output can be accepted atomically.

The VHDL split and merge reference implementations use one-entry elastic registers, so data and valid remain stable under downstream backpressure.

## Current boundaries

The profile intentionally rejects implicit tensor fanout. Branching must use an explicit multi-output node so transaction replication and backpressure behavior are represented in the graph and reports.

Current scope:

- one graph input and one graph output;
- homogeneous scalar width per maintained physical graph;
- unary AXI-stream HLS stages;
- grouped multi-port VHDL stages;
- explicit split and merge nodes;
- ready/valid backpressure;
- XSim numeric validation and Vivado synthesis project generation.

Future work generalizes multi-port HLS, heterogeneous widths, independent-port handshakes, and wider DAG topologies.
