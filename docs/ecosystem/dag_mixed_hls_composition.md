# Mixed HLS DAG composition

FPGAI's branch-aware HLS path extends the existing mixed built-in/external
compiler flow beyond the historical `current_buffer` sequential profile.

## Current maintained DAG profile

The first maintained graph is:

```text
             +-> Relu --------+
Input -------+                Add -> Sigmoid -> Output
             +-> ScaleBias ---+
                 external HLS
```

The compiler now:

1. builds producer/consumer and tensor live ranges;
2. allocates reusable activation buffers only across non-overlapping live
   ranges with compatible resolved scalar types;
3. records generated-buffer <-> IR-tensor provenance;
4. schedules supported unary operators from their declared tensor inputs;
5. invokes external `flat_array_v1` implementations from any legal unary
   branch input, rather than only the historical current tensor;
6. lowers `Add` through the existing `add_vec_typed` HLS implementation;
7. preserves the sequential emitter for non-branching graphs.

## Reports

The ecosystem compile emits:

- `reports/tensor_liveness.json`
- `reports/hls_buffer_allocation.json`
- `reports/hls_bottleneck_diagnostics.json`

The buffer-allocation report is the source of generated resource provenance
used by bottleneck diagnostics. Architecture mechanisms remain user-selectable;
FPGAI does not silently enable partitioning or alternative memories.

## Current boundaries

The branch-aware emitter intentionally supports a bounded first profile:

- one graph input and one graph output;
- static tensor shapes;
- built-in Relu, LeakyRelu, Sigmoid, Identity/Flatten/Reshape, and Add;
- unary external `flat_array_v1` HLS implementations;
- equal flattened sizes for Add operands.

Conv/Dense DAG scheduling, arbitrary multi-output nodes, and multi-input
external ABIs remain later extensions. Unsupported cases fail explicitly.
