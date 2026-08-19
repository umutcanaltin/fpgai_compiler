# Quantized hardware lowering

FPGAI represents resolved PTQ/QAT parameters on IR tensors. Physical mixed-backend
composition can now use those tensor contracts to derive stream widths and insert an
explicit compiler-owned requantization node.

## Requantization contract

The initial hardware profile supports per-tensor affine integer quantization:

`q_dst = round((q_src - zp_src) * scale_src / scale_dst) + zp_dst`

The compiler converts the scale ratio to a constant integer multiplier and right
shift. The generated bridge follows the destination tensor's rounding and overflow
policy and records whether the conversion is classified as lossy.

Supported in this profile:

- heterogeneous integer stream widths from 2 to 32 bits;
- signed symmetric or asymmetric per-tensor parameters;
- nearest, floor, and ceil rounding;
- saturating or wrapping destination overflow;
- ready/valid passthrough with backpressure;
- HLS and VHDL nodes on either side of an explicit requantization node.

Per-channel activation requantization requires channel-aware stream semantics and is
not silently lowered by this profile.

## Physical DAG representation

A width/scale change is explicit in the IR:

```text
int8 tensor
    |
Requantize
    |
int16 tensor
```

This prevents implicit reinterpretation of stream bits at backend boundaries. The
physical report records tensor widths and the full `fpgai.requantization-contract/v1`
for each compiler-owned bridge.

## Validation profile

The maintained `quantized_bridge` profile in
`scripts/run_dag_mixed_backend_validation.py` validates:

```text
int8 (scale 0.5)
    -> requantize to int16 (scale 0.25)
    -> HLS Add1
    -> requantize to int8 (scale 0.5)
```

For integer input `7`, the expected path is `7 -> 14 -> 15 -> 8`.
