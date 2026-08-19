# Quantized residual CNN mixed-backend validation

This maintained validation composes the already calibrated and synthesized PTQ residual CNN HLS top with a VHDL elastic transport stage.

The neural-network compute remains in Vitis HLS in this step. The VHDL stage validates a real mixed-language physical boundary carrying packed quantized activations; it is intentionally reported as a transport bridge rather than misreported as a partitioned neural-network operator.

## Transport contract

- activation element: signed int8
- AXI-stream transport word: 32 bits
- packing: four int8 values per word
- lane order: least-significant byte first
- VHDL handshake: grouped ready/valid, one-word elastic buffer

## HLS control protocol

The maintained PTQ example selects:

```yaml
targets:
  hls:
    control_protocol: ap_ctrl_none
```

This is required for direct streaming physical composition. The default DAG HLS behavior remains `s_axilite`; the choice is explicit and selectable.

## Validation levels

The runner performs:

1. exact packed-word XSim comparison against the integer PTQ reference,
2. mixed HLS/VHDL synthesis,
3. optional Vivado opt/place/route,
4. implementation timing, utilization, and power reporting.

Implementation reports use WNS/TNS as timing-constraint results. They do not claim measured Fmax.

## Command

First regenerate the quantized HLS project after selecting `ap_ctrl_none`, then run:

```bash
python scripts/run_quantized_residual_cnn_mixed_backend.py \
  --ptq-build build/quantized_residual_cnn_ptq \
  --out build/quantized_residual_cnn_mixed_backend
```

Use `--synthesis-only` to stop before place/route.
