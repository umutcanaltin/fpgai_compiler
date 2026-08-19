# Quantized VHDL operator partition

FPGAI can partition the terminal quantized ReLU of the maintained residual-CNN
validation model into VHDL while keeping Conv and residual Add compute in HLS.

The maintained selection is YAML-driven:

```yaml
targets:
  hls:
    control_protocol: ap_ctrl_none
  mixed_backend:
    final_relu_backend: vhdl
```

Use `configs/examples/quantized_residual_cnn_ptq_mixed.yml` for the maintained
partitioned validation. PTQ calibration is performed on the complete model
first. The compiler then records the terminal ReLU input/output quantization
contracts, removes only that terminal ReLU from the HLS graph, validates the
remaining HLS graph against its integer reference, and preserves the complete
model integer reference for mixed-language XSim validation.

The first VHDL ReLU implementation consumes four signed int8 activations packed
least-significant-byte first into a 32-bit transport word. This implementation
requires equal input/output scale and zero point. Unsupported scale-changing
ReLU partitions are rejected instead of silently changing semantics.

## AXI-stream sidebands

DAG HLS physical composition now discovers optional `TKEEP`, `TSTRB`, and
`TLAST` signals in addition to `TDATA`, `TVALID`, and `TREADY`.

For fixed-size packetized tensors, `HLSPhysicalBinding.input_packet_words` and
`output_packet_words` define tensor packet boundaries. Input `TKEEP`/`TSTRB`
are driven explicitly, input `TLAST` is generated on the final word, and output
sidebands are connected and checked during simulation. Sideband behavior is
recorded in the physical report.

The current maintained int8x4 residual-CNN profile uses four complete 32-bit
words. Partial final-word masks are intentionally not inferred yet; a future
transport contract must provide the exact valid-byte mask for such tensors.
