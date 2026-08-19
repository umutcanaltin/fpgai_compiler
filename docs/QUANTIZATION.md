# Quantization

FPGAI treats quantization as a compiler-visible numeric contract rather than a reporting-only feature.

## Modes

`numerics.quantization.mode` accepts:

- `none`: existing fixed-point precision policy without PTQ/QAT calibration semantics.
- `ptq`: post-training quantization using a representative calibration set.
- `qat`: quantization-aware training using fake quantization and a straight-through gradient contract.

## PTQ foundation

The current PTQ implementation provides deterministic calibration primitives for:

- symmetric and asymmetric quantization;
- signed and unsigned integer ranges;
- per-tensor activation quantization;
- per-channel weight quantization;
- min/max calibration;
- percentile calibration;
- nearest/floor/ceil rounding;
- saturating or wrapping overflow behavior;
- quantize/dequantize/fake-quant reference execution;
- numeric error validation.

Quantization parameters contain the scale, zero point, observed range, bit width, scheme, granularity, axis, rounding policy, and saturation policy.

## QAT foundation

QAT uses the same quantization parameter contract as PTQ. The current reference primitive provides fake-quant forward execution and an explicit straight-through gradient contract. Integration into the generated training graph and hardware lowering is the next implementation stage.

## IR semantics

Tensor entries can carry quantization metadata. This allows future lowering and physical-boundary code to make width, packing, scale, zero-point, rounding, and saturation decisions from IR-visible information rather than hidden analysis state.

## Example

```yaml
numerics:
  quantization:
    mode: ptq
    weights:
      bits: 8
      scheme: symmetric
      granularity: per_channel
      axis: 0
    activations:
      bits: 8
      scheme: symmetric
      granularity: per_tensor
    calibration:
      method: percentile
      percentile: 99.9
      samples: 128
```

This configuration is a numeric contract. A compilation flow must not claim PTQ/QAT hardware validation until calibrated parameters have been propagated through lowering, generated HLS/VHDL types, transport packing, and numeric validation.

## Model-level PTQ calibration

`fpgai.quantization.calibrate_model_ptq` calibrates an FPGAI graph from a representative dataset and a caller-supplied tensor-trace executor. The trace callback keeps quantization independent from any one frontend while reusing the compiler's maintained reference execution path.

The model-level result attaches resolved quantization parameters directly to FPGAI tensor metadata, returns integer-valued quantized constants, and records per-tensor numeric validation metrics. Calibration datasets can be loaded from `.npy` or `.npz` through `load_calibration_samples`; the first dimension is the sample dimension.

For `.npz` files with multiple arrays, configure an explicit `array_key`. This avoids guessing which array is the model input.
