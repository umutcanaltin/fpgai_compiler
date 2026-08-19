# Quantized residual CNN PTQ validation

FPGAI includes a maintained residual CNN PTQ validation path that connects representative-data calibration to integer HLS lowering.

The current validated profile uses signed 8-bit per-tensor activations, signed 8-bit per-output-channel Conv weights, 32-bit integer Conv accumulators, explicit integer requantization, quantized ReLU, and quantized residual Add.

Configuration is maintained in `configs/examples/quantized_residual_cnn_ptq.yml`. The configuration controls activation/weight bit widths, quantization scheme, granularity, rounding, saturation, calibration method and sample count, FPGA part, and HLS clock target.

The runner writes canonical reports under the selected output directory:

- `reports/ptq_calibration.json`
- `reports/quantized_hls_lowering.json`
- `reports/quantized_numeric_validation.json`
- `reports/quantized_hls_tool_result.json`

The integer HLS testbench compares the generated hardware path against FPGAI's integer lowering reference, separating calibration/quantization semantics from HLS implementation behavior.

Current HLS lowering coverage for this profile is Conv, ReLU, and Add. Unsupported operators or quantization granularities fail explicitly rather than falling back to float arithmetic.
