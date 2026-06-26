# FPGAI Project Status

## Current sprint

Sprint 21: full real-pipeline sweep validation toward paper artifacts.

## Latest completed change

Fixed CNN inference HLS code generation.

Completed fixes:
- Tiled Conv HLS helper now supports flat parameter arrays emitted by the parameter generator.
- Stale default type aliases in tiled Conv helper were removed.
- Parameter emitter no longer emits file-scope `BIND_STORAGE` pragmas that Vitis HLS rejects during csynth.
- `configs/sweeps/inference_policy.yml` now uses real precision candidate `fx16_6` instead of stale `fixed`.

Validated:
- `inference_precision_single`: passed.
- `inference_precision`: 4/4 passed.
- `inference_policy`: 4/4 passed.
- `inference_policy_fx16_check`: 1/1 passed with no skipped/unapplied precision mode.

## Truth boundary

Supported after this sprint:
- CNN inference HLS generation through CSim and csynth for the tested sweeps.
- Precision and policy sweeps generate real HLS reports for tested CNN designs.

Still to validate:
- All remaining sweeps across precision, tiling, memory, parallelism, pipeline, training, and hardware knobs.
- Vivado report generation.
- Bitstream generation.
- Real board runtime inference/training timing and accuracy plots.

## Next step

Run full sweep validation in gates:
1. Inspect every sweep config.
2. Run every HLS/benchmark sweep.
3. Collect artifact sensitivity and HLS reports.
4. Run Vivado report/bitstream paths only after HLS sweeps are green.

## Sprint 23A precision materialization fix

- Found precision sweep bug: `precision_mode` was applied only to `analysis.precision_sweep.selected_candidate`.
- Generated/materialized configs kept `numerics.defaults` at `ap_fixed<16,6>` for all precision modes.
- Patched `fpgai/experiments/config_materializer.py` so `precision_mode` also materializes:
  - `analysis.precision_sweep.selected_candidate`
  - `numerics.precision_mode`
  - `numerics.defaults`
- Next validation: regenerate a small precision sweep and verify generated HLS types differ across `fx8_3`, `fx10_4`, `fx12_4`, `fx14_5`, `fx16_6`.

## Sprint 23B.3 precision layout activation validation

The precision layout report now correctly detects nonzero activation/input/output counts.

Validation sweep:

`paper_experiments/full_pipeline_gate/sweeps/precision_layout_report_fixcheck_v2`

For the same model, element counts are stable across precision modes:
- input elements: 784
- output elements: 10
- weight elements: 6798
- bias elements: 14
- activation-buffer elements: 14387

Precision-dependent byte counts now scale correctly.

`fx8_3`:
- input raw bytes: 784
- output raw bytes: 10
- weight raw bytes: 6798
- bias raw bytes: 28
- activation-buffer raw bytes: 14387
- input AXIS bytes: 784
- output AXIS bytes: 12
- weight AXI bytes: 6800
- activation AXI bytes: 14400

`fx16_6`:
- input raw bytes: 1568
- output raw bytes: 20
- weight raw bytes: 13596
- bias raw bytes: 42
- activation-buffer raw bytes: 28774
- input AXIS bytes: 1568
- output AXIS bytes: 20
- weight AXI bytes: 13600
- activation AXI bytes: 28784

Conclusion: precision now affects the compiler's central accounting for activations, weights, bias, accumulators, AXIS communication, AXI/DDR communication, and activation storage pressure.

Next implementation step: wire this precision layout into real generated HLS AXIS input/output packing, then DDR/runtime weight packing, then embedded BRAM packing.
