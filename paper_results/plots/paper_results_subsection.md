# Paper-ready results subsection draft

## Results

The frozen FPGAI paper matrix contains 10 inference design rows and 9 on-device-training design rows. All reported HLS and Vivado values in this section are generated from existing synthesis and implementation artifacts; board-runtime latency, energy, and training-curve measurements are intentionally excluded until physical KV260 runtime artifacts are imported.

HLS metrics are available for 19 design rows and Vivado implementation metrics are available for 19 design rows. Numeric-validation rows are available for 19 design rows, with 10 row(s) currently marked as passed. The current matrix has 0 row(s) with imported runtime measurements, so runtime performance claims remain pending.

### Numeric behavior validation

- Inference numeric validation: 1/10 row(s) passed or contain reference-output comparison evidence.
- Training numeric validation: 9/9 row(s) passed or contain gradient/weight-update comparison evidence.
- **T0_sgd_tiled_m_axi**: gradient cosine=1, weight-delta cosine=0.9573, status=`passed`.
- **T1_momentum_tiled_m_axi**: gradient cosine=1, weight-delta cosine=0.9573, status=`passed`.
- **T2_adam_tiled_m_axi**: gradient cosine=1, weight-delta cosine=0.6845, status=`passed`.
- **T3_cross_entropy_tiled_m_axi**: gradient cosine=1, weight-delta cosine=0.8835, status=`passed`.
- **T4_tile32_m_axi**: gradient cosine=1, weight-delta cosine=0.9573, status=`passed`.

### Task-aware decision reporting

- **I0_baseline_fx16_embedded**: decision=`recommended_quality`, quality metric `prediction_agreement`=1, LUT saving vs fx16=0%, BRAM saving vs fx16=0%.
- **I1_precision_fx8_embedded**: decision=`not_recommended_for_quality`, quality metric `prediction_agreement`=0, LUT saving vs fx16=36.9%, BRAM saving vs fx16=66.4%.
- **I2_precision_fx24_embedded**: decision=`recommended_quality`, quality metric `prediction_agreement`=1, LUT saving vs fx16=-32.21%, BRAM saving vs fx16=-34.4%.
These decision labels are not a hard pass/fail gate: they summarize the behavioral cost of a YAML precision choice together with HLS/Vivado latency/resource effects. When labels or regression targets are provided, FPGAI reports task accuracy or target-error deltas; otherwise it reports generated-vs-reference prediction/output agreement.

### Inference design effects

- **precision_fx8_vs_fx16** (`strong_result`): HLS latency cycles: I1_precision_fx8_embedded increased by 0.51% versus I0_baseline_fx16_embedded (delta 1673).
- **precision_fx24_vs_fx16** (`expected_tradeoff`): HLS latency cycles: I2_precision_fx24_embedded changed by 0.00% versus I0_baseline_fx16_embedded (delta 13).
- **parallel_pe2_vs_pe1** (`expected_tradeoff`): HLS latency cycles: I3_parallel_pe2 decreased by 29.67% versus I0_baseline_fx16_embedded (delta -9.77e+04).
- **parallel_pe4_vs_pe1** (`expected_tradeoff`): HLS latency cycles: I4_parallel_pe4 decreased by 47.25% versus I0_baseline_fx16_embedded (delta -1.556e+05).
- **parallel_pe4_vs_pe2** (`expected_tradeoff`): HLS latency cycles: I4_parallel_pe4 decreased by 25.00% versus I3_parallel_pe2 (delta -5.788e+04).
- **weight_import_m_axi_vs_embedded** (`expected_tradeoff`): HLS latency cycles: I7_weight_import_m_axi decreased by 30.48% versus I0_baseline_fx16_embedded (delta -1.004e+05).
- **pipeline_latency_first_vs_baseline** (`expected_tradeoff`): HLS latency cycles: I5_pipeline_latency_first increased by 3.99% versus I0_baseline_fx16_embedded (delta 1.315e+04).
- **pipeline_resource_first_vs_baseline** (`no_observable_effect`): HLS latency cycles: I6_pipeline_resource_first changed by 0.00% versus I0_baseline_fx16_embedded (delta 0).

### Training design effects

- **training_momentum_vs_sgd** (`no_observable_effect`): HLS latency cycles: T1_momentum_tiled_m_axi changed by 0.00% versus T0_sgd_tiled_m_axi (delta 0).
- **training_adam_vs_sgd** (`expected_tradeoff`): HLS latency cycles: T2_adam_tiled_m_axi changed by 0.00% versus T0_sgd_tiled_m_axi (delta 0).
- **training_cross_entropy_vs_mse** (`expected_tradeoff`): HLS latency cycles: T3_cross_entropy_tiled_m_axi increased by 2.63% versus T0_sgd_tiled_m_axi (delta 25).
- **training_tile32_vs_tile64** (`strong_result`): HLS latency cycles: T4_tile32_m_axi decreased by 13.46% versus T0_sgd_tiled_m_axi (delta -128).
- **training_tile128_vs_tile64** (`expected_tradeoff`): HLS latency cycles: T5_tile128_m_axi increased by 26.92% versus T0_sgd_tiled_m_axi (delta 256).
- **training_accum_batch2_vs_sgd** (`no_observable_effect`): HLS latency cycles: T6_accum_batch2_m_axi changed by 0.00% versus T0_sgd_tiled_m_axi (delta 0).
- **training_bitstream_vs_sgd** (`deployability_result`): HLS latency cycles: T7_deployable_training_bitstream changed by 0.00% versus T0_sgd_tiled_m_axi (delta 0).

### Claim boundary

The generated comparisons support HLS/Vivado artifact claims, deployability/package claims, and explicit no-observable-effect classifications. They do not yet support measured FPGA latency, measured energy, or measured training convergence claims. Those claims require board-runtime CSV/report imports and should remain in the pending-measurement tables.

### Classification summary

- `strong_result`: 2 comparison(s) covering 18 metric row(s).
- `expected_tradeoff`: 9 comparison(s) covering 81 metric row(s).
- `deployability_result`: 1 comparison(s) covering 9 metric row(s).
- `no_observable_effect`: 3 comparison(s) covering 27 metric row(s).
