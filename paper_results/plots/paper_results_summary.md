# FPGAI paper results summary

- compile_output_count: `19`
- created_figures: `19`
- pending_figures: `4`

## Current paper story

The current frozen subset supports an inference-first and training-second results section. HLS/Vivado figures are generated from existing synthesis/implementation artifacts. Real latency, energy, and FPGA training-curve figures are intentionally pending until board-runtime measurements are imported.

## Numeric validation summary

- numeric_validation_rows: `19`
- numeric_validation_passed_rows: `10`
- Use `tables/table_10_numeric_validation_summary.md`, `table_11_inference_precision_numeric_tradeoff.md`, and `table_12_training_numeric_validation.md` for numeric behavior claims.
- Use `tables/table_14_inference_task_quality_tradeoff.md`, `table_15_precision_decision_matrix.md`, `table_16_training_task_quality_tradeoff.md`, and `table_17_training_precision_decision_matrix.md` for user-facing decision reports.

## Key computed comparisons
- `strong_result` — HLS latency cycles: I1_precision_fx8_embedded increased by 0.51% versus I0_baseline_fx16_embedded (delta 1673).
- `expected_tradeoff` — HLS latency cycles: I2_precision_fx24_embedded changed by 0.00% versus I0_baseline_fx16_embedded (delta 13).
- `expected_tradeoff` — HLS latency cycles: I3_parallel_pe2 decreased by 29.67% versus I0_baseline_fx16_embedded (delta -9.77e+04).
- `expected_tradeoff` — HLS latency cycles: I4_parallel_pe4 decreased by 47.25% versus I0_baseline_fx16_embedded (delta -1.556e+05).
- `expected_tradeoff` — HLS latency cycles: I4_parallel_pe4 decreased by 25.00% versus I3_parallel_pe2 (delta -5.788e+04).
- `expected_tradeoff` — HLS latency cycles: I7_weight_import_m_axi decreased by 30.48% versus I0_baseline_fx16_embedded (delta -1.004e+05).
- `expected_tradeoff` — HLS latency cycles: I5_pipeline_latency_first increased by 3.99% versus I0_baseline_fx16_embedded (delta 1.315e+04).
- `no_observable_effect` — HLS latency cycles: I6_pipeline_resource_first changed by 0.00% versus I0_baseline_fx16_embedded (delta 0).
- `no_observable_effect` — HLS latency cycles: T1_momentum_tiled_m_axi changed by 0.00% versus T0_sgd_tiled_m_axi (delta 0).
- `expected_tradeoff` — HLS latency cycles: T2_adam_tiled_m_axi changed by 0.00% versus T0_sgd_tiled_m_axi (delta 0).
- `expected_tradeoff` — HLS latency cycles: T3_cross_entropy_tiled_m_axi increased by 2.63% versus T0_sgd_tiled_m_axi (delta 25).
- `strong_result` — HLS latency cycles: T4_tile32_m_axi decreased by 13.46% versus T0_sgd_tiled_m_axi (delta -128).
- `expected_tradeoff` — HLS latency cycles: T5_tile128_m_axi increased by 26.92% versus T0_sgd_tiled_m_axi (delta 256).
- `no_observable_effect` — HLS latency cycles: T6_accum_batch2_m_axi changed by 0.00% versus T0_sgd_tiled_m_axi (delta 0).
- `deployability_result` — HLS latency cycles: T7_deployable_training_bitstream changed by 0.00% versus T0_sgd_tiled_m_axi (delta 0).

## Result classification summary
- `strong_result`: `2` comparison(s), `18` metric row(s).
- `expected_tradeoff`: `9` comparison(s), `81` metric row(s).
- `deployability_result`: `1` comparison(s), `9` metric row(s).
- `no_observable_effect`: `3` comparison(s), `27` metric row(s).

## Use in paper

- Use `figure_captions.md` and `table_captions.md` for caption drafting.
- Use `paper_claims_from_artifacts.md` to avoid over-claiming runtime behavior.
- Open `plot_gallery.html` locally to inspect all generated SVG plots in one page.
