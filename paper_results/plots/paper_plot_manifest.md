# FPGAI paper plot manifest

- status: `created`
- compile_output_count: `19`

## Sections
### inference
- design_count: `10`
- real_runtime_rows: `0`
- numeric_validation_rows: `10`
- numeric_passed_rows: `1`
- required_real_measurements: `["runtime_latency_ms_mean", "runtime_throughput", "runtime_accuracy", "vivado_power_w", "runtime_energy_mj_estimated"]`

### training
- design_count: `9`
- real_runtime_rows: `0`
- real_training_curve_rows: `0`
- numeric_validation_rows: `9`
- numeric_passed_rows: `9`
- required_real_measurements: `["runtime_training_step_ms", "runtime_loss_before", "runtime_loss_after", "board_training_curve.csv"]`

## Created figures
- `figure_00_plot_status`: `paper_results/plots/figures/figure_00_plot_status.svg`
- `figure_01_inference_hls_latency`: `paper_results/plots/figures/figure_01_inference_hls_latency.svg`
- `figure_02_inference_vivado_lut`: `paper_results/plots/figures/figure_02_inference_vivado_lut.svg`
- `figure_05_training_hls_latency`: `paper_results/plots/figures/figure_05_training_hls_latency.svg`
- `figure_06_training_vivado_lut`: `paper_results/plots/figures/figure_06_training_vivado_lut.svg`
- `figure_09_knob_status_counts`: `paper_results/plots/figures/figure_09_knob_status_counts.svg`
- `figure_10_training_vivado_power`: `paper_results/plots/figures/figure_10_training_vivado_power.svg`
- `figure_11_training_vivado_dsp`: `paper_results/plots/figures/figure_11_training_vivado_dsp.svg`
- `figure_12_all_hls_lut`: `paper_results/plots/figures/figure_12_all_hls_lut.svg`
- `figure_13_all_vivado_power`: `paper_results/plots/figures/figure_13_all_vivado_power.svg`
- `figure_14_all_vivado_wns`: `paper_results/plots/figures/figure_14_all_vivado_wns.svg`
- `figure_15_precision_numeric_error`: `paper_results/plots/figures/figure_15_precision_numeric_error.svg`
- `figure_16_precision_resource_numeric_tradeoff`: `paper_results/plots/figures/figure_16_precision_resource_numeric_tradeoff.svg`
- `figure_17_training_gradient_fidelity`: `paper_results/plots/figures/figure_17_training_gradient_fidelity.svg`
- `figure_18_inference_quality_vs_lut`: `paper_results/plots/figures/figure_18_inference_quality_vs_lut.svg`
- `figure_19_precision_decision_lut_saving`: `paper_results/plots/figures/figure_19_precision_decision_lut_saving.svg`
- `figure_20_error_vs_bram`: `paper_results/plots/figures/figure_20_error_vs_bram.svg`
- `figure_21_training_loss_delta`: `paper_results/plots/figures/figure_21_training_loss_delta.svg`
- `figure_22_training_update_vs_lut`: `paper_results/plots/figures/figure_22_training_update_vs_lut.svg`

## Pending figures
- `figure_03_inference_real_latency_ms`: pending real inference board-runtime measurements
- `figure_04_inference_energy_mj`: pending real inference latency/power measurements
- `figure_07_training_step_ms`: pending real training board-runtime measurements
- `figure_08_training_curve_loss`: pending real FPGA training-curve rows

## Paper tables
- `table_01_experiment_overview`: `paper_results/plots/tables/table_01_experiment_overview.md` rows=`19`
- `table_02_design_knobs`: `paper_results/plots/tables/table_02_design_knobs.md` rows=`343`
- `table_03_hls_results`: `paper_results/plots/tables/table_03_hls_results.md` rows=`19`
- `table_04_vivado_results`: `paper_results/plots/tables/table_04_vivado_results.md` rows=`19`
- `table_05_runtime_status`: `paper_results/plots/tables/table_05_runtime_status.md` rows=`19`
- `table_06_pending_measurements`: `paper_results/plots/tables/table_06_pending_measurements.md` rows=`76`
- `table_07_knob_effect_summary`: `paper_results/plots/tables/table_07_knob_effect_summary.md` rows=`35`
- `table_08_result_comparisons`: `paper_results/plots/tables/table_08_result_comparisons.md` rows=`135`
- `table_09_result_classification_summary`: `paper_results/plots/tables/table_09_result_classification_summary.md` rows=`4`
- `table_10_numeric_validation_summary`: `paper_results/plots/tables/table_10_numeric_validation_summary.md` rows=`19`
- `table_11_inference_precision_numeric_tradeoff`: `paper_results/plots/tables/table_11_inference_precision_numeric_tradeoff.md` rows=`3`
- `table_12_training_numeric_validation`: `paper_results/plots/tables/table_12_training_numeric_validation.md` rows=`9`
- `table_13_training_precision_numeric_tradeoff`: `paper_results/plots/tables/table_13_training_precision_numeric_tradeoff.md` rows=`9`
- `table_14_inference_task_quality_tradeoff`: `paper_results/plots/tables/table_14_inference_task_quality_tradeoff.md` rows=`10`
- `table_15_precision_decision_matrix`: `paper_results/plots/tables/table_15_precision_decision_matrix.md` rows=`3`
- `table_16_training_task_quality_tradeoff`: `paper_results/plots/tables/table_16_training_task_quality_tradeoff.md` rows=`9`
- `table_17_training_precision_decision_matrix`: `paper_results/plots/tables/table_17_training_precision_decision_matrix.md` rows=`9`

## Narrative files
- `paper_results_summary_md`: `paper_results/plots/paper_results_summary.md`
- `figure_captions_md`: `paper_results/plots/figure_captions.md`
- `table_captions_md`: `paper_results/plots/table_captions.md`
- `paper_claims_from_artifacts_md`: `paper_results/plots/paper_claims_from_artifacts.md`
- `plot_gallery_md`: `paper_results/plots/plot_gallery.md`
- `plot_gallery_html`: `paper_results/plots/plot_gallery.html`
- `paper_results_subsection_md`: `paper_results/plots/paper_results_subsection.md`

Plots are generated only from existing report/runtime artifacts. Missing board-runtime latency, energy, and training-curve measurements are marked pending, not fabricated.
