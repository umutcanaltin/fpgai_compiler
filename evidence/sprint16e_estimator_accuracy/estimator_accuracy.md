# Sprint 16E Resource Estimator Accuracy

This table compares FPGAI resource predictions against Vivado implementation resources only when usable non-placeholder prediction artifacts are found. Values `0` and `1` are treated as placeholders/missing.

## Summary

- comparison rows: 17
- rows with any usable prediction: 13
- skipped rows without usable prediction: 4
- overall comparison count: 13
- overall mean absolute percentage error: 92.432%
- overall median absolute percentage error: 96.364%
- overall worst absolute percentage error: 96.61%

- LUT: count=0, mean_abs_pct_error=, median_abs_pct_error=, worst_abs_pct_error=
- FF: count=0, mean_abs_pct_error=, median_abs_pct_error=, worst_abs_pct_error=
- BRAM: count=13, mean_abs_pct_error=92.432, median_abs_pct_error=96.364, worst_abs_pct_error=96.61
- DSP: count=0, mean_abs_pct_error=, median_abs_pct_error=, worst_abs_pct_error=

## Estimator accuracy table

| design | status | usable_predictions | pred_lut | vivado_lut | lut_abs_error_pct | pred_ff | vivado_ff | ff_abs_error_pct | pred_bram | vivado_bram | bram_abs_error_pct | pred_dsp | vivado_dsp | dsp_abs_error_pct | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| training_cnn_stream_converge_2epoch_b2_balanced | pass | 1 |  | 34024 |  |  | 43149 |  | 2 | 59 | 96.61 |  | 109 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| training_cnn_stream_converge_3epoch_b2_balanced | pass | 1 |  | 34024 |  |  | 43149 |  | 2 | 59 | 96.61 |  | 109 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| training_cnn_stream_converge_3epoch_b4_balanced | pass | 1 |  | 34024 |  |  | 43149 |  | 2 | 59 | 96.61 |  | 109 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_parallel_1 | pass | 0 |  | 32768 |  |  | 43188 |  |  | 54 |  |  | 51 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;bram:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_parallel_2 | pass | 1 |  | 34024 |  |  | 43149 |  | 2 | 59 | 96.61 |  | 109 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_parallel_4 | timing_fail | 1 |  |  |  |  | 60668 |  | 10 | 73 | 86.301 |  | 220 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_prec_fx12_like | pass | 0 |  | 25569 |  |  | 33699 |  |  | 17 |  |  | 70 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;bram:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_prec_fx16_like | pass | 1 |  | 32089 |  |  | 41304 |  | 2 | 49 | 95.918 |  | 127 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_prec_fx8_like | pass | 0 |  | 21738 |  |  | 29374 |  |  | 35 |  |  | 64 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;bram:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_pipeline_aggressive_strict | pass | 1 |  | 34024 |  |  | 43149 |  | 2 | 59 | 96.61 |  | 109 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_pipeline_balanced_strict | pass | 1 |  | 34935 |  |  | 46048 |  | 2 | 59 | 96.61 |  | 93 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_pipeline_conservative_strict | pass | 1 |  | 34085 |  |  | 45473 |  | 2 | 55 | 96.364 |  | 72 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_parallel_1_feasible | pass | 0 |  | 29199 |  |  | 20834 |  |  | 9 |  |  | 53 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;bram:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_parallel_2_feasible | timing_fail | 1 |  | 31808 |  |  | 22415 |  | 2 | 15 | 86.667 |  | 96 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_parallel_2_relaxed_180mhz | timing_fail | 1 |  | 31701 |  |  | 22415 |  | 2 | 15 | 86.667 |  | 96 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_parallel_3_candidate | timing_fail | 1 |  |  |  |  | 40572 |  | 10 | 63 | 84.127 |  | 172 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |
| hw_parallel_4_expected_resource_fail | timing_fail | 1 |  |  |  |  | 29912 |  | 10 | 71 | 85.915 |  | 220 |  | lut:placeholder_prediction_1;ff:placeholder_prediction_1;dsp:placeholder_prediction_1 |

## Safe claim

FPGAI estimator outputs are compared against Vivado implementation resources where usable non-placeholder prediction artifacts are available.

## Limitation

This collector summarizes existing estimator artifacts only. It does not rerun estimation. Placeholder values such as 0 or 1 are treated as missing to avoid artificial error tables. Candidate files are listed in `estimator_candidate_files.txt`.
