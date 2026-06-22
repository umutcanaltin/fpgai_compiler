# Sprint 16D Hardware Knob Evidence

This file summarizes paper-facing hardware knob tables generated from Sprint 16C Vivado evidence.

## Summary

- precision: rows=3, pass_rows=3, timing_fail_rows=0, bitstream_rows=3, complete_resource_rows=3
- pipeline: rows=3, pass_rows=3, timing_fail_rows=0, bitstream_rows=3, complete_resource_rows=3
- parallel: rows=8, pass_rows=3, timing_fail_rows=5, bitstream_rows=5, complete_resource_rows=5

## Precision table

# Sprint 16D Precision Hardware Table

Precision rows are selected from Sprint 16C Vivado implementation evidence.

Rows: 3

| precision_mode | design | status | bitstream | xsa | wns_ns | fmax_mhz | safe_clock_mhz | power_w | lut | ff | bram | dsp | notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fx8_like | hw_prec_fx8_like | pass | True | True | 1.269 | 268.025 | 243.634 | 2.094 | 21738 | 29374 | 35 | 64 |  |
| fx12_like | hw_prec_fx12_like | pass | True | True | 0.716 | 233.427 | 212.185 | 2.165 | 25569 | 33699 | 17 | 70 |  |
| fx16_like | hw_prec_fx16_like | pass | True | True | 0.52 | 223.214 | 202.902 | 2.323 | 32089 | 41304 | 49 | 127 |  |


## Pipeline table

# Sprint 16D Pipeline Policy Hardware Table

Pipeline rows are selected from Sprint 16C Vivado implementation evidence.

Rows: 3

| pipeline_policy | design | status | bitstream | xsa | wns_ns | fmax_mhz | safe_clock_mhz | power_w | lut | ff | bram | dsp | notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| conservative | hw_pipeline_conservative_strict | pass | True | True | 0.945 | 246.609 | 224.168 | 2.278 | 34085 | 45473 | 55 | 72 |  |
| balanced | hw_pipeline_balanced_strict | pass | True | True | 0.773 | 236.574 | 215.046 | 2.333 | 34935 | 46048 | 59 | 93 |  |
| aggressive | hw_pipeline_aggressive_strict | pass | True | True | 0.702 | 232.666 | 211.494 | 2.358 | 34024 | 43149 | 59 | 109 |  |


## Parallel feasibility envelope

# Sprint 16D Parallel Feasibility Envelope

Parallel rows are selected from Sprint 16C Vivado implementation evidence.

Rows: 8

| parallel_factor | design | status | bitstream | xsa | wns_ns | fmax_mhz | safe_clock_mhz | power_w | lut | ff | bram | dsp | notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | hw_parallel_1 | pass | True | True | 0.73 | 234.192 | 212.881 | 2.259 | 32768 | 43188 | 54 | 51 |  |
| 1 | hw_parallel_1_feasible | pass | True | True | 0.253 | 210.659 | 191.489 | 2.184 | 29199 | 20834 | 9 | 53 |  |
| 2 | hw_parallel_2 | pass | True | True | 0.702 | 232.666 | 211.494 | 2.358 | 34024 | 43149 | 59 | 109 |  |
| 2 | hw_parallel_2_feasible | timing_fail | True | True | -0.47 | 182.815 | 166.179 | 2.291 | 31808 | 22415 | 15 | 96 |  |
| 2 | hw_parallel_2_relaxed_180mhz | timing_fail | True | True | -0.124 | 195.16 | 177.4 | 2.289 | 31701 | 22415 | 15 | 96 |  |
| 3 | hw_parallel_3_candidate | timing_fail | False | False | -9.347 | 69.701 | 63.358 | 3.552 |  | 40572 | 63 | 172 | missing: lut |
| 4 | hw_parallel_4 | timing_fail | False | False | -4.729 | 102.785 | 93.432 | 2.678 |  | 60668 | 73 | 220 | missing: lut |
| 4 | hw_parallel_4_expected_resource_fail | timing_fail | False | False | -9.123 | 70.806 | 64.363 | 2.397 |  | 29912 | 71 | 220 | missing: lut |


## Safe claim

FPGAI hardware knobs produce measurable implementation-level differences in resources, timing, and power for evaluated designs.

## Limitations

- Tables are generated from existing Vivado evidence and do not rerun Vivado.
- Claims are limited to evaluated design points and target flow.
- FPGAI does not claim global design-space optimality.
