# Sprint 16D Pipeline Policy Hardware Table

Pipeline rows are selected from Sprint 16C Vivado implementation evidence.

Rows: 3

| pipeline_policy | design | status | bitstream | xsa | wns_ns | fmax_mhz | safe_clock_mhz | power_w | lut | ff | bram | dsp | notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| conservative | hw_pipeline_conservative_strict | pass | True | True | 0.945 | 246.609 | 224.168 | 2.278 | 34085 | 45473 | 55 | 72 |  |
| balanced | hw_pipeline_balanced_strict | pass | True | True | 0.773 | 236.574 | 215.046 | 2.333 | 34935 | 46048 | 59 | 93 |  |
| aggressive | hw_pipeline_aggressive_strict | pass | True | True | 0.702 | 232.666 | 211.494 | 2.358 | 34024 | 43149 | 59 | 109 |  |
