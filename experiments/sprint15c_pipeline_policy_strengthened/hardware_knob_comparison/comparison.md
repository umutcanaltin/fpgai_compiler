# FPGAI hardware knob comparison

| design | axis | precision | parallel | pipeline | reports | bit | xsa | power_w | wns_ns | lut | ff | bram | dsp | hls_cycles | energy_j |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hw_pipeline_aggressive_strict | pipeline |  |  | aggressive | True | True | True | 2.358 | 0.702 | 34024 | 43149 | 59 | 109 |  | 7.99362e-06 |
| hw_pipeline_balanced_strict | pipeline |  |  | balanced | True | True | True | 2.333 | 0.773 | 34935 | 46048 | 59 | 93 |  | 7.90887e-06 |
| hw_pipeline_conservative_strict | pipeline |  |  | conservative | True | True | True | 2.278 | 0.945 | 34085 | 45473 | 55 | 72 |  | 7.72242e-06 |
