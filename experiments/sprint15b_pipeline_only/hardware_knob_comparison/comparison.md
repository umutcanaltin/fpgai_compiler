# FPGAI hardware knob comparison

| design | axis | precision | parallel | pipeline | reports | bit | xsa | power_w | wns_ns | lut | ff | bram | dsp | hls_cycles | energy_j |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hw_pipeline_aggressive_only | pipeline |  |  | aggressive | True |  |  | 1.661 | -2.429 | 30013 | 39661 | 56 | 109 |  | 5.63079e-06 |
| hw_pipeline_balanced_only | pipeline |  |  | balanced | True |  |  | 1.661 | -2.429 | 30013 | 39661 | 56 | 109 |  | 5.63079e-06 |
| hw_pipeline_conservative_only | pipeline |  |  | conservative | True |  |  | 1.62 | -1.175 | 30108 | 41924 | 52 | 81 |  | 5.4918e-06 |
