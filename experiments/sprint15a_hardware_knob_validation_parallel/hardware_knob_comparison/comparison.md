# FPGAI hardware knob comparison

| design | axis | precision | parallel | pipeline | reports | bit | xsa | power_w | wns_ns | lut | ff | bram | dsp | hls_cycles | energy_j |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hw_prec_fx12_like | precision | fx12_like |  |  | True | True | True | 2.165 | 0.716 | 25569 | 33699 | 44 | 70 |  | 7.33935e-06 |
| hw_prec_fx16_like | precision | fx16_like |  |  | True | True | True | 2.323 | 0.52 | 32089 | 41304 | 49 | 127 |  | 7.87497e-06 |
| hw_prec_fx8_like | precision | fx8_like |  |  | True | True | True | 2.094 | 1.269 | 21738 | 29374 | 35 | 64 |  | 7.09866e-06 |
| hw_parallel_1 | parallel |  | 1 |  | True | True | True | 2.259 | 0.73 | 32768 | 43188 | 54 | 51 |  | 7.65801e-06 |
| hw_parallel_2 | parallel |  | 2 |  | True | True | True | 2.358 | 0.702 | 34024 | 43149 | 59 | 109 |  | 7.99362e-06 |
| hw_parallel_4 | parallel |  | 4 |  | True |  |  | 2.678 | -4.729 | 65404 | 60668 | 73 | 220 |  | 9.07842e-06 |
