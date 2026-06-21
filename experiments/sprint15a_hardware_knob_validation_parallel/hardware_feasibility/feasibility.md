# FPGAI hardware feasibility classification

| design | status | reason | bitstream_exists | xsa_exists | power_w | wns_ns | lut | lut_util_pct | ff | bram | dsp | dsp_util_pct | energy_j |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hw_parallel_1 | pass | bitstream/xsa generated and timing closed or no timing violation parsed | True | True | 2.259 | 0.73 | 32768 | 61.59 | 43188 | 54 | 51 | 23.18 | 7.65801e-06 |
| hw_parallel_2 | pass | bitstream/xsa generated and timing closed or no timing violation parsed | True | True | 2.358 | 0.702 | 34024 | 63.95 | 43149 | 59 | 109 | 49.55 | 7.99362e-06 |
| hw_parallel_4 | resource_fail | LUT 65404>53200 | False | False | 2.678 | -4.729 | 65404 | 122.94 | 60668 | 73 | 220 | 100.0 | 9.07842e-06 |
| hw_prec_fx12_like | pass | bitstream/xsa generated and timing closed or no timing violation parsed | True | True | 2.165 | 0.716 | 25569 | 48.06 | 33699 | 44 | 70 | 31.82 | 7.33935e-06 |
| hw_prec_fx16_like | pass | bitstream/xsa generated and timing closed or no timing violation parsed | True | True | 2.323 | 0.52 | 32089 | 60.32 | 41304 | 49 | 127 | 57.73 | 7.87497e-06 |
| hw_prec_fx8_like | pass | bitstream/xsa generated and timing closed or no timing violation parsed | True | True | 2.094 | 1.269 | 21738 | 40.86 | 29374 | 35 | 64 | 29.09 | 7.09866e-06 |
