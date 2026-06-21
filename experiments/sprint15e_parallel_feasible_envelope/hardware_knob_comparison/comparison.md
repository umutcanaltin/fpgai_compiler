# FPGAI hardware knob comparison

| design | axis | precision | parallel | pipeline | reports | bit | xsa | power_w | wns_ns | lut | ff | bram | dsp | hls_cycles | energy_j |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hw_parallel_1_feasible | parallel |  | 1 |  | True | True | True | 2.184 | 0.253 | 29199 | 20834 | 52 | 53 |  | 7.40376e-06 |
| hw_parallel_2_feasible | parallel |  | 2 |  | True | True | True | 2.291 | -0.47 | 31808 | 22415 | 58 | 96 |  | 7.76649e-06 |
| hw_parallel_2_relaxed_180mhz | parallel |  | 2 |  | True | True | True | 2.289 | -0.124 | 31701 | 22415 | 58 | 96 |  | 7.75971e-06 |
| hw_parallel_3_candidate | parallel |  | 3 |  | True |  |  | 3.552 | -9.347 | 100425 | 40572 | 63 | 172 |  | 1.20413e-05 |
| hw_parallel_4_expected_resource_fail | parallel |  | 4 |  | True |  |  | 2.397 | -9.123 | 70452 | 29912 | 71 | 220 |  | 8.12583e-06 |
