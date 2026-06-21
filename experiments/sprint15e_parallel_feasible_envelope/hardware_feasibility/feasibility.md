# FPGAI hardware feasibility classification

| design | status | reason | bitstream_exists | xsa_exists | power_w | wns_ns | lut | lut_util_pct | ff | bram | dsp | dsp_util_pct | energy_j |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hw_parallel_1_feasible | pass | bitstream/xsa generated and timing closed or no timing violation parsed | True | True | 2.184 | 0.253 | 29199 | 54.89 | 20834 | 52 | 53 | 24.09 | 7.403760000000001e-06 |
| hw_parallel_2_feasible | timing_fail | negative WNS -0.47 ns | True | True | 2.291 | -0.47 | 31808 | 59.79 | 22415 | 58 | 96 | 43.64 | 7.76649e-06 |
| hw_parallel_2_relaxed_180mhz | timing_fail | negative WNS -0.124 ns | True | True | 2.289 | -0.124 | 31701 | 59.59 | 22415 | 58 | 96 | 43.64 | 7.75971e-06 |
| hw_parallel_3_candidate | resource_fail | LUT 100425>53200 | False | False | 3.552 | -9.347 | 100425 | 188.77 | 40572 | 63 | 172 | 78.18 | 1.2041280000000001e-05 |
| hw_parallel_4_expected_resource_fail | resource_fail | LUT 70452>53200 | False | False | 2.397 | -9.123 | 70452 | 132.43 | 29912 | 71 | 220 | 100.0 | 8.12583e-06 |
