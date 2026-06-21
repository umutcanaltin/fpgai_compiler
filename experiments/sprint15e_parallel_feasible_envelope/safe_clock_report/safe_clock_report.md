# FPGAI safe clock recommendation report

| design | status | target_clock_mhz | wns_ns | estimated_fmax_mhz | recommended_safe_clock_mhz | lut | lut_util_pct | dsp | dsp_util_pct | bitstream_exists | xsa_exists | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hw_parallel_1_feasible | pass | 179.986 | 0.253 | 188.573 | 171.43 | 29199 | 54.885 | 53 | 24.091 | True | True | Timing closes at target clock; recommended clock includes safety margin. |
| hw_parallel_2_feasible | timing_fail | 179.986 | -0.47 | 165.948 | 150.861 | 31808 | 59.789 | 96 | 43.636 | True | True | Bitstream/XSA generated but timing is negative; use lower clock or safer policy. |
| hw_parallel_2_relaxed_180mhz | timing_fail | 179.986 | -0.124 | 176.056 | 160.051 | 31701 | 59.588 | 96 | 43.636 | True | True | Bitstream/XSA generated but timing is negative; use lower clock or safer policy. |
| hw_parallel_3_candidate | resource_fail | 179.986 | -9.347 | 67.101 | 61.001 | 100425 | 188.769 | 172 | 78.182 | False | False | Does not fit this FPGA resource envelope; select lower parallelism or larger FPGA. |
| hw_parallel_4_expected_resource_fail | resource_fail | 179.986 | -9.123 | 68.125 | 61.931 | 70452 | 132.429 | 220 | 100 | False | False | Does not fit this FPGA resource envelope; select lower parallelism or larger FPGA. |
