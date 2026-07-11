# HLS results

| section | Design | hls_status | HLS latency cycles | HLS LUT | HLS DSP | HLS BRAM | hls_ii | hls_clock_period_ns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inference | I0_baseline_fx16_embedded | parsed | 329247 | 19572 | 7 | 115 | — | — |
| inference | I1_precision_fx8_embedded | parsed | 330920 | 18260 | 4 | 46 | — | — |
| inference | I2_precision_fx24_embedded | parsed | 329260 | 21445 | 13 | 151 | — | — |
| inference | I3_parallel_pe2 | parsed | 231550 | 37430 | 18 | 140 | — | — |
| inference | I4_parallel_pe4 | parsed | 173667 | 47704 | 66 | 180 | — | — |
| inference | I5_pipeline_latency_first | parsed | 342397 | 37334 | 11 | 126 | — | — |
| inference | I6_pipeline_resource_first | parsed | 329247 | 19572 | 7 | 115 | — | — |
| inference | I7_weight_import_m_axi | parsed | 228893 | 30234 | 7 | 120 | — | — |
| inference | I8_deployable_bitstream_candidate | parsed | 329247 | 19572 | 7 | 115 | — | — |
| inference | I9_board_runtime_candidate | parsed | 329247 | 19572 | 7 | 115 | — | — |
| training | T0_sgd_tiled_m_axi | failed | 951 | 61934 | 21 | 44 | — | — |
| training | T1_momentum_tiled_m_axi | failed | 951 | 61934 | 21 | 44 | — | — |
| training | T3_cross_entropy_tiled_m_axi | failed | 976 | 72136 | 53 | 44 | — | — |
| training | T4_tile32_m_axi | failed | 823 | 61858 | 21 | 44 | — | — |
| training | T5_tile128_m_axi | failed | 1207 | 61963 | 21 | 44 | — | — |
| training | T6_accum_batch2_m_axi | failed | 951 | 61934 | 21 | 44 | — | — |
| training | T7_deployable_training_bitstream | failed | 951 | 61934 | 21 | 44 | — | — |
| training | T8_real_fpga_training_curve_candidate | failed | 951 | 61934 | 21 | 44 | — | — |
