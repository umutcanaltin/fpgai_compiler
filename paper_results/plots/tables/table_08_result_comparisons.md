# Computed result comparisons

| comparison | group | baseline_design | variant_design | metric_label | Baseline | Variant | delta | Change % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| precision_fx8_vs_fx16 | Precision | I0_baseline_fx16_embedded | I1_precision_fx8_embedded | HLS latency cycles | 329247 | 330920 | 1673 | 0.51 |
| precision_fx8_vs_fx16 | Precision | I0_baseline_fx16_embedded | I1_precision_fx8_embedded | HLS LUT | 19572 | 18260 | -1312 | -6.70 |
| precision_fx8_vs_fx16 | Precision | I0_baseline_fx16_embedded | I1_precision_fx8_embedded | HLS DSP | 7 | 4 | -3 | -42.86 |
| precision_fx8_vs_fx16 | Precision | I0_baseline_fx16_embedded | I1_precision_fx8_embedded | HLS BRAM | 115 | 46 | -69 | -60.00 |
| precision_fx8_vs_fx16 | Precision | I0_baseline_fx16_embedded | I1_precision_fx8_embedded | Vivado LUT | 12708 | 8019 | -4689 | -36.90 |
| precision_fx8_vs_fx16 | Precision | I0_baseline_fx16_embedded | I1_precision_fx8_embedded | Vivado DSP | 5 | 4 | -1 | -20.00 |
| precision_fx8_vs_fx16 | Precision | I0_baseline_fx16_embedded | I1_precision_fx8_embedded | Vivado BRAM | 62.5 | 21 | -41.5 | -66.40 |
| precision_fx8_vs_fx16 | Precision | I0_baseline_fx16_embedded | I1_precision_fx8_embedded | Power W | 2.842 | 2.786 | -0.056 | -1.97 |
| precision_fx8_vs_fx16 | Precision | I0_baseline_fx16_embedded | I1_precision_fx8_embedded | WNS ns | 2.579 | 1.854 | -0.725 | -28.11 |
| parallel_pe2_vs_pe1 | Parallelism | I0_baseline_fx16_embedded | I3_parallel_pe2 | HLS latency cycles | 329247 | 231550 | -9.77e+04 | -29.67 |
| parallel_pe2_vs_pe1 | Parallelism | I0_baseline_fx16_embedded | I3_parallel_pe2 | HLS LUT | 19572 | 37430 | 1.786e+04 | 91.24 |
| parallel_pe2_vs_pe1 | Parallelism | I0_baseline_fx16_embedded | I3_parallel_pe2 | HLS DSP | 7 | 18 | 11 | 157.14 |
| parallel_pe2_vs_pe1 | Parallelism | I0_baseline_fx16_embedded | I3_parallel_pe2 | HLS BRAM | 115 | 140 | 25 | 21.74 |
| parallel_pe2_vs_pe1 | Parallelism | I0_baseline_fx16_embedded | I3_parallel_pe2 | Vivado LUT | 12708 | 39702 | 2.699e+04 | 212.42 |
| parallel_pe2_vs_pe1 | Parallelism | I0_baseline_fx16_embedded | I3_parallel_pe2 | Vivado DSP | 5 | 12 | 7 | 140.00 |
| parallel_pe2_vs_pe1 | Parallelism | I0_baseline_fx16_embedded | I3_parallel_pe2 | Vivado BRAM | 62.5 | 71 | 8.5 | 13.60 |
| parallel_pe2_vs_pe1 | Parallelism | I0_baseline_fx16_embedded | I3_parallel_pe2 | Power W | 2.842 | 2.924 | 0.082 | 2.89 |
| parallel_pe2_vs_pe1 | Parallelism | I0_baseline_fx16_embedded | I3_parallel_pe2 | WNS ns | 2.579 | 0.614 | -1.965 | -76.19 |
| deployable_inference_vs_baseline | Deployability | I0_baseline_fx16_embedded | I8_deployable_bitstream_candidate | HLS latency cycles | 329247 | 329247 | 0 | 0.00 |
| deployable_inference_vs_baseline | Deployability | I0_baseline_fx16_embedded | I8_deployable_bitstream_candidate | HLS LUT | 19572 | 19572 | 0 | 0.00 |
| deployable_inference_vs_baseline | Deployability | I0_baseline_fx16_embedded | I8_deployable_bitstream_candidate | HLS DSP | 7 | 7 | 0 | 0.00 |
| deployable_inference_vs_baseline | Deployability | I0_baseline_fx16_embedded | I8_deployable_bitstream_candidate | HLS BRAM | 115 | 115 | 0 | 0.00 |
| deployable_inference_vs_baseline | Deployability | I0_baseline_fx16_embedded | I8_deployable_bitstream_candidate | Vivado LUT | 12708 | 12708 | 0 | 0.00 |
| deployable_inference_vs_baseline | Deployability | I0_baseline_fx16_embedded | I8_deployable_bitstream_candidate | Vivado DSP | 5 | 5 | 0 | 0.00 |
| deployable_inference_vs_baseline | Deployability | I0_baseline_fx16_embedded | I8_deployable_bitstream_candidate | Vivado BRAM | 62.5 | 62.5 | 0 | 0.00 |
| deployable_inference_vs_baseline | Deployability | I0_baseline_fx16_embedded | I8_deployable_bitstream_candidate | Power W | 2.842 | 2.842 | 0 | 0.00 |
| deployable_inference_vs_baseline | Deployability | I0_baseline_fx16_embedded | I8_deployable_bitstream_candidate | WNS ns | 2.579 | 2.579 | 0 | 0.00 |
| training_tile32_vs_sgd | Training memory | T0_sgd_tiled_m_axi | T4_tile32_m_axi | HLS latency cycles | 951 | 823 | -128 | -13.46 |
| training_tile32_vs_sgd | Training memory | T0_sgd_tiled_m_axi | T4_tile32_m_axi | HLS LUT | 61934 | 61858 | -76 | -0.12 |
| training_tile32_vs_sgd | Training memory | T0_sgd_tiled_m_axi | T4_tile32_m_axi | HLS DSP | 21 | 21 | 0 | 0.00 |
| training_tile32_vs_sgd | Training memory | T0_sgd_tiled_m_axi | T4_tile32_m_axi | HLS BRAM | 44 | 44 | 0 | 0.00 |
| training_tile32_vs_sgd | Training memory | T0_sgd_tiled_m_axi | T4_tile32_m_axi | Vivado LUT | 29370 | 28989 | -381 | -1.30 |
| training_tile32_vs_sgd | Training memory | T0_sgd_tiled_m_axi | T4_tile32_m_axi | Vivado DSP | 15 | 15 | 0 | 0.00 |
| training_tile32_vs_sgd | Training memory | T0_sgd_tiled_m_axi | T4_tile32_m_axi | Vivado BRAM | 17.5 | 17.5 | 0 | 0.00 |
| training_tile32_vs_sgd | Training memory | T0_sgd_tiled_m_axi | T4_tile32_m_axi | Power W | 2.941 | 2.925 | -0.016 | -0.54 |
| training_tile32_vs_sgd | Training memory | T0_sgd_tiled_m_axi | T4_tile32_m_axi | WNS ns | 2.896 | 3.115 | 0.219 | 7.56 |
| training_bitstream_vs_sgd | Training deployability | T0_sgd_tiled_m_axi | T7_deployable_training_bitstream | HLS latency cycles | 951 | 951 | 0 | 0.00 |
| training_bitstream_vs_sgd | Training deployability | T0_sgd_tiled_m_axi | T7_deployable_training_bitstream | HLS LUT | 61934 | 61934 | 0 | 0.00 |
| training_bitstream_vs_sgd | Training deployability | T0_sgd_tiled_m_axi | T7_deployable_training_bitstream | HLS DSP | 21 | 21 | 0 | 0.00 |
| training_bitstream_vs_sgd | Training deployability | T0_sgd_tiled_m_axi | T7_deployable_training_bitstream | HLS BRAM | 44 | 44 | 0 | 0.00 |
| training_bitstream_vs_sgd | Training deployability | T0_sgd_tiled_m_axi | T7_deployable_training_bitstream | Vivado LUT | 29370 | 29370 | 0 | 0.00 |
| training_bitstream_vs_sgd | Training deployability | T0_sgd_tiled_m_axi | T7_deployable_training_bitstream | Vivado DSP | 15 | 15 | 0 | 0.00 |
| training_bitstream_vs_sgd | Training deployability | T0_sgd_tiled_m_axi | T7_deployable_training_bitstream | Vivado BRAM | 17.5 | 17.5 | 0 | 0.00 |
| training_bitstream_vs_sgd | Training deployability | T0_sgd_tiled_m_axi | T7_deployable_training_bitstream | Power W | 2.941 | 2.941 | 0 | 0.00 |
| training_bitstream_vs_sgd | Training deployability | T0_sgd_tiled_m_axi | T7_deployable_training_bitstream | WNS ns | 2.896 | 2.896 | 0 | 0.00 |
