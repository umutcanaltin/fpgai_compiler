# Design knob effects

| design | group | hls_lut | lut_vs_baseline | hls_dsp | dsp_vs_baseline | hls_latency_cycles | latency_vs_baseline | vivado_status |
|---|---|---|---|---|---|---|---|---|
| kv260_baseline_safe_fx16 | baseline | 5991 | 1.000 | 10 | 1.000 | 182.0 | 1.000 | vivado_impl_bitstream_ready |
| kv260_precision_fx16_6 | precision | 5056 | 0.844 | 24 | 2.400 | 90.0 | 0.495 | hls_only |
| kv260_precision_fx12_4 | precision | 4876 | 0.814 | 20 | 2.000 | 85.0 | 0.467 | hls_only |
| kv260_precision_fx8_3 | precision | 4119 | 0.688 | 8 | 0.800 | 72.0 | 0.396 | vivado_impl_bitstream_ready |
| kv260_parallel_x1 | parallelism | 4869 | 0.813 | 6 | 0.600 | 247.0 | 1.357 | hls_only |
| kv260_parallel_x2 | parallelism | 5991 | 1.000 | 10 | 1.000 | 182.0 | 1.000 | hls_only |
| kv260_parallel_x4 | parallelism | 5056 | 0.844 | 24 | 2.400 | 90.0 | 0.495 | hls_only |
| kv260_parallel_x8 | parallelism | 3584 | 0.598 | 35 | 3.500 | 80.0 | 0.440 | vivado_impl_bitstream_ready |
| kv260_pipeline_balanced_ii2 | pipeline | 5056 | 0.844 | 24 | 2.400 | 90.0 | 0.495 | hls_only |
| kv260_pipeline_aggressive_ii1 | pipeline | 5041 | 0.841 | 24 | 2.400 | 88.0 | 0.484 | hls_only |
| kv260_tiling_small | tiling | 3899 | 0.651 | 14 | 1.400 | 93.0 | 0.511 | hls_only |
| kv260_tiling_medium | tiling | 4275 | 0.714 | 22 | 2.200 | 86.0 | 0.473 | hls_only |
| kv260_tiling_large | tiling | 4305 | 0.719 | 24 | 2.400 | 81.0 | 0.445 | hls_only |
| kv260_memory_bram | memory | 5056 | 0.844 | 24 | 2.400 | 90.0 | 0.495 | hls_only |
| kv260_memory_uram | memory | 5056 | 0.844 | 24 | 2.400 | 90.0 | 0.495 | vivado_impl_bitstream_ready |
| kv260_combined_aggressive_fx8 | combined | 3472 | 0.580 | 7 | 0.700 | 59.0 | 0.324 | hls_only |
