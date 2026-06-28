# Paper artifact index

| design | board | mode | group | prediction_available | hls_status | vivado_ok | vivado_failure_class | bitstream_exists | xsa_exists | paper_status |
|---|---|---|---|---|---|---|---|---|---|---|
| pynq_z2_baseline_safe_fx16 | pynq_z2 | inference | baseline | True | full_csynth | None |  | False | False | hls_only |
| kv260_baseline_safe_fx16 | kv260 | inference | baseline | True | full_csynth | True |  | True | True | vivado_impl_bitstream_ready |
| kr260_baseline_safe_fx16 | kr260 | inference | baseline | True | full_csynth | None |  | False | False | hls_only |
| kv260_precision_fx16_6 | kv260 | inference | precision | True | full_csynth | None |  | False | False | hls_only |
| kv260_precision_fx12_4 | kv260 | inference | precision | True | full_csynth | None |  | False | False | hls_only |
| kv260_precision_fx8_3 | kv260 | inference | precision | True | full_csynth | True |  | True | True | vivado_impl_bitstream_ready |
| kv260_parallel_x1 | kv260 | inference | parallelism | True | full_csynth | None |  | False | False | hls_only |
| kv260_parallel_x2 | kv260 | inference | parallelism | True | full_csynth | None |  | False | False | hls_only |
| kv260_parallel_x4 | kv260 | inference | parallelism | True | full_csynth | None |  | False | False | hls_only |
| kv260_parallel_x8 | kv260 | inference | parallelism | True | full_csynth | True |  | True | True | vivado_impl_bitstream_ready |
| kv260_pipeline_balanced_ii2 | kv260 | inference | pipeline | True | full_csynth | None |  | False | False | hls_only |
| kv260_pipeline_aggressive_ii1 | kv260 | inference | pipeline | True | full_csynth | None |  | False | False | hls_only |
| kv260_tiling_small | kv260 | inference | tiling | True | full_csynth | None |  | False | False | hls_only |
| kv260_tiling_medium | kv260 | inference | tiling | True | full_csynth | None |  | False | False | hls_only |
| kv260_tiling_large | kv260 | inference | tiling | True | full_csynth | None |  | False | False | hls_only |
| kv260_memory_bram | kv260 | inference | memory | True | full_csynth | None |  | False | False | hls_only |
| kv260_memory_uram | kv260 | inference | memory | True | full_csynth | True |  | True | True | vivado_impl_bitstream_ready |
| kv260_combined_aggressive_fx8 | kv260 | inference | combined | True | full_csynth | None |  | False | False | hls_only |
| training_kv260_safe_fx16_6 | kv260 | training_on_device | training | True | full_csynth | False | vivado_impl_failed_board_capacity_lut_overutilized | False | False | vivado_board_capacity_rejected |
| training_kv260_aggressive_fx8_3 | kv260 | training_on_device | training | True | full_csynth | None |  | False | False | hls_only |
