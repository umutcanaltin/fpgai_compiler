# Experiment overview

| section | Design | mode | board | precision | memory_strategy | pipeline_policy | hls_status | vivado_status | runtime_status | support_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inference | I0_baseline_fx16_embedded | inference | kv260 | fx16_6 | — | — | parsed | passed | not_run | vivado_implementation_available |
| inference | I1_precision_fx8_embedded | inference | kv260 | fx8_3 | — | — | parsed | passed | not_run | vivado_implementation_available |
| inference | I2_precision_fx24_embedded | inference | kv260 | fx24_10 | — | — | parsed | passed | not_run | vivado_implementation_available |
| inference | I3_parallel_pe2 | inference | kv260 | fx16_6 | — | — | parsed | passed | not_run | vivado_implementation_available |
| inference | I4_parallel_pe4 | inference | kv260 | fx16_6 | — | — | parsed | passed | not_run | vivado_implementation_available |
| inference | I5_pipeline_latency_first | inference | kv260 | fx16_6 | — | — | parsed | passed | not_run | vivado_implementation_available |
| inference | I6_pipeline_resource_first | inference | kv260 | fx16_6 | — | — | parsed | passed | not_run | vivado_implementation_available |
| inference | I7_weight_import_m_axi | inference | kv260 | fx16_6 | — | — | parsed | passed | not_run | vivado_implementation_available |
| inference | I8_deployable_bitstream_candidate | inference | kv260 | fx16_6 | — | — | parsed | passed | not_run | vivado_implementation_available |
| inference | I9_board_runtime_candidate | inference | kv260 | fx16_6 | — | — | parsed | passed | not_run | vivado_implementation_available |
| training | T0_sgd_tiled_m_axi | training_on_device | kv260 | fx16_6, grad fx24_10 | ddr | — | failed | passed | not_run | vivado_implementation_available |
| training | T1_momentum_tiled_m_axi | training_on_device | kv260 | fx16_6, grad fx24_10 | ddr | — | failed | passed | not_run | vivado_implementation_available |
| training | T3_cross_entropy_tiled_m_axi | training_on_device | kv260 | fx16_6, grad fx24_10 | ddr | — | failed | passed | not_run | vivado_implementation_available |
| training | T4_tile32_m_axi | training_on_device | kv260 | fx16_6, grad fx24_10 | ddr | — | failed | passed | not_run | vivado_implementation_available |
| training | T5_tile128_m_axi | training_on_device | kv260 | fx16_6, grad fx24_10 | ddr | — | failed | passed | not_run | vivado_implementation_available |
| training | T6_accum_batch2_m_axi | training_on_device | kv260 | fx16_6, grad fx24_10 | ddr | — | failed | passed | not_run | vivado_implementation_available |
| training | T7_deployable_training_bitstream | training_on_device | kv260 | fx16_6, grad fx24_10 | ddr | — | failed | passed | not_run | vivado_implementation_available |
| training | T8_real_fpga_training_curve_candidate | training_on_device | kv260 | fx16_6, grad fx24_10 | ddr | — | failed | passed | not_run | vivado_implementation_available |
