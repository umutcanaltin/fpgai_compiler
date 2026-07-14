# Numeric validation summary

| section | design_id | precision | numeric status | numeric_quality | claim allowed | max abs error | mae | rmse | cosine | gradient cosine | weight-after cosine | weight-delta cosine | loss_validation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inference | I0_baseline_fx16_embedded | fx16_6 | failed_tolerance | failed_numeric_validation | false | 0.459 | 0.0916 | 0.1826 | 0.8394 | — | — | — | not_applicable |
| inference | I1_precision_fx8_embedded | fx8_3 | failed_tolerance | failed_numeric_validation | false | 0.9062 | 0.1625 | 0.3001 | 0.3162 | — | — | — | not_applicable |
| inference | I2_precision_fx24_embedded | fx24_10 | passed | passed | true | 2.146e-05 | 4.294e-06 | 9.464e-06 | 1 | — | — | — | not_applicable |
| inference | I3_parallel_pe2 | fx16_6 | failed_tolerance | failed_numeric_validation | false | 0.459 | 0.0916 | 0.1826 | 0.8394 | — | — | — | not_applicable |
| inference | I4_parallel_pe4 | fx16_6 | failed_tolerance | failed_numeric_validation | false | 0.459 | 0.0916 | 0.1826 | 0.8394 | — | — | — | not_applicable |
| inference | I5_pipeline_latency_first | fx16_6 | failed_tolerance | failed_numeric_validation | false | 0.459 | 0.0916 | 0.1826 | 0.8394 | — | — | — | not_applicable |
| inference | I6_pipeline_resource_first | fx16_6 | failed_tolerance | failed_numeric_validation | false | 0.459 | 0.0916 | 0.1826 | 0.8394 | — | — | — | not_applicable |
| inference | I7_weight_import_m_axi | fx16_6 | failed_tolerance | failed_numeric_validation | false | 0.459 | 0.0916 | 0.1826 | 0.8394 | — | — | — | not_applicable |
| inference | I8_deployable_bitstream_candidate | fx16_6 | failed_tolerance | failed_numeric_validation | false | 0.459 | 0.0916 | 0.1826 | 0.8394 | — | — | — | not_applicable |
| inference | I9_board_runtime_candidate | fx16_6 | failed_tolerance | failed_numeric_validation | false | 0.459 | 0.0916 | 0.1826 | 0.8394 | — | — | — | not_applicable |
| training | T0_sgd_tiled_m_axi | fx16_6, grad fx24_10 | passed | passed | true | — | — | — | — | 1 | 1 | 0.9573 | not_requested |
| training | T1_momentum_tiled_m_axi | fx16_6, grad fx24_10 | passed | passed | true | — | — | — | — | 1 | 1 | 0.9573 | not_requested |
| training | T2_adam_tiled_m_axi | fx16_6, grad fx24_10 | passed | passed | true | — | — | — | — | 1 | 0.9962 | 0.6845 | not_requested |
| training | T3_cross_entropy_tiled_m_axi | fx16_6, grad fx24_10 | passed | passed | true | — | — | — | — | 1 | 1 | 0.8835 | artifact_missing |
| training | T4_tile32_m_axi | fx16_6, grad fx24_10 | passed | passed | true | — | — | — | — | 1 | 1 | 0.9573 | not_requested |
| training | T5_tile128_m_axi | fx16_6, grad fx24_10 | passed | passed | true | — | — | — | — | 1 | 1 | 0.9573 | not_requested |
| training | T6_accum_batch2_m_axi | fx16_6, grad fx24_10 | passed | passed | true | — | — | — | — | 1 | 1 | 0.9573 | not_requested |
| training | T7_deployable_training_bitstream | fx16_6, grad fx24_10 | passed | passed | true | — | — | — | — | 1 | 1 | 0.9573 | not_requested |
| training | T8_real_fpga_training_curve_candidate | fx16_6, grad fx24_10 | passed | passed | true | — | — | — | — | 1 | 1 | 0.9573 | not_requested |
