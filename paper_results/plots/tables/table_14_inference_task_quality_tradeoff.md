# Inference task-quality tradeoff

| Design | precision | task | dataset_source | samples | decision | prediction agreement | class_change_count | ref top-1 | generated top-1 | ref top-1 acc | generated top-1 acc | top-1 delta pct | ref target MAE | generated target MAE | MAE increase | HLS latency cycles | Vivado LUT | Vivado BRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I0_baseline_fx16_embedded | fx16_6 | classification | not_provided | 1 | recommended_quality | 1 | 0 | 9 | 9 | — | — | — | — | — | — | 329247 | 12708 | 62.5 |
| I1_precision_fx8_embedded | fx8_3 | classification | not_provided | 1 | not_recommended_for_quality | 0 | 1 | 9 | 0 | — | — | — | — | — | — | 330920 | 8019 | 21 |
| I2_precision_fx24_embedded | fx24_10 | classification | not_provided | 1 | recommended_quality | 1 | 0 | 9 | 9 | — | — | — | — | — | — | 329260 | 16801 | 84 |
| I3_parallel_pe2 | fx16_6 | classification | not_provided | 1 | recommended_quality | 1 | 0 | 9 | 9 | — | — | — | — | — | — | 231550 | 39702 | 71 |
| I4_parallel_pe4 | fx16_6 | classification | not_provided | 1 | recommended_quality | 1 | 0 | 9 | 9 | — | — | — | — | — | — | 173667 | 41233 | 91 |
| I5_pipeline_latency_first | fx16_6 | classification | not_provided | 1 | recommended_quality | 1 | 0 | 9 | 9 | — | — | — | — | — | — | 342397 | 44323 | 64 |
| I6_pipeline_resource_first | fx16_6 | classification | not_provided | 1 | recommended_quality | 1 | 0 | 9 | 9 | — | — | — | — | — | — | 329247 | 12708 | 62.5 |
| I7_weight_import_m_axi | fx16_6 | classification | not_provided | 1 | recommended_quality | 1 | 0 | 9 | 9 | — | — | — | — | — | — | 228893 | 22251 | 71 |
| I8_deployable_bitstream_candidate | fx16_6 | classification | not_provided | 1 | recommended_quality | 1 | 0 | 9 | 9 | — | — | — | — | — | — | 329247 | 12708 | 62.5 |
| I9_board_runtime_candidate | fx16_6 | classification | not_provided | 1 | recommended_quality | 1 | 0 | 9 | 9 | — | — | — | — | — | — | 329247 | 12708 | 62.5 |
