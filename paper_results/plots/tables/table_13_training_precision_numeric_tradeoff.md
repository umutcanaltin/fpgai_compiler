# Training precision numeric tradeoff

| Design | precision | HLS latency cycles | Vivado LUT | Vivado DSP | Vivado BRAM | Power W | numeric_validation_status | numeric_quality | gradient cosine | weight-after cosine | weight-delta cosine | grad_mae | grad_max_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_sgd_tiled_m_axi | fx16_6, grad fx24_10 | 951 | 29370 | 15 | 17.5 | 2.941 | passed | passed | 1 | 1 | 0.9573 | 0.002081 | 0.01139 |
| T1_momentum_tiled_m_axi | fx16_6, grad fx24_10 | 951 | 29370 | 15 | 17.5 | 2.941 | passed | passed | 1 | 1 | 0.9573 | 0.002081 | 0.01139 |
| T2_adam_tiled_m_axi | fx16_6, grad fx24_10 | 951 | 30563 | 15 | 18 | 2.93 | passed | passed | 1 | 0.9962 | 0.6845 | 0.002081 | 0.01139 |
| T3_cross_entropy_tiled_m_axi | fx16_6, grad fx24_10 | 976 | 34727 | 44 | 17.5 | 3.007 | passed | passed | 1 | 1 | 0.8835 | 0.001027 | 0.004856 |
| T4_tile32_m_axi | fx16_6, grad fx24_10 | 823 | 28989 | 15 | 17.5 | 2.925 | passed | passed | 1 | 1 | 0.9573 | 0.002081 | 0.01139 |
| T5_tile128_m_axi | fx16_6, grad fx24_10 | 1207 | 29818 | 15 | 17.5 | 2.951 | passed | passed | 1 | 1 | 0.9573 | 0.002081 | 0.01139 |
| T6_accum_batch2_m_axi | fx16_6, grad fx24_10 | 951 | 29370 | 15 | 17.5 | 2.941 | passed | passed | 1 | 1 | 0.9573 | 0.002081 | 0.01139 |
| T7_deployable_training_bitstream | fx16_6, grad fx24_10 | 951 | 29370 | 15 | 17.5 | 2.941 | passed | passed | 1 | 1 | 0.9573 | 0.002081 | 0.01139 |
| T8_real_fpga_training_curve_candidate | fx16_6, grad fx24_10 | 951 | 29370 | 15 | 17.5 | 2.941 | passed | passed | 1 | 1 | 0.9573 | 0.002081 | 0.01139 |
