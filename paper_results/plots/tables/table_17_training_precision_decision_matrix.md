# Training precision decision matrix

| Design | precision | decision | gradient cosine | weight-delta cosine | loss_delta | HLS latency cycles | Vivado LUT | Vivado DSP | Vivado BRAM | Power W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_sgd_tiled_m_axi | fx16_6, grad fx24_10 | reference_aligned_training_step | 1 | 0.9573 | -0.003028 | 951 | 29370 | 15 | 17.5 | 2.941 |
| T1_momentum_tiled_m_axi | fx16_6, grad fx24_10 | reference_aligned_training_step | 1 | 0.9573 | -0.003028 | 951 | 29370 | 15 | 17.5 | 2.941 |
| T2_adam_tiled_m_axi | fx16_6, grad fx24_10 | aggressive_training_update | 1 | 0.6845 | -0.04351 | 951 | 30563 | 15 | 18 | 2.93 |
| T3_cross_entropy_tiled_m_axi | fx16_6, grad fx24_10 | acceptable_training_tradeoff | 1 | 0.8835 | -0.001997 | 976 | 34727 | 44 | 17.5 | 3.007 |
| T4_tile32_m_axi | fx16_6, grad fx24_10 | reference_aligned_training_step | 1 | 0.9573 | -0.003028 | 823 | 28989 | 15 | 17.5 | 2.925 |
| T5_tile128_m_axi | fx16_6, grad fx24_10 | reference_aligned_training_step | 1 | 0.9573 | -0.003028 | 1207 | 29818 | 15 | 17.5 | 2.951 |
| T6_accum_batch2_m_axi | fx16_6, grad fx24_10 | reference_aligned_training_step | 1 | 0.9573 | -0.003028 | 951 | 29370 | 15 | 17.5 | 2.941 |
| T7_deployable_training_bitstream | fx16_6, grad fx24_10 | reference_aligned_training_step | 1 | 0.9573 | -0.003028 | 951 | 29370 | 15 | 17.5 | 2.941 |
| T8_real_fpga_training_curve_candidate | fx16_6, grad fx24_10 | reference_aligned_training_step | 1 | 0.9573 | -0.003028 | 951 | 29370 | 15 | 17.5 | 2.941 |
