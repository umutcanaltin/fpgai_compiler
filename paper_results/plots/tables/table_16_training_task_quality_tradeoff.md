# Training task-quality tradeoff

| Design | precision | decision | decision reason | initial_loss | final_loss | loss_delta | gradient cosine | weight-after cosine | weight-delta cosine | gradient MAE | weight-after MAE | HLS latency cycles | Vivado LUT | Vivado DSP | Vivado BRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_sgd_tiled_m_axi | fx16_6, grad fx24_10 | reference_aligned_training_step | gradient direction is aligned with the Python reference | 0.1308 | 0.1278 | -0.003028 | 1 | 1 | 0.9573 | 0.002081 | 0.000665 | 951 | 29370 | 15 | 17.5 |
| T1_momentum_tiled_m_axi | fx16_6, grad fx24_10 | reference_aligned_training_step | gradient direction is aligned with the Python reference | 0.1308 | 0.1278 | -0.003028 | 1 | 1 | 0.9573 | 0.002081 | 0.000665 | 951 | 29370 | 15 | 17.5 |
| T2_adam_tiled_m_axi | fx16_6, grad fx24_10 | aggressive_training_update | weight-delta cosine is materially lower than the reference-update direction | 0.1308 | 0.08728 | -0.04351 | 1 | 0.9962 | 0.6845 | 0.002081 | 0.01631 | 951 | 30563 | 15 | 18 |
| T3_cross_entropy_tiled_m_axi | fx16_6, grad fx24_10 | acceptable_training_tradeoff | training update differs from reference but preserves usable gradient/weight evidence | 0.5643 | 0.5623 | -0.001997 | 1 | 1 | 0.8835 | 0.001027 | 0.000648 | 976 | 34727 | 44 | 17.5 |
| T4_tile32_m_axi | fx16_6, grad fx24_10 | reference_aligned_training_step | gradient direction is aligned with the Python reference | 0.1308 | 0.1278 | -0.003028 | 1 | 1 | 0.9573 | 0.002081 | 0.000665 | 823 | 28989 | 15 | 17.5 |
| T5_tile128_m_axi | fx16_6, grad fx24_10 | reference_aligned_training_step | gradient direction is aligned with the Python reference | 0.1308 | 0.1278 | -0.003028 | 1 | 1 | 0.9573 | 0.002081 | 0.000665 | 1207 | 29818 | 15 | 17.5 |
| T6_accum_batch2_m_axi | fx16_6, grad fx24_10 | reference_aligned_training_step | gradient direction is aligned with the Python reference | 0.1308 | 0.1278 | -0.003028 | 1 | 1 | 0.9573 | 0.002081 | 0.000665 | 951 | 29370 | 15 | 17.5 |
| T7_deployable_training_bitstream | fx16_6, grad fx24_10 | reference_aligned_training_step | gradient direction is aligned with the Python reference | 0.1308 | 0.1278 | -0.003028 | 1 | 1 | 0.9573 | 0.002081 | 0.000665 | 951 | 29370 | 15 | 17.5 |
| T8_real_fpga_training_curve_candidate | fx16_6, grad fx24_10 | reference_aligned_training_step | gradient direction is aligned with the Python reference | 0.1308 | 0.1278 | -0.003028 | 1 | 1 | 0.9573 | 0.002081 | 0.000665 | 951 | 29370 | 15 | 17.5 |
