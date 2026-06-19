# Sprint 13E native accumulated mini-batch evidence

| design | status | weights_mode | hls_ok | training_compare | native_accumulated_optimizer | accumulated_batch | averaged_gradients | gradient_accumulation_mode | optimizer_apply_mode | reset_accumulator_mode | optimizer_location | train_steps | batch_size | total_forward_backward_calls | optimizer_update_calls | weight_words | grad_cosine | weight_after_cosine | weight_delta_cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| training_cnn_stream_native_accum_1step_b2_balanced |  | stream | True | True | True | True | True | True | True | True | hls_top_accumulated_optimizer | 1 | 2 | 2 | 1 | 6810 | 0.5514771342277527 | 0.626254141330719 | 0.5514865517616272 |
