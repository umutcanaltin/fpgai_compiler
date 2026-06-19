# Sprint 13D accumulated mini-batch evidence

| design | status | weights_mode | hls_ok | training_compare | accumulated_batch | averaged_gradients | train_steps | batch_size | total_forward_backward_calls | optimizer_update_calls | optimizer_location | weight_words | grad_cosine | weight_after_cosine | weight_delta_cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| training_cnn_stream_accum_1step_b2_balanced | passed | stream | True | True | False | False | 1 | 2 | 2 |  |  | 6810 | 0.0 | 0.626254141330719 | 0.5514865517616272 |
| training_cnn_stream_accum_2step_b2_balanced | passed | stream | True | True | False | False | 2 | 2 | 4 |  |  | 6810 | 0.0 | 0.626254141330719 | 0.5514865517616272 |
| training_cnn_stream_accum_2step_b4_balanced | passed | stream | True | True | False | False | 2 | 4 | 8 |  |  | 6810 | 0.0 | 0.626254141330719 | 0.5514865517616272 |
