# Sprint 13C training batch/multi-step evidence

| design | status | weights_mode | hls_ok | training_plan | training_reference | training_compare | has_multistep_summary | train_steps | batch_size | total_train_calls | weight_words | hls_weights_before_bin | hls_grads_bin | hls_weights_after_bin | grad_cosine | weight_after_cosine | weight_delta_cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| training_cnn_embedded_2step_b1_balanced | passed | embedded | True | True | True | True | True | 2 | 1 | 2 | 6810 | True | True | True | 0.0 | 0.626254141330719 | 0.5514865517616272 |
| training_cnn_stream_2step_b1_balanced | passed | stream | True | True | True | True | True | 2 | 1 | 2 | 6810 | True | True | True | 0.0 | 0.626254141330719 | 0.5514865517616272 |
