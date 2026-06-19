# Training accelerator evidence

| design | status | model | hls_ok | training_plan | training_reference | training_compare | has_training_top | has_preload_mode | has_emit_weights_mode | has_train_mode | has_sgd_update | has_backward | hls_grads_bin | hls_weights_before_bin | hls_weights_after_bin | grad_cosine | grad_mae | grad_max_abs | weight_after_cosine | weight_after_mae | weight_after_max_abs | weight_delta_cosine | weight_delta_mae | weight_delta_max_abs | forward_layers | backward_layers | param_grad_weight_layers | param_grad_bias_layers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| training_cnn_embedded_balanced | passed | models/cnn_mnist.onnx | True | True | True | False | True | False | True | True | True | True | False | False | False |  |  |  |  |  |  |  |  |  |  |  |  |  |
| training_cnn_stream_balanced | passed | models/cnn_mnist.onnx | True | True | True | False | True | True | True | True | True | True | False | False | False |  |  |  |  |  |  |  |  |  |  |  |  |  |
| training_cnn_embedded_latency_first | passed | models/cnn_mnist.onnx | True | True | True | False | True | False | True | True | True | True | False | False | False |  |  |  |  |  |  |  |  |  |  |  |  |  |
| training_cnn_stream_latency_first | passed | models/cnn_mnist.onnx | True | True | True | False | True | True | True | True | True | True | False | False | False |  |  |  |  |  |  |  |  |  |  |  |  |  |
