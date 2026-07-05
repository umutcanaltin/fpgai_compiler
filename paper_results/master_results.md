# FPGAI Paper Master Results

Rows: 6

| design_id | model | board | mode | estimated_lut | estimated_dsp | estimated_latency_ms | hls_status | vivado_implementation_status | runtime_status | support_status |
|---|---|---|---|---|---|---|---|---|---|---|
| mnist_mlp_embedded | main_graph | kv260 | inference |  |  | 0.00072 | not_requested | not_requested | not_run | static_validation_passed |
| mnist_mlp_import_weights | main_graph | kv260 | inference |  |  | 0.00072 | not_requested | not_requested | not_run | static_validation_passed |
| mnist_mlp_training_sgd | main_graph | kv260 | training_on_device |  |  | 0.00072 | not_requested | not_requested | not_run | static_validation_passed |
| cpp_only | main_graph | kv260 | inference |  |  | 0.00058 | not_requested | not_requested | not_run | static_validation_passed |
| hls_project | main_graph | kv260 | inference |  |  | 0.00058 | not_requested | not_requested | not_run | static_validation_passed |
| vivado_project | main_graph | kv260 | inference |  |  | 0.00058 | not_requested | not_requested | not_run | static_validation_passed |
