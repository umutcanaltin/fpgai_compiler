# FPGAI Paper Master Results

Rows: 13

| design_id | model | board | mode | estimated_lut | estimated_dsp | estimated_latency_ms | hls_status | vivado_implementation_status | runtime_status | support_status |
|---|---|---|---|---|---|---|---|---|---|---|
| compact_onchip_mnist_mlp | main_graph | kv260 | inference |  |  | 0.259375 | parsed | not_requested | not_run | hls_synthesis_available |
| compact_onchip_mnist_mlp_vivado | main_graph | kv260 | inference |  |  | 0.259375 | parsed | passed | not_run | vivado_implementation_available |
| compact_onchip_mnist_training | main_graph | kv260 | training_on_device |  |  | 0.259375 | parsed | not_requested | not_run | hls_synthesis_available |
| medium_ddr_cifar_cnn | main_graph | kv260 | inference |  |  | 0.04223 | parsed | not_requested | not_run | hls_synthesis_available |
| medium_ddr_cifar_cnn_vivado | main_graph | kv260 | inference |  |  | 0.04223 | parsed | passed | not_run | vivado_implementation_available |
| medium_ddr_cifar_training | main_graph | kv260 | training_on_device |  |  | 0.04223 | parsed | not_requested | not_run | hls_synthesis_available |
| large_ddr_stress_cnn | main_graph | kv260 | inference |  |  | 1.37666 | parsed | not_requested | not_run | hls_synthesis_available |
| mnist_mlp_embedded | main_graph | kv260 | inference |  |  | 0.00072 | not_requested | not_requested | not_run | static_validation_passed |
| mnist_mlp_import_weights | main_graph | kv260 | inference |  |  | 0.00072 | not_requested | not_requested | not_run | static_validation_passed |
| mnist_mlp_training_sgd | main_graph | kv260 | training_on_device |  |  | 0.00072 | not_requested | not_requested | not_run | static_validation_passed |
| cpp_only | main_graph | kv260 | inference |  |  | 0.00058 | not_requested | not_requested | not_run | static_validation_passed |
| hls_project | main_graph | kv260 | inference |  |  | 0.00058 | not_requested | not_requested | not_run | static_validation_passed |
| vivado_project | main_graph | kv260 | inference |  |  | 0.00058 | not_requested | not_requested | not_run | static_validation_passed |
