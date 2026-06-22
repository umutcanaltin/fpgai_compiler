# Sprint 16F Communication-Aware Ablation Evidence

This table summarizes transfer-volume ablations derived from existing FPGAI communication/memory artifacts. It does not claim physical-board runtime improvement.

## Summary

- design count: 17
- rows: 51
- rows with reduced transfer volume vs raw: 34

| design | mode | precision | input_elements | raw_bytes | transfer_bytes | dma_words_32b | compression_ratio_vs_raw | assumed_sparsity | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| hw_parallel_1 | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_parallel_1 | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_parallel_1 | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_parallel_2 | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_parallel_2 | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_parallel_2 | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_parallel_4 | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_parallel_4 | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_parallel_4 | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_prec_fx12_like | raw_fp32 | fx12_like | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_prec_fx12_like | precision_packed | fx12_like | 784 | 3136 | 1176 | 294 | 2.6667 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_prec_fx12_like | zero_run_length_sparse | fx12_like | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_prec_fx16_like | raw_fp32 | fx16_like | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_prec_fx16_like | precision_packed | fx16_like | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_prec_fx16_like | zero_run_length_sparse | fx16_like | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_prec_fx8_like | raw_fp32 | fx8_like | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_prec_fx8_like | precision_packed | fx8_like | 784 | 3136 | 784 | 196 | 4.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_prec_fx8_like | zero_run_length_sparse | fx8_like | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_pipeline_aggressive_strict | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_pipeline_aggressive_strict | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_pipeline_aggressive_strict | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_pipeline_balanced_strict | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_pipeline_balanced_strict | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_pipeline_balanced_strict | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_pipeline_conservative_strict | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_pipeline_conservative_strict | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_pipeline_conservative_strict | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_parallel_1_feasible | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_parallel_1_feasible | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_parallel_1_feasible | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_parallel_2_feasible | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_parallel_2_feasible | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_parallel_2_feasible | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_parallel_2_relaxed_180mhz | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_parallel_2_relaxed_180mhz | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_parallel_2_relaxed_180mhz | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_parallel_3_candidate | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_parallel_3_candidate | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_parallel_3_candidate | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| hw_parallel_4_expected_resource_fail | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| hw_parallel_4_expected_resource_fail | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| hw_parallel_4_expected_resource_fail | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| training_cnn_stream_converge_2epoch_b2_balanced | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| training_cnn_stream_converge_2epoch_b2_balanced | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| training_cnn_stream_converge_2epoch_b2_balanced | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| training_cnn_stream_converge_3epoch_b2_balanced | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| training_cnn_stream_converge_3epoch_b2_balanced | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| training_cnn_stream_converge_3epoch_b2_balanced | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |
| training_cnn_stream_converge_3epoch_b4_balanced | raw_fp32 | fixed_point | 784 | 3136 | 3136 | 784 | 1.0 |  | fallback_assumption_mnist_input_784;baseline_32bit_float_transfer |
| training_cnn_stream_converge_3epoch_b4_balanced | precision_packed | fixed_point | 784 | 3136 | 1568 | 392 | 2.0 |  | fallback_assumption_mnist_input_784;modeled_precision_packing |
| training_cnn_stream_converge_3epoch_b4_balanced | zero_run_length_sparse | fixed_point | 784 | 3136 | 312 | 78 | 10.0513 | 0.9 | fallback_assumption_mnist_input_784;modeled_zrl_sparse_input_sparsity_0.9 |

## Safe claim

FPGAI communication artifacts can be used to quantify transfer-volume trade-offs between raw, precision-packed, and sparse zero-run-length-style transfer formats for evaluated designs.

## Limitation

This sprint reports transfer-volume evidence only. It does not measure physical-board DMA latency or runtime speedup. Physical runtime validation is reserved for later board-validation sprints.
