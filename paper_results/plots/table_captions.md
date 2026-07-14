# Table captions

| table | path | row_count | caption |
| --- | --- | --- | --- |
| table_01_experiment_overview | paper_results/plots/tables/table_01_experiment_overview.md | 19 | Overview of frozen paper experiment rows, pipeline mode, compact precision label, and artifact support status. |
| table_02_design_knobs | paper_results/plots/tables/table_02_design_knobs.md | 343 | Resolved YAML/hardware knob settings and where each knob was applied in the compiler/hardware flow. |
| table_03_hls_results | paper_results/plots/tables/table_03_hls_results.md | 19 | Vitis HLS synthesis metrics used for paper HLS latency/resource comparisons. |
| table_04_vivado_results | paper_results/plots/tables/table_04_vivado_results.md | 19 | Vivado implementation metrics used for paper resource, timing, and power comparisons. |
| table_05_runtime_status | paper_results/plots/tables/table_05_runtime_status.md | 19 | Board-runtime measurement availability and pending runtime metrics. |
| table_06_pending_measurements | paper_results/plots/tables/table_06_pending_measurements.md | 76 | Explicit list of real board measurements still required before claiming runtime or training-curve results. |
| table_07_knob_effect_summary | paper_results/plots/tables/table_07_knob_effect_summary.md | 35 | Aggregated coverage of knob statuses across generated hardware knob contracts. |
| table_08_result_comparisons | paper_results/plots/tables/table_08_result_comparisons.md | 135 | Pairwise paper comparisons computed from the frozen experiment subset, including automatic result classification and interpretation. |
| table_09_result_classification_summary | paper_results/plots/tables/table_09_result_classification_summary.md | 4 | Summary count of comparison classifications used to separate strong results, expected tradeoffs, deployability rows, no-observable-effect rows, and pending runtime claims. |
| table_10_numeric_validation_summary | paper_results/plots/tables/table_10_numeric_validation_summary.md | 19 | Numeric validation status for each frozen design, including inference output-error metrics and training gradient/weight-update metrics when available. |
| table_11_inference_precision_numeric_tradeoff | paper_results/plots/tables/table_11_inference_precision_numeric_tradeoff.md | 3 | Inference precision rows combining HLS/Vivado resource metrics with numeric output fidelity metrics. |
| table_12_training_numeric_validation | paper_results/plots/tables/table_12_training_numeric_validation.md | 9 | Training numeric validation metrics for gradients, weight updates, losses, optimizer-state export, accumulation, and tiled I/O where available. |
| table_13_training_precision_numeric_tradeoff | paper_results/plots/tables/table_13_training_precision_numeric_tradeoff.md | 9 | Training precision/numeric behavior table prepared for current training rows and future training precision variants. |
| table_14_inference_task_quality_tradeoff | paper_results/plots/tables/table_14_inference_task_quality_tradeoff.md | 10 | Task-aware inference decision table combining generated-vs-reference prediction agreement, optional classification/regression dataset metrics, and HLS/Vivado costs. |
| table_15_precision_decision_matrix | paper_results/plots/tables/table_15_precision_decision_matrix.md | 3 | Precision decision matrix that lets users choose fx8/fx16/fx24 using quality metrics, output error, and resource savings. |
| table_16_training_task_quality_tradeoff | paper_results/plots/tables/table_16_training_task_quality_tradeoff.md | 9 | Task-aware training decision table combining loss/update fidelity and HLS/Vivado costs. |
| table_17_training_precision_decision_matrix | paper_results/plots/tables/table_17_training_precision_decision_matrix.md | 9 | Training decision matrix prepared for current training rows and future training precision/dataset variants. |
