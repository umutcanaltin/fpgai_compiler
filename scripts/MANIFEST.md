# FPGAI Script Manifest

This file tracks the temporary status of files under `scripts/`.

The professional public interface of FPGAI is:

    python main.py ...
    fpgai ...

The `scripts/` directory is transitional. Scripts should either be migrated into
the `fpgai/` package, converted into tests, or removed.

## Completed migrations

The following obsolete or migrated scripts have been removed from `scripts/`:

    scripts/compile.py
    scripts/test_weights_pack.py

Validation helpers moved to `fpgai/validation/`:

    scripts/compare_host_vs_onnx.py
    scripts/compare_vitis_vs_host_vs_onnx.py
    scripts/verify.py

Runtime helpers moved to `fpgai/runtime/`:

    scripts/build_bitstream.py
    scripts/generate_kv260_runtime.py
    scripts/make_input_bin.py
    scripts/make_weights_bin.py
    scripts/run_hostcpp.py
    scripts/run_vitis_csim.py

Experiment runner scripts replaced by public CLI commands:

    scripts/run_all_training_experiments.py
    scripts/run_fpgai_experiments.py
    scripts/run_inference_policy_sweep.py
    scripts/run_paper_experiments.py
    scripts/run_policy_sweep.py
    scripts/run_training_policy_sweep.py

Model-suite utilities moved to `fpgai/experiments/`:

    scripts/create_model_suite.py
    scripts/generate_models.py

Use public workflows instead:

    python main.py sweep inspect --config configs/sweeps/precision_selection.yml
    python main.py sweep run --config configs/sweeps/precision_selection.yml --out experiments/precision_selection
    python main.py experiment inspect --config configs/experiments/arxiv_paper.yml
    python main.py experiment run --config configs/experiments/arxiv_paper.yml --out paper_experiments/arxiv

## Remaining migration targets

### Runtime / backend utilities

    scripts/run_vivado_bridge.py

Target package location:

    fpgai/backends/vivado/

### Reporting utilities

    scripts/analyze_fpgai_experiments.py
    scripts/classify_hardware_feasibility.py
    scripts/collect_claim_support_v2.py
    scripts/collect_comm_ablation.py
    scripts/collect_estimator_accuracy.py
    scripts/collect_hardware_knob_tables.py
    scripts/collect_hls_calibration_samples.py
    scripts/collect_paper_evidence.py
    scripts/collect_training_convergence_evidence.py
    scripts/collect_vivado_impl_evidence.py
    scripts/compare_fpgai_experiments.py
    scripts/compare_hardware_knobs.py
    scripts/extract_hls_calibration_dataset.py
    scripts/extract_memory_binding_evidence.py
    scripts/extract_parallel_policy_evidence.py
    scripts/extract_training_accelerator_evidence.py
    scripts/extract_training_accumulated_batch_evidence.py
    scripts/extract_training_batch_multistep_evidence.py
    scripts/extract_training_multi_epoch_convergence_evidence.py
    scripts/extract_training_native_accumulated_batch_evidence.py
    scripts/extract_vivado_bridge_evidence.py
    scripts/generate_paper_artifacts.py
    scripts/inspect_pipeline_policy_artifacts.py
    scripts/make_estimator_tables.py
    scripts/paper_frontier.py
    scripts/paper_training_precision.py
    scripts/paper_training_table.py
    scripts/plot_policy_results.py
    scripts/plot_training_loss_curves.py
    scripts/report_safe_clock_recommendations.py
    scripts/run_hls_calibration_validation.py
    scripts/summarize_hls_calibration_claims.py
    scripts/summarize_policy_results.py

Target package location:

    fpgai/reporting/

Important naming rule for migration:

Generated report text may use neutral terms such as `results`, `reports`,
`artifacts`, and `traceability`. Avoid exposing old internal names in public
README/docs.

### Developer-only diagnostics

    scripts/diagnose_training_csim.py
    scripts/probe_fpgai_config_schema.py

These should not be public workflows. They can stay temporarily until there is a
better developer workflow.

## Migration order

1. Move reporting utilities into `fpgai/reporting/`.
2. Move Vivado bridge command logic into `fpgai/backends/vivado/`.
3. Replace public documentation references to `scripts/`.
4. Remove or archive remaining legacy scripts.
5. Leave at most this README and manifest during transition.
