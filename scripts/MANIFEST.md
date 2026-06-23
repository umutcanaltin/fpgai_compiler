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

Reporting utilities moved to `fpgai/reporting/`.

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

### Developer-only diagnostics

    scripts/diagnose_training_csim.py
    scripts/probe_fpgai_config_schema.py

These should not be public workflows. They can stay temporarily until there is a
better developer workflow or tests cover the same behavior.

## Migration order

1. Move Vivado bridge command logic into `fpgai/backends/vivado/`.
2. Replace public documentation references to `scripts/`.
3. Remove or archive remaining legacy scripts.
4. Leave at most this README and manifest during transition.
