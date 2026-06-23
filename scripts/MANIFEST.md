# FPGAI Script Manifest

This directory is intentionally empty except for documentation.

The professional public interface of FPGAI is:

    python main.py ...
    fpgai ...

All previous executable scripts have been removed, migrated into the `fpgai/`
package, or replaced by public CLI workflows.

## Completed migrations

Obsolete helpers removed:

    scripts/compile.py
    scripts/test_weights_pack.py

Validation helpers moved to:

    fpgai/validation/

Runtime helpers moved to:

    fpgai/runtime/

Experiment runner scripts replaced by public CLI commands:

    python main.py sweep inspect --config configs/sweeps/precision_selection.yml
    python main.py sweep run --config configs/sweeps/precision_selection.yml --out experiments/precision_selection
    python main.py experiment inspect --config configs/experiments/arxiv_paper.yml
    python main.py experiment run --config configs/experiments/arxiv_paper.yml --out paper_experiments/arxiv

Model-suite utilities moved to:

    fpgai/experiments/model_suite.py
    fpgai/experiments/model_generation.py

Reporting utilities moved to:

    fpgai/reporting/

Vivado bridge utility moved to:

    fpgai/backends/vivado/run_bridge.py

Developer diagnostics moved to:

    fpgai/devtools/diagnose_training_csim.py
    fpgai/devtools/probe_config_schema.py

## Policy

Do not add new executable workflows under `scripts/`.

Reusable logic belongs inside `fpgai/` and should be exposed through the public
CLI when it is stable.
