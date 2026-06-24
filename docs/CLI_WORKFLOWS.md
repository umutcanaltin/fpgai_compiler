# FPGAI CLI Workflows

This document defines the public workflows for FPGAI. The `fpgai` CLI is the user-facing interface. Files under `scripts/` are transitional compatibility or developer tools and should not be the normal workflow in documentation.

## Single compile config

Use this for one model/configuration.

```bash
fpgai inspect --config configs/examples/inference_compile.yml
fpgai compile --config configs/examples/inference_compile.yml
fpgai benchmark --config configs/examples/inference_compile.yml
```

`fpgai inspect` is pure inspection. It must not run Vitis HLS, Vivado, benchmark, or compile.

## Sweep config

Use this for design-space exploration.

```bash
fpgai sweep inspect --config configs/sweeps/inference_precision.yml
fpgai sweep run \
  --config configs/sweeps/inference_precision.yml \
  --out experiments/inference_precision \
  --max-design-points 1 \
  --timeout-sec 1200
```

Sweep output goes under `experiments/`. Generated experiment artifacts should not normally be committed.

## Paper experiments config

Use this workflow for reproducibility and paper-result generation.

Inspect the paper experiment configuration before running it:

    fpgai experiment inspect --config configs/experiments/arxiv_paper.yml

Run the paper experiment configuration and write generated outputs under `paper_experiments/`:

    fpgai experiment run \
      --config configs/experiments/arxiv_paper.yml \
      --out paper_experiments/arxiv

Generated experiment outputs should go under `paper_experiments/` or `experiments/`, not into tracked source directories.

## Report and paper artifact generation

Use `fpgai report` to turn existing experiment outputs into reviewer-facing summaries, tables, plots, and traceability reports. These commands reuse the implementation under `fpgai/reporting/`; users do not need to call internal scripts directly.

Build a summary report from an experiment output directory:

    fpgai report build \
      --input experiments/inference_precision \
      --out reports/inference_precision

Generate paper tables and figures from a sweep/result CSV:

    fpgai report paper-artifacts \
      --csv experiments/inference_precision/policy_sweep_results.csv \
      --out reports/inference_precision/paper_artifacts

Generate Pareto/frontier artifacts:

    fpgai report frontier \
      --csv experiments/inference_precision/policy_sweep_results.csv \
      --out reports/inference_precision/frontier

Generate estimator-vs-real resource and latency tables:

    fpgai report estimator \
      --csv experiments/inference_precision/policy_sweep_results.csv \
      --out reports/inference_precision/estimator

Report outputs should go under `reports/` and should not normally be committed.

## Logging policy

Default compile and benchmark commands should print concise FPGAI summaries and paths to logs. Full Vitis/Vivado output should only be shown with `--verbose`.

```bash
fpgai compile --config configs/examples/inference_compile.yml
fpgai compile --config configs/examples/inference_compile.yml --verbose
fpgai compile --config configs/examples/inference_compile.yml --quiet
```

## Quick compile without HLS/Vivado

Use the quick compile example to validate the front half of the pipeline without requiring Vitis HLS or Vivado:

    fpgai compile --config configs/examples/quick_compile.yml --quiet

This writes the manifest, IR plans, model profile, and pre-HLS resource/timing prediction artifacts under:

    build/fpgai_quick_compile/

## Prediction artifacts

Use inspect output for pre-HLS model/resource/timing artifacts:

    fpgai inspect --config configs/examples/inference_compile.yml --out reports/inference_prediction

This writes:

    reports/inference_prediction/model_profile.json
    reports/inference_prediction/resource_prediction.json
    reports/inference_prediction/timing_prediction.json
    reports/inference_prediction/prediction_summary.md

A normal compile also writes the same pre-HLS prediction artifacts under:

    <project.out_dir>/reports/model_profile.json
    <project.out_dir>/reports/resource_prediction.json
    <project.out_dir>/reports/timing_prediction.json
    <project.out_dir>/reports/prediction_summary.md

These files are estimates generated before HLS/Vivado. Real HLS/Vivado reports remain separate artifacts.
