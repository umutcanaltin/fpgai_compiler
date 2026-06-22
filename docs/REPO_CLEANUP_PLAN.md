# FPGAI Repository Cleanup Plan

This document defines the professional repository structure for FPGAI after the iteration-heavy development phase.

## Goal

FPGAI should be usable by three groups:

1. **Users** who want to compile an ONNX model using YAML.
2. **Researchers** who want to reproduce paper experiments.
3. **Contributors** who want to extend the compiler without understanding old development history.

## Rule: YAML first, scripts second

Compilation and experiment configuration should be driven by YAML files. Python scripts should be thin entrypoints or paper/experiment artifact utilities, not one-off development logic.

Allowed public entrypoints:

```text
fpgai compile --config <config.yml>
fpgai inspect --config <config.yml>
fpgai benchmark --config <config.yml>
PYTHONPATH="$PWD" python -B scripts/run_fpgai_experiments.py --sweep <sweep.yml> --out <out>
python main.py report build --config configs/experiments/<paper>.yml
```

Everything else should eventually move to one of these categories:

```text
scripts/legacy/       Old development-only scripts kept for reproducibility, not public API.
scripts/paper/        Paper table/figure generation only.
scripts/dev/          Developer diagnostics and migration helpers only.
scripts/examples/     Small demonstration helpers only.
```

Do not delete historical development scripts immediately. First classify them, then migrate stable behavior into package code or YAML workflows.

## Cleanup phases

### Phase P0 — Documentation and policy

Add repository policy files explaining:

- YAML-first workflows.
- Public vs legacy scripts.
- Paper artifact policy.
- Contributor workflow.

### Phase P1 — Script inventory

Classify every file under `scripts/` as one of:

```text
public_entrypoint
paper_artifact
experiment_runner
dev_diagnostic
legacy_workflow
model_generation
runtime_helper
```

Output should be a table in `docs/SCRIPT_INVENTORY.md`.

### Phase P2 — Stable CLI

Promote stable user workflows to the installed `fpgai` CLI:

```text
fpgai compile --config configs/examples/inference_compile.yml
fpgai inspect --config configs/examples/inference_compile.yml
fpgai benchmark --config configs/examples/inference_compile.yml
fpgai sweep --config configs/sweeps/<sweep>.yml --out experiments/<name>
fpgai experiment run --config configs/experiments/arxiv_paper.yml
```

Until these subcommands exist, the documented stable commands remain the current `fpgai compile/inspect/benchmark` plus `scripts/run_fpgai_experiments.py`.

### Phase P3 — Estimator hardening

The current estimator experiment artifact is not strong enough for a broad paper claim. The repo needs a real estimator export format:

```text
build/estimator/resource_prediction.json
```

Required fields:

```json
{
  "schema_version": 1,
  "design_name": "...",
  "model": "...",
  "target": {"board": "...", "part": "..."},
  "prediction_scope": "hls_kernel_only|full_bd_estimate",
  "resources": {"lut": 0, "ff": 0, "bram": 0, "dsp": 0},
  "method": "analytical|calibrated|hybrid",
  "notes": []
}
```

Predicted values must not be placeholders such as 0 or 1. If no reliable estimate is available, the file should say `status: unavailable` instead of emitting fake values.

### Phase P4 — Paper artifact separation

Paper scripts should read experiment artifact CSV/JSON and write only paper-ready tables/figures. They should not run compilation.

Suggested output:

```text
paper_experiments/arxiv_tables/
paper_experiments/arxiv_figures/
paper_experiments/arxiv_summary.md
```

### Phase P5 — Real board validation

Physical FPGA runtime, board loss curves, and simulation-vs-board comparisons should be kept separate from HLS/Vivado experiment artifact.

Suggested output:

```text
paper_experiments/board_runtime/
paper_experiments/board_training_loss/
paper_experiments/board_vs_sim/
```

## Claims policy

Do not claim:

```text
FPGAI accurately predicts LUT/FF/DSP/BRAM
FPGAI improves physical DMA runtime
FPGAI demonstrates board-level training convergence
```

unless the corresponding experiment folders exist and pass.

Safe current direction:

```text
FPGAI supports YAML-driven compilation and experiment sweeps, and paper claims are tied to reproducible experiment artifacts.
```
