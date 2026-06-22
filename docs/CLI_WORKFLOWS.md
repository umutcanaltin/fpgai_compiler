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

Use this for reproducibility and paper-result generation.

```bash
fpgai experiment inspect --config configs/experiments/arxiv_paper.yml
```

Future experiment artifact commands should follow this pattern:

```bash
fpgai experiment run --config configs/experiments/arxiv_paper.yml --out paper_experiments/arxiv
```

## Logging policy

Default compile and benchmark commands should print concise FPGAI summaries and paths to logs. Full Vitis/Vivado output should only be shown with `--verbose`.

```bash
fpgai compile --config configs/examples/inference_compile.yml
fpgai compile --config configs/examples/inference_compile.yml --verbose
fpgai compile --config configs/examples/inference_compile.yml --quiet
```
