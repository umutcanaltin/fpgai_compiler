# Scripts Directory

This directory contains transitional compatibility utilities and developer/paper helpers.

The public user interface is the `fpgai` CLI:

```bash
fpgai inspect --config configs/examples/inference_compile.yml
fpgai compile --config configs/examples/inference_compile.yml
fpgai benchmark --config configs/examples/inference_compile.yml
fpgai sweep inspect --config configs/sweeps/inference_precision.yml
fpgai sweep run --config configs/sweeps/inference_precision.yml --out experiments/inference_precision
fpgai evidence inspect --config configs/paper/arxiv_evidence.yml
```

New user-facing workflows should be added under the `fpgai` package and exposed through `fpgai.cli`, not added as standalone scripts.

Existing scripts should be migrated over time into:

```text
fpgai/experiments/
fpgai/evidence/
fpgai/runtime/
fpgai/backends/
```
