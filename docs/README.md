# FPGAI Documentation

This directory contains user-facing and developer-facing documentation for FPGAI.

## Start here

- [`../README.md`](../README.md): project overview, main workflow, supported boards, and current limitations.
- [`CLI_WORKFLOWS.md`](CLI_WORKFLOWS.md): practical command examples for inspect, compile, benchmark, sweeps, reports, Vivado bridge generation, and runtime packages.
- [`FPGAI_PROJECT_STATUS.md`](FPGAI_PROJECT_STATUS.md): concise current implementation status and truth boundaries.

## Workflow references

- [`CONFIG_FIRST_WORKFLOW.md`](CONFIG_FIRST_WORKFLOW.md): config-first development and usage policy.
- [`inspect_command.md`](inspect_command.md): inspect command behavior.
- [`logging.md`](logging.md): quiet/verbose logging behavior.

## Research and reproducibility

- [`PAPER_ARTIFACT_POLICY.md`](PAPER_ARTIFACT_POLICY.md): claim levels and artifact policy for paper experiments.

## Generated outputs

Generated build outputs normally live under:

```text
build/
experiments/
```

These generated outputs should normally not be committed.
