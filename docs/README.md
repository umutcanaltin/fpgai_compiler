# FPGAI Documentation

This directory contains user-facing and developer-facing documentation for FPGAI.

## Start here

- [`../README.md`](../README.md): project overview, main workflow, supported boards, and current limitations.
- [`CLI_WORKFLOWS.md`](CLI_WORKFLOWS.md): practical command examples for inspect, compile, benchmark, sweeps, reports, Vivado bridge generation, and runtime packages.
- [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md): current implementation status and validation boundaries.
- [`FEATURE_MATRIX.md`](FEATURE_MATRIX.md): feature support contract and implementation status.
- [`DEVELOPMENT_ROADMAP.md`](DEVELOPMENT_ROADMAP.md): forward technical roadmap.
- [`CONFIGURATION_KNOBS.md`](CONFIGURATION_KNOBS.md): configuration and hardware-knob reference.

## Workflow references

- [`REPOSITORY_STRUCTURE.md`](REPOSITORY_STRUCTURE.md): repository ownership, naming, readability, and generated-artifact policy.
- [`CONFIG_FIRST_WORKFLOW.md`](CONFIG_FIRST_WORKFLOW.md): config-first development and usage policy.
- [`inspect_command.md`](inspect_command.md): inspect command behavior.
- [`logging.md`](logging.md): quiet/verbose logging behavior.

## Research and reproducibility


## Generated outputs

Generated build outputs normally live under:

```text
build/
experiments/
```

These generated outputs should normally not be committed.
