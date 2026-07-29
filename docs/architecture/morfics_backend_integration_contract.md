# Morfics-to-FPGAI Backend Integration Contract

## Purpose

FPGAI remains a research platform, but its compiler boundary must be reliable enough for Morfics to invoke it as an automated backend. This document defines the architectural target; it does not add Morfics production code to FPGAI.

## Required public operations

FPGAI should eventually expose versioned operations for:

1. project preflight and capability inspection;
2. deterministic package resolution;
3. compilation and validation;
4. artifact and report collection;
5. reproducibility verification.

The CLI and Python API must call the same compiler service layer.

## Compile request

A future `fpgai.compile/v1` request must carry:

- model or model-package reference;
- inference or training mode;
- target board, backend, and toolchain constraints;
- precision, architecture, memory, and transport choices;
- implementation preferences and fallback policy;
- requested validation level;
- requested build stages;
- project-local and authorized package locations.

## Preflight result

Before a costly build, FPGAI must report:

- model and configuration hashes;
- supported and unsupported ONNX nodes;
- available implementation candidates;
- accepted and rejected candidates with reasons;
- required toolchains;
- compatible boards and backends;
- inference and training capability;
- estimated resources and known legality failures.

## Compile result

A stable result object must expose:

- build identifier and status;
- versioned manifest path;
- selected packages and implementations;
- stage-by-stage tool results;
- validation and deployability status;
- generated artifacts and checksums;
- warnings, failures, and unsupported combinations;
- runtime-package location when requested.

Morfics must never need to infer success by parsing terminal output.

## Canonical stage statuses

The shared status vocabulary should include:

- `not_requested`
- `pending`
- `generated`
- `passed`
- `failed`
- `tool_missing`
- `unsupported`
- `blocked`
- `over_limit`
- `not_validated`

## Package isolation

Managed invocation requires:

- explicit package versions and dependency locks;
- checksums and source provenance;
- no silent package replacement;
- project-local override reporting;
- separation of metadata discovery from plugin-code execution;
- isolated generated-output directories;
- preservation of user-owned source files.

## Ownership boundary

FPGAI provides research compiler contracts, compilation, validation, and artifacts. Morfics provides authentication, uploads, queues, workers, secrets, private package authorization, hardware scheduling, deployment, runtime operation, telemetry, billing, and production support.
