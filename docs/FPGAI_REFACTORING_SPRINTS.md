# FPGAI Modular Refactoring Sprint Track

## Objective

FPGAI now contains inference, training, numerical validation, transport, runtime,
HLS, Vivado, reporting, and paper-experiment mechanisms. Refactoring will proceed
incrementally so each major mechanism has a clear owner module and stable public
interface without breaking existing YAML, CLI, generated artifacts, or tests.

The goal is **cohesive mechanism modules**, not an arbitrary one-file-per-function
layout. Closely related helpers remain together; unrelated compiler stages must
not accumulate in the same implementation file.

## Mandatory refactoring rules

1. Inspect existing owners and call paths before moving code.
2. Preserve public imports, CLI behavior, YAML semantics, artifact schemas, and
   generated C++ unless a sprint explicitly changes a validated contract.
3. Move one bounded mechanism at a time behind a compatibility facade.
4. Add focused tests before deleting the old implementation.
5. Do not create detached scripts or duplicate pipelines.
6. Remove the old owner after migration; do not leave parallel implementations.
7. Run focused and broad regression suites after each extraction.
8. Record implementation status and unsupported combinations explicitly.

## R1 — Training HLS mechanism extraction

Status: started.

Target modules:

- `training_fused_update.py` — Dense/Adam fused parameter-gradient lowering.
- `training_optimizer_adam.py` — Adam state, bias correction, arithmetic, export.
- `training_optimizer_momentum.py` — Momentum state and update lowering.
- `training_gradient_export.py` — full/tiled/recompute export mechanisms.
- `training_accumulation.py` — accumulate/apply/reset runtime modes.
- `training_memory_bindings.py` — BRAM/URAM/DDR storage directives.

`top_train_cpp.py` remains the orchestration facade and base graph emitter.

## R2 — Training reference decomposition

Target modules:

- forward reference execution;
- loss and output-gradient reference;
- backward activation propagation;
- parameter-gradient generation;
- optimizer reference semantics;
- capture serialization and canonical layout.

`training_reference.py` remains the public entry point during migration.

## R3 — Training testbench decomposition

Target modules:

- dataset and stream feeding;
- runtime mode invocation;
- loss/accuracy evaluation;
- update trace and checkpoints;
- numeric artifact export;
- optimizer-state capture.

The generated testbench must remain a single understandable C++ translation
unit; only the Python generator responsibilities are separated.

## R4 — Numeric validation decomposition

Target modules:

- capture contracts and canonicalization;
- comparability;
- tensor comparison and mismatch localization;
- optimizer-state validation;
- mechanism-equivalence promotion;
- report serialization.

## R5 — Compiler orchestration decomposition

Split compile orchestration by stable stages:

- model import and IR preparation;
- architecture planning;
- code generation;
- HLS execution;
- Vivado execution;
- runtime packaging;
- validation and reports.

The `Compiler` class remains a facade and must not duplicate stage logic.

## R6 — Backend and report cleanup

- Remove dead wrappers, unused helpers, stale compatibility branches, and
  unreachable code only after repository-wide usage inspection.
- Consolidate report schemas and professional status terminology.
- Add module ownership documentation and dependency-direction checks.

## Completion criteria for every refactoring sprint

- existing behavior remains available;
- YAML knobs still materially affect generated implementation;
- focused tests pass;
- broad regression tests pass;
- generated C++ remains readable;
- no duplicate owner remains;
- project status documentation is updated.
