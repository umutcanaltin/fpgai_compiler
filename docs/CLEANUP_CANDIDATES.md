# Cleanup candidates

This document starts U0-lite cleanup tracking. It is intentionally conservative: uncertain files should be listed here before deletion.

## Current U0-lite policy

- Do not delete uncertain scripts or old experiment artifacts yet.
- Prefer migrating YAML examples/configs to canonical keys first.
- Keep legacy alias support in the compiler until W0/W strict schema work.
- Remove only obvious duplicates in later cleanup patches after grep/import/test checks.

## Candidate categories for later U0/W0

- Old configs that still use deprecated aliases after Q0 examples are complete.
- Sweep templates that should live under a single documented sweep format.
- Historical paper experiment outputs that should move out of the public package or be marked archival.
- Detached scripts that should become library/CLI commands or be removed if unused.
- Oversized tests that can be split only after coverage-equivalent fixtures exist.

## Newly confirmed refactoring candidates

- `fpgai/engine/compiler.py` is more than 5,600 lines and still coordinates inference, training, reporting, HLS, Vivado, runtime packaging, and validation. Split mode-specific workflow coordinators behind the existing `Compiler` entry point after P3D-F3 stabilizes.
- `fpgai/backends/hls/testbench_train.py` now owns runtime ABI setup, dataset scheduling, trace capture, curves, and checkpoint export. Preserve it as the single generated-testbench owner, but extract internal emitter helpers when doing so can be coverage-preserving.
- Audit old training sweeps that use `training.execution.*`. Migrate them to `training.batch.*` and `training.validation.*`; keep aliases only for compatibility.
- Do not add separate multi-epoch runner scripts. The schedule belongs in `fpgai.engine.training` and must be consumed by compiler, reference, HLS testbench, runtime, and reports.
- Several historical sweep outputs may contain duplicate `already_completed` records created by the old resume behavior. Do not edit those paper artifacts silently; provide a migration/audit command before using them for publication tables.
- Audit documentation-generated YAML inventories after the multi-epoch and canonical-key changes. `docs/YAML_REPO_AUDIT.*` may be stale until the documentation modernization sprint regenerates it from the current repository.

- `validation_data/mnist_samples.npz` is an all-zero, single-class synthetic fixture in the uploaded source snapshot. Keep it only if it has a clearly named mechanism-smoke owner; do not use it in learning/convergence paper sweeps. Consider renaming it to an explicit synthetic-fixture name during the cleanup sprint and updating all references atomically.
- `configs/sweeps/training_multi_epoch_convergence.yml` is retained for compatibility, but its current balanced-10 experiment is a learning smoke rather than a convergence study. Rename the file/output namespace during the paper-experiment cleanup once downstream references and historical result paths are migrated.

- Consolidate training learning-report assembly currently embedded in `Compiler._compile_training` into an existing reporting owner during the compiler workflow refactor. Do not create a detached experiment collector.
- Remove deprecated execution-count aliases after paper tooling and downstream consumers use the canonical schema.

## Experiment artifact-path normalization

Legacy sweep results may contain relative or incorrectly resolved `project_out_dir` values. Retain append-only history, but regenerate paper-facing result sets with the absolute output-path and required-artifact contract introduced in P3D-F3D.1.
