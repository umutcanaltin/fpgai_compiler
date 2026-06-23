# FPGAI Project Status

This file is updated during cleanup and feature-completion work. It tracks what has been done, what remains, and what the next safe step is.

## Working rules

- Do not add a new file before checking whether an existing file can be reused, rewired, merged, or rewritten.
- No duplicate reporting, runtime, compiler, or experiment logic.
- No public paper/README claim unless it is implemented, tested, documented, and produces reproducible artifacts.
- Use tests during development as safety checks.
- At the end, remove or reorganize temporary/debug/sprint tests and keep only professional public tests.
- Public workflows should go through `main.py` / `fpgai` CLI.
- Avoid public “evidence” wording; use experiments, reports, artifacts, traceability.

## Current target

Build FPGAI into a professional end-to-end FPGA/SoC compiler pipeline:

ONNX model → model analysis → resource/timing prediction → design selection → Vitis HLS generation → HLS reports → Vivado build/reports → bitstream → runtime package → real FPGA inference/training → paper plots/reports.

## Current phase

Sprint 1: repository truth, tests, package metadata, and cleanup baseline.

## Completed

- Loaded next-chat handoff rules.
- Inspected actual repository file layout.
- Confirmed current tests exist under `tests/`.
- Confirmed `docs/FPGAI_PROJECT_STATUS.md` was missing before this file was created.

## Current state

- `tests/test_precision_config.py` exists in the current repo.
- The repo contains many source, analysis, backend, experiment, reporting, benchmark, and frontend modules.
- Existing reporting functionality is already present under `fpgai/reporting/`; do not add duplicate reporting files before inspecting and wiring existing modules.
- Current docs include cleanup/legacy planning documents that must be inspected before deleting or rewriting:
  - `docs/REPO_CLEANUP_PLAN.md`
  - `docs/development_roadmap.md`
  - `docs/cli_workflows_legacy.md`
- Untracked root archives were observed:
  - `configs.zip`
  - `docs.zip`
  - `fpgai.zip`
  - `models.zip`
  - `scripts.zip`
  - `tests.zip`
- `__pycache__` and `.pyc` files appear in the working tree listing and must be classified as tracked or untracked before cleanup.

## Files changed

- `docs/FPGAI_PROJECT_STATUS.md` created.

## Tests run

- Not yet run in this cleanup pass.
- Next step is to check CLI import/help and test collection only after confirming Git tracking of generated files.

## Decisions made

- Start from real repository inventory, not old assumed test paths.
- Create the status file before any feature or cleanup patch.
- Do not add new modules until existing modules are inspected for reuse/merge/wiring.
- Do not delete `__pycache__`, `.pyc`, zip backups, or stale docs until Git tracking and contents are checked.

## Remaining Sprint 1 tasks

1. Classify generated/cache/archive files as tracked or untracked.
2. Check basic CLI import/help.
3. Run test collection or focused pytest groups.
4. Fix collection/import issues.
5. Resolve package/license/README mismatch.
6. Inspect stale docs before deleting or rewriting.
7. Update this status file after every meaningful change.

## Latest update

- Inspected package/license/README metadata.
- Found mismatch:
  - `LICENSE.md` says academic, educational, and non-commercial research use only.
  - `pyproject.toml` said MIT.
  - README described FPGAI as open-source.
- Patched README and `pyproject.toml` to align with `LICENSE.md`.
- No source/compiler behavior changed.

## Latest update

- Full Sprint 1 baseline is green.
- Final pytest result:
  - `258 passed, 1 skipped in 1.41s`
  - exit code `0`
- Remaining skipped test is the optional HLS CSim integration correctness test when no testable HLS/Vitis CSim YAML config is present.
- Sprint 1 test stabilization is complete enough to commit.

## Latest update

- Inspected exact active source blocks for the last 2 focused failures.
- Found that `top_cpp.py` has multiple `emit_top_cpp` wrappers; the active runtime-weight wrapper is the final one.
- Patched the active runtime-weight wrapper to prepend honest requested-mode planning comments while preserving existing stream/DDR helper insertion.
- Patched the active training mode-6 insertion block using a robust post-loss marker search.

## Latest update

- The first fixture patch did not apply because the exact string replacement patterns were too brittle.
- Applied a more robust line-based test repair that inserts `parent.mkdir(parents=True, exist_ok=True)` after temporary nested config path assignments.
- This remains a test-only repair; compiler/source logic is unchanged.

## Latest update

- Repaired test fixtures that wrote nested temporary config paths without creating parent directories first.
- Changed only tests, not compiler logic.
- Patched:
  - `tests/test_precision_config.py`
  - `tests/test_experiment_config_materializer_signature_compat.py`
  - `tests/test_experiment_config_materializer_strict.py`
- Reason: baseline pytest showed many failures caused by missing `tmp_path/configs/examples/` parent directories.

## Latest update

- Adjusted workflow for long outputs: commands now write logs to `repo_audit/sprint1_baseline/` so the user can upload a ZIP instead of pasting terminal output.
- Next baseline run will capture CLI checks, pytest collection, full pytest output, and Git status.

## Latest update

- Basic CLI sanity check passed:
  - `python -B -m py_compile main.py fpgai/cli.py`
  - `python main.py --help`
  - `python main.py report --help`
  - `python main.py validate --help`
- Pytest collection succeeded with 259 collected tests.
- `report` currently exposes only `build`; future reporting work should wire existing `fpgai/reporting/` modules into this CLI group instead of adding duplicate tools.

## Latest update

- Basic CLI sanity check passed:
  - `python -B -m py_compile main.py fpgai/cli.py`
  - `python main.py --help`
  - `python main.py report --help`
  - `python main.py validate --help`
- Pytest collection succeeded with 259 collected tests.
- `report` currently exposes only `build`; future reporting work should wire existing `fpgai/reporting/` modules into this CLI group instead of adding duplicate tools.

## Latest update

- Classified generated/cache/archive files.
- Confirmed `__pycache__` and `.pyc` files are not tracked by Git.
- Confirmed root ZIP backup archives were untracked local files.
- Removed local root ZIP backup archives:
  - `configs.zip`
  - `docs.zip`
  - `fpgai.zip`
  - `models.zip`
  - `scripts.zip`
  - `tests.zip`
- Updated `.gitignore` to ignore root-level `/*.zip` backup archives.

## Current next step

Run focused tests for repaired config/materializer fixtures, then rerun full pytest baseline.
