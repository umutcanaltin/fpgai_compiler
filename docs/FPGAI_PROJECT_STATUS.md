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

Sprint 4: model inspection and resource/timing prediction.

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

- Wired `fpgai inspect --config ... --out <dir>` through the existing `inspect_from_config()` helper.
- The command now writes `model_profile.json` and `prediction_summary.md` without replacing the main CLI dispatch branch.
- Existing `--json-output` behavior is preserved.
- No new files were added.

## Latest update

- Wired the existing `inspect` command to emit model-inspection artifacts.
- `fpgai inspect --config ... --out <dir>` now writes:
  - `model_profile.json`
  - `prediction_summary.md`
- Reused `fpgai.analysis.model_inspection`; no new prediction module was added.
- Resource and timing prediction files remain for the next Sprint 4 step.

## Latest update

- Started Sprint 4: model inspection and prediction.
- Sprint 4 rule: inspect and reuse existing analysis modules before adding any new prediction files.
- Target public artifacts for this sprint:
  - `model_profile.json`
  - `resource_prediction.json`
  - `timing_prediction.json`
  - `prediction_summary.md`
- Target behavior: FPGAI should expose model/profile/resource/timing predictions before HLS/Vivado compile, while clearly marking estimates as estimates.
- Current first task: inventory existing `fpgai/analysis` modules and CLI inspect/estimate paths.

## Latest update

- Added compile-summary visibility for `pipeline_stages`.
- Reused `CompileResult.summary()` and the existing compile `manifest.json`.
- Added a regression test in the existing CLI quiet logging test file.
- No new files were added.

## Latest update

- Corrected Sprint 3 pipeline stage regression test placement.
- Removed `pipeline_stages` assertions from paper experiment manifest tests because those manifests are not compile manifests.
- Added a runtime helper test for compile pipeline stage metadata in the existing compiler artifact test file.
- No new files were added.

## Latest update

- Added a runtime regression check for Sprint 3 pipeline stage metadata.
- The test verifies the emitted/constructed pipeline stage list includes the canonical stage names:
  - `load_config`
  - `import_model`
  - `generate_hls`
  - `run_hls`
  - `vivado_bridge`
  - `runtime_package`
- No new test file was added; existing tests were reused.

## Latest update

- Added explicit `pipeline_stages` metadata to the existing compile `manifest.json`.
- Reused `fpgai/engine/compiler.py`; no new pipeline orchestrator or pipeline file was added.
- Stage metadata currently describes the existing compile flow:
  - config load
  - model import
  - model/graph analysis
  - architecture planning
  - optional reports
  - host C++ generation
  - HLS generation
  - optional Vitis HLS run
  - training artifacts when applicable
  - Vivado bridge as `not_requested`
  - runtime package as `not_implemented`
- Added a source-level regression test in existing `tests/test_compiler_artifact_meta.py`.

## Latest update

- Started Sprint 3 canonical pipeline inspection.
- Confirmed current compile flow already has a central manifest path in `fpgai/engine/compiler.py`.
- Current safest Sprint 3 improvement is to extend the existing `manifest.json` with explicit pipeline stage status instead of creating a new pipeline orchestrator or new pipeline file.
- Current owners:
  - CLI compile entry: `fpgai/cli.py`
  - compile orchestration: `fpgai/engine/compiler.py`
  - result model: `fpgai/engine/result.py`
  - HLS backend: `fpgai/backends/hls/`
  - Vivado bridge/backend: `fpgai/backends/vivado/`
- Next step: inspect `_emit_manifest` exactly and add `pipeline_stages` only inside the existing manifest output.

## Latest update

- Applied Sprint 2B public-wording cleanup without requiring uploaded logs.
- Patched known stale public/help/reporting strings from the broad grep output.
- Replaced old `python scripts/...`, `Sprint`, and public `evidence` wording with CLI/report/artifact terminology where behavior stays equivalent.
- Kept deeper internal compatibility comments for later targeted cleanup.
- No new files were added.

## Latest update

- Cleaned two remaining stale names from the selected artifact modules:
  - replaced the `sprint13b` default experiment path with `experiments/training_stream_compare`
  - renamed a local `evidence` variable to `artifact_lines`
- Focused artifact module cleanup remains source-compatible and does not add files.

## Latest update

- Cleaned stale `evidence`/`Sprint` wording in selected artifact-reporting modules.
- Replaced old extractor usage strings with package-module usage.
- Renamed selected generated output folders/files from `*_evidence` to `*_artifacts`.
- Patched:
  - `fpgai/reporting/training_native_accumulated_batch_artifacts.py`
  - `fpgai/reporting/training_accumulated_batch_artifacts.py`
  - `fpgai/reporting/training_multi_epoch_convergence_artifacts.py`
  - `fpgai/reporting/training_accelerator_artifacts.py`
  - `fpgai/reporting/memory_binding_artifacts.py`
  - `fpgai/reporting/parallel_policy_artifacts.py`
- No new files were added.

## Latest update

- Inspected `scripts/` reporting cleanup state.
- Confirmed `scripts/` contains only documentation:
  - `scripts/README.md`
  - `scripts/MANIFEST.md`
- No script files were deleted.
- Patched stale user-visible references in existing reporting/Vivado modules:
  - replaced old `python scripts/...` usage strings with package-module usage
  - replaced several public “evidence”/“sprint” labels with artifacts, reports, validation, or traceability wording
  - updated `claim_traceability` default output from `evidence/reproducibility` to `reports/reproducibility`
- This patch did not add new files.

## Latest update

- Updated `docs/CLI_WORKFLOWS.md` with public report CLI documentation.
- Used a safer section replacement that avoids embedded Markdown code fences inside the patch script.
- Documented:
  - `fpgai report build`
  - `fpgai report paper-artifacts`
  - `fpgai report frontier`
  - `fpgai report estimator`
- Clarified that report commands reuse `fpgai/reporting/` and generated outputs belong under `reports/`.

## Latest update

- Second report runtime smoke-test run reduced failures to one fixture-column mismatch.
- `paper-artifacts` also expected `hls_ok`.
- Patched the tiny `paper-artifacts` CSV fixture to include `hls_ok=True`.

## Latest update

- First report runtime smoke-test run found fixture-column mismatches, not CLI failures:
  - `paper-artifacts` expected `compile_ok`.
  - `frontier` expected `latency_seconds_min` or `latency_seconds_max`.
- Patched the tiny CSV test fixtures to match existing reporting module schemas.

## Latest update

- Added runtime smoke tests for public report subcommands using tiny CSV fixtures.
- Reused `tests/test_cli_report_validate.py`; no new test file was added.
- Covered:
  - `fpgai report paper-artifacts`
  - `fpgai report frontier`
  - `fpgai report estimator`
- Tests verify that each command exits successfully and writes expected output artifacts.

## Latest update

- Added tests for the new public `report` CLI subcommands.
- Reused existing `tests/test_cli_report_validate.py` instead of creating a new test file.
- Covered help paths for:
  - `fpgai report`
  - `fpgai report paper-artifacts`
  - `fpgai report frontier`
  - `fpgai report estimator`

## Latest update

- Inspected existing reporting modules and current `report` CLI wiring before adding anything.
- Reused existing reporting modules instead of creating new files:
  - `fpgai.reporting.generate_paper_artifacts`
  - `fpgai.reporting.paper_frontier`
  - `fpgai.reporting.estimator_tables`
- Wired them into the public CLI as:
  - `fpgai report paper-artifacts`
  - `fpgai report frontier`
  - `fpgai report estimator`
- Existing report logic remains in `fpgai/reporting/`; `fpgai/cli.py` only dispatches to it.

## Latest update

- Sprint 1 closeout check passed.
- Full pytest result after docs cleanup:
  - `258 passed, 1 skipped in 1.43s`
  - exit code `0`
- Remaining skipped test is optional HLS/Vitis CSim integration correctness when no testable YAML config is present.
- Remaining grep hits are acceptable generated-output/limitations/function-name references, not stale public sprint/cleanup workspace language.
- Sprint 1 is complete.

## Latest update

- Inspected public docs cleanup candidates before deleting anything.
- Deleted stale internal planning/cleanup docs that made the repository look like an unfinished sprint workspace:
  - `docs/REPO_CLEANUP_PLAN.md`
  - `docs/development_roadmap.md`
  - `docs/cli_workflows_legacy.md`
- Kept current user-facing docs:
  - `docs/CLI_WORKFLOWS.md`
  - `docs/CONFIG_FIRST_WORKFLOW.md`
  - `docs/PAPER_ARTIFACT_POLICY.md`
  - `docs/inspect_command.md`
  - `docs/logging.md`
- Reworded `docs/logging.md` to avoid calling FPGAI open-source while the license is academic/non-commercial research only.

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
