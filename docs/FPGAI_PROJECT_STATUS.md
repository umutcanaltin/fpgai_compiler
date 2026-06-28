# FPGAI Project Status

## Current sprint

Sprint 21: full real-pipeline sweep validation toward paper artifacts.

## Latest completed change

Fixed CNN inference HLS code generation.

Completed fixes:
- Tiled Conv HLS helper now supports flat parameter arrays emitted by the parameter generator.
- Stale default type aliases in tiled Conv helper were removed.
- Parameter emitter no longer emits file-scope `BIND_STORAGE` pragmas that Vitis HLS rejects during csynth.
- `configs/sweeps/inference_policy.yml` now uses real precision candidate `fx16_6` instead of stale `fixed`.

Validated:
- `inference_precision_single`: passed.
- `inference_precision`: 4/4 passed.
- `inference_policy`: 4/4 passed.
- `inference_policy_fx16_check`: 1/1 passed with no skipped/unapplied precision mode.

## Truth boundary

Supported after this sprint:
- CNN inference HLS generation through CSim and csynth for the tested sweeps.
- Precision and policy sweeps generate real HLS reports for tested CNN designs.

Still to validate:
- All remaining sweeps across precision, tiling, memory, parallelism, pipeline, training, and hardware knobs.
- Vivado report generation.
- Bitstream generation.
- Real board runtime inference/training timing and accuracy plots.

## Next step

Run full sweep validation in gates:
1. Inspect every sweep config.
2. Run every HLS/benchmark sweep.
3. Collect artifact sensitivity and HLS reports.
4. Run Vivado report/bitstream paths only after HLS sweeps are green.

## Sprint 23A precision materialization fix

- Found precision sweep bug: `precision_mode` was applied only to `analysis.precision_sweep.selected_candidate`.
- Generated/materialized configs kept `numerics.defaults` at `ap_fixed<16,6>` for all precision modes.
- Patched `fpgai/experiments/config_materializer.py` so `precision_mode` also materializes:
  - `analysis.precision_sweep.selected_candidate`
  - `numerics.precision_mode`
  - `numerics.defaults`
- Next validation: regenerate a small precision sweep and verify generated HLS types differ across `fx8_3`, `fx10_4`, `fx12_4`, `fx14_5`, `fx16_6`.

## Sprint 23B.3 precision layout activation validation

The precision layout report now correctly detects nonzero activation/input/output counts.

Validation sweep:

`paper_experiments/full_pipeline_gate/sweeps/precision_layout_report_fixcheck_v2`

For the same model, element counts are stable across precision modes:
- input elements: 784
- output elements: 10
- weight elements: 6798
- bias elements: 14
- activation-buffer elements: 14387

Precision-dependent byte counts now scale correctly.

`fx8_3`:
- input raw bytes: 784
- output raw bytes: 10
- weight raw bytes: 6798
- bias raw bytes: 28
- activation-buffer raw bytes: 14387
- input AXIS bytes: 784
- output AXIS bytes: 12
- weight AXI bytes: 6800
- activation AXI bytes: 14400

`fx16_6`:
- input raw bytes: 1568
- output raw bytes: 20
- weight raw bytes: 13596
- bias raw bytes: 42
- activation-buffer raw bytes: 28774
- input AXIS bytes: 1568
- output AXIS bytes: 20
- weight AXI bytes: 13600
- activation AXI bytes: 28784

Conclusion: precision now affects the compiler's central accounting for activations, weights, bias, accumulators, AXIS communication, AXI/DDR communication, and activation storage pressure.

Next implementation step: wire this precision layout into real generated HLS AXIS input/output packing, then DDR/runtime weight packing, then embedded BRAM packing.

## Sprint 24C-mini board-fit reporting and user guidance

Completed board-fit foundation for board-aware compiler decisions.

Implemented:
- Centralized board capacity metadata in `fpgai/backends/vivado/boards.py`.
- Added capacity fields for:
  - LUT
  - FF
  - BRAM_18K
  - URAM
  - DSP
  - DDR bytes
  - default/safe clock MHz
  - board part
  - FPGA part
  - PS type
  - overlay/deployment style
- Added reusable board-fit classification in `fpgai/reporting/hardware_feasibility.py`.
- Added board-fit statuses:
  - `fits`
  - `near_limit`
  - `over_limit`
  - `unknown`
- Added prediction-time board-fit artifacts:
  - `reports/board_fit.json`
  - `reports/board_fit.md`
- Added user-guiding Markdown output that explains:
  - selected board
  - limiting dimension
  - used vs available resources
  - utilization percentage
  - whether Vivado is allowed by fit
  - YAML actions to reduce the limiting dimension
- Added board-fit artifact links/status to normal compile CLI output.

Current behavior:
- Fabric/DDR resource overuse can produce `over_limit`.
- Clock above board safe/default guide rail produces `near_limit` when no hard max clock is known.
- Clock `near_limit` keeps `vivado_allowed=True` because real timing truth must come from Vitis HLS/Vivado timing reports.
- Prediction-based board fit is explicitly marked as not replacing HLS, Vivado implementation, timing, power, or real board runtime validation.

Validated:
- `pytest -q tests/test_vivado_bridge_boards.py tests/test_compiler_artifact_meta.py tests/test_cli_quiet_logging.py`
  - 30 passed.
- Fast compile smoke with HLS/Vivado disabled:
  - Config: `paper_experiments/full_pipeline_gate/smoke/board_fit_compile.yml`
  - Output: `paper_experiments/full_pipeline_gate/smoke/board_fit_compile`
  - Generated:
    - `reports/board_fit.json`
    - `reports/board_fit.md`
  - CLI summary showed:
    - Board fit JSON
    - Board fit summary
    - Board fit status: `near_limit`
    - Board fit limiting: `target_clock_mhz`
- Smoke board-fit normalized prediction resources:
  - LUT: 3822 / 117120 = 3.26% -> fits
  - FF: 4554 / 234240 = 1.94% -> fits
  - BRAM_18K: 12 / 288 = 4.17% -> fits
  - DSP: 18 / 1248 = 1.44% -> fits
  - target clock: 200 / safe 100 = near_limit
  - Vivado allowed by fit: true

Design rule established:
- User YAML board/clock/resource intent must be visible in compiler artifacts.
- Compiler must guide users toward board-correct designs instead of silently allowing impossible deployment.
- Manual YAML overrides remain allowed, but board-fit reports must expose their effect on selected board capacity.
- Board-fit prediction is an early gate/report; final deployability still requires HLS, Vivado, bitstream, and real board runtime reports.

Next step:
- Add `hardware.fit_policy` / `targets.platform.fit_policy` with modes:
  - `report_only`
  - `warn`
  - `enforce`
- Wire `fit_policy=enforce` into Vivado/bitstream flow so board `over_limit` designs are not launched into implementation as deployable candidates.
- Then add HLS/Vivado-resource refresh so board-fit can use real csynth/Vivado resources when available instead of only predictions.

## Sprint 24D YAML hardware knob contract

Completed YAML hardware-control traceability so policies are presets only and manual YAML decisions are visible.

Implemented:
- Added hardware knob contract artifacts:
  - `reports/hardware_knob_contract.json`
  - `reports/hardware_knob_contract.md`
- Added manifest entry:
  - `hardware_knob_contract`
- Added compile summary output:
  - Hardware knob contract: available
  - JSON path
  - Markdown summary path
  - knob count
  - manual YAML knob count
  - changed/clamped count
  - report-only count
- Added layer-aware tiling reporting:
  - Dense tiling is evaluated against Dense layers only.
  - Conv tiling is evaluated against Conv layers only.
  - If a knob is configured but no matching layer exists, status is `not_applicable`, not falsely `changed_or_clamped`.

Contract rule:
- `manual YAML override > policy preset > compiler default`

Reported statuses:
- `applied`
- `changed_or_clamped`
- `not_applicable`
- `not_requested`
- `report_only`
- `unknown`

Validated:
- `pytest -q tests/test_cli_quiet_logging.py tests/test_compiler_artifact_meta.py tests/test_dense_tiling_codegen.py tests/test_vivado_bridge_boards.py`
  - 45 passed.
- Hardware knob smoke:
  - Config: `paper_experiments/full_pipeline_gate/smoke/hardware_knob_contract.yml`
  - Output: `paper_experiments/full_pipeline_gate/smoke/hardware_knob_contract`
  - Generated:
    - `reports/hardware_knob_contract.json`
    - `reports/hardware_knob_contract.md`
  - CLI showed:
    - Hardware knob contract: available
    - Knob count: 18
    - Manual YAML knobs: 12
    - Changed/clamped: 1
    - Report-only: 1

Smoke result:
- Manual YAML overrides applied:
  - `optimization.parallel.pe = 3`
  - `optimization.parallel.simd = 2`
  - `optimization.parallel.unroll_factor = 4`
  - `optimization.parallel.partition_factor = 5`
  - `optimization.parallel.array_partition_mode = cyclic`
  - `optimization.pipeline.style = aggressive`
  - `optimization.pipeline.ii = 2`
- Dense tiling was `changed_or_clamped` because requested tile exceeded the toy Dense layer dimensions.
- Conv tiling was `not_applicable` because the smoke model has no Conv layer.
- `targets.platform.fit_policy = report_only` is visible but not yet enforced.

Truth boundary:
- This sprint proves YAML-to-planner/report traceability and visible CLI reporting.
- HLS traceability is proven only where generated macros, comments, template arguments, or pragmas expose the knob.
- Vivado and runtime truth still require real Vivado reports, bitstreams, and real board execution artifacts.

Next step:
- Implement `targets.platform.fit_policy` / `hardware.fit_policy` modes:
  - `report_only`
  - `warn`
  - `enforce`
- For `enforce`, board-fit `over_limit` must prevent Vivado/bitstream implementation from being treated as a deployable candidate.

## Sprint 24E board-aware policy templates

Completed first board-aware policy scaling.

Goal:
- Template policies are no longer blind fixed presets.
- The same policy name can produce different effective hardware knobs depending on selected board capacity.
- Manual YAML overrides still have priority and are not silently reduced.

Precedence:
- `manual_yaml_override`
- `board_aware_policy_scaling`
- `policy_preset`
- `compiler_default`

Implemented:
- Added board-aware policy scaling in planner.
- Added board tiering based on canonical Vivado board registry.
- Added small-board scaling for PYNQ-Z2:
  - Policy-generated `pe`, `simd`, `unroll_factor`, and `partition_factor` are capped to safer values.
- KV260/KR260 keep stronger preset values.
- Manual YAML overrides are detected and preserved.
- Compile-plan notes now record:
  - selected board
  - board tier
  - input policy
  - whether scaling happened
  - manual override paths
  - policy-resource-awareness changes/reasons
- Hardware knob contract now labels board-aware values as:
  - `board_aware_policy`
- Contract path helpers now support list-index paths like:
  - `targets.platform.clocks.0.target_mhz`
- Removed duplicate `_contract_status` definitions.
- Contract status now treats numeric equivalence such as `200` and `200.0` as `applied`.

Validated:
- Unit/source tests passed:
  - `tests/test_compiler_artifact_meta.py`
  - `tests/test_dense_tiling_codegen.py`
  - `tests/test_vivado_bridge_boards.py`
  - `tests/test_cli_quiet_logging.py`
  - Result: 50 passed.
- Real compile smoke:
  - `paper_experiments/full_pipeline_gate/smoke/policy_board_aware/pynq_z2.yml`
  - `paper_experiments/full_pipeline_gate/smoke/policy_board_aware/kv260.yml`

Smoke result:
- PYNQ-Z2 + `Latency-First`:
  - `optimization.parallel.pe = 2`, source `board_aware_policy`
  - `optimization.parallel.simd = 2`, source `board_aware_policy`
  - `optimization.parallel.unroll_factor = 2`, source `board_aware_policy`
  - `optimization.parallel.partition_factor = 2`, source `board_aware_policy`
- KV260 + `Latency-First`:
  - `optimization.parallel.pe = 4`, source `policy_preset`
  - `optimization.parallel.simd = 4`, source `policy_preset`
  - `optimization.parallel.unroll_factor = 4`, source `policy_preset`
  - `optimization.parallel.partition_factor = 4`, source `policy_preset`
- Manual clock config is correctly reported from:
  - `targets.platform.clocks.0.target_mhz`
  - requested `200`
  - effective `200.0`
  - source `manual_yaml`
  - status `applied`

Remaining:
- Current schema still requires a non-empty `targets.platform.clocks` list.
- Future sprint should allow a clock entry without `target_mhz`, so board-aware policy can select safe/default clock automatically while still preserving the named clock.
- Future sprint should connect `fit_policy: enforce` to Vivado/bitstream gating.

## Sprint 24F fit_policy enforcement

Completed operational `fit_policy` enforcement across main compile reports and the separate Vivado bridge runner.

Goal:
- `targets.platform.fit_policy` is no longer report-only.
- Over-limit board-fit predictions can block deployable hardware stages when requested.
- Manual aggressive YAML choices remain allowed, but deployment gating is truthful and explicit.

Supported policies:
- `report_only`
  - Records board-fit result.
  - Does not block.
- `warn`
  - Records board-fit result.
  - Emits warning state in `fit_policy_gate`.
  - Does not block.
- `enforce`
  - If board fit is `over_limit`, blocks deployable hardware stages:
    - `vivado_impl`
    - `bitstream`
    - `deployable_runtime_overlay`

Implemented in main compile:
- Added `fit_policy_gate` to `manifest.json`.
- Gate reads:
  - `targets.platform.fit_policy`
  - fallback `hardware.fit_policy`
  - fallback default `report_only`
- Gate uses `prediction_artifacts.board_fit`.
- Gate records:
  - policy
  - board-fit status
  - limiting dimension
  - Vivado allowed by board fit
  - over-limit boolean
  - blocked boolean
  - warning boolean
  - severity
  - blocked stages
  - reason
- CLI summary now includes `Fit policy gate`.

Implemented in hardware knob contract:
- `targets.platform.fit_policy` row now points to `fit_policy_gate` and Vivado/bitstream gating decision.
- Removed old wording that said enforcement would happen in a later sprint.

Implemented in Vivado bridge runner:
- `fpgai.backends.vivado.run_bridge` now reads compile `manifest.json`.
- If `fit_policy_gate.blocked=true` and `--run-vivado-impl` is requested:
  - it refuses Vivado implementation/bitstream execution,
  - writes `vivado_bridge/manifest.json`,
  - writes `vivado_bridge_run_artifacts.json`,
  - records `fit_policy_gate_blocked=true`,
  - records the blocking reason,
  - marks `vivado_ran=false`,
  - marks `vivado_ok=false`,
  - marks `bitstream_exists=false`,
  - marks `xsa_exists=false`.
- The automated Vivado bridge summary table now reports the gate reason instead of a misleading missing-HLS error.

Validated:
- Unit/source tests passed:
  - `tests/test_vivado_bridge_boards.py`
  - `tests/test_compiler_artifact_meta.py`
  - `tests/test_cli_quiet_logging.py`
  - Result: 39 passed.
- Main compile smoke:
  - `paper_experiments/full_pipeline_gate/smoke/fit_policy_gate/enforce_over_limit.yml`
  - Board: `pynq_z2`
  - Manual aggressive knobs: `pe=999`, `simd=999`, `unroll_factor=999`, `partition_factor=999`
  - `fit_policy=enforce`
  - Result:
    - board_fit status: `over_limit`
    - limiting dimension: `dsp`
    - `fit_policy_gate.blocked=true`
    - blocked stages: `vivado_impl`, `bitstream`, `deployable_runtime_overlay`
- Vivado bridge smoke:
  - Command requested `--run-vivado-impl`
  - Runner blocked before Vivado execution.
  - Bridge manifest:
    - `fit_policy_gate_blocked=true`
    - `vivado_synth_requested=true`
    - `vivado_impl_requested=true`
    - `bitstream_requested=true`
    - `vivado_ran=false`
    - `vivado_ok=false`
    - `bitstream_exists=false`
    - `xsa_exists=false`

Remaining:
- Add explicit smoke coverage for `fit_policy=warn` and `fit_policy=report_only`.
- Add optional strict schema enum validation for `targets.platform.fit_policy`.
- Future sprint should allow `targets.platform.clocks: [{name: ap_clk}]` without `target_mhz`, so board-aware policy can select safe/default clock automatically.

## Sprint 24G fit_policy mode coverage and enum validation

Completed full `fit_policy` coverage for all supported modes.

Goal:
- Prove `report_only`, `warn`, and `enforce` behavior explicitly.
- Ensure invalid `fit_policy` values are rejected by config validation.
- Keep deployment gating truthful and visible.

Implemented:
- Added direct `_fit_policy_gate` behavior tests:
  - `report_only + over_limit` -> `blocked=false`, `warning=false`, `severity=info`
  - `warn + over_limit` -> `blocked=false`, `warning=true`, `severity=warning`
  - `enforce + over_limit` -> `blocked=true`, `warning=false`, `severity=error`
- Added real smoke configs under:
  - `paper_experiments/full_pipeline_gate/smoke/fit_policy_modes/report_only.yml`
  - `paper_experiments/full_pipeline_gate/smoke/fit_policy_modes/warn.yml`
  - `paper_experiments/full_pipeline_gate/smoke/fit_policy_modes/enforce.yml`
- Verified all three compile manifests produce expected `fit_policy_gate`.
- Verified Vivado bridge behavior:
  - `report_only`: does not block Vivado bridge gate
  - `warn`: does not block Vivado bridge gate
  - `enforce`: blocks `--run-vivado-impl` before Vivado execution
- Added strict config validation for:
  - `targets.platform.fit_policy`
  - `hardware.fit_policy`
- Allowed values:
  - `report_only`
  - `warn`
  - `enforce`
- Invalid values such as `aggressive` are now rejected.

Validated:
- Tests passed:
  - `tests/test_config_truth.py`
  - `tests/test_compiler_artifact_meta.py`
  - `tests/test_cli_quiet_logging.py`
  - `tests/test_vivado_bridge_boards.py`
  - Result: 48 passed.
- Positive config load check passed for:
  - `report_only`
  - `warn`
  - `enforce`

Remaining:
- Sprint 24H should relax clock validation so a named clock without `target_mhz` is valid.
- Board-aware policy should then materialize default/safe board clock into the effective plan and hardware knob contract.

## Sprint 24H.2 KV260 named-clock and manual-clock validation

Completed board-aware clock default smoke coverage beyond PYNQ-Z2.

Goal:
- Prove name-only clock entries work for KV260 as well as PYNQ-Z2.
- Prove manual `target_mhz` still has priority over board-aware/default clock selection.

Validated smoke configs:
- `paper_experiments/full_pipeline_gate/smoke/clock_defaults/pynq_named_clock.yml`
- `paper_experiments/full_pipeline_gate/smoke/clock_defaults/kv260_named_clock.yml`
- `paper_experiments/full_pipeline_gate/smoke/clock_defaults/pynq_manual_clock_150.yml`

Results:
- PYNQ-Z2 named clock:
  - requested clock: `null`
  - effective clock: `100.0`
  - source: `board_aware_policy`
- KV260 named clock:
  - requested clock: `null`
  - effective clock: `100.0`
  - source: `board_aware_policy`
- PYNQ-Z2 manual clock:
  - requested clock: `150`
  - effective clock: `150.0`
  - source: `manual_yaml`

Contract confirmed:
- manual YAML clock > board-aware/default clock > policy preset > compiler default.

Validated:
- Clock smoke verification passed.
- Regression tests passed:
  - `tests/test_config_truth.py`
  - `tests/test_dense_tiling_codegen.py`
  - `tests/test_compiler_artifact_meta.py`
  - `tests/test_cli_quiet_logging.py`
  - `tests/test_vivado_bridge_boards.py`

Remaining:
- Consider renaming clock source from `board_aware_policy` to clearer `board_default`.
- Clean duplicate/awkward Sprint 24F status text in `docs/FPGAI_PROJECT_STATUS.md`.
- Next sprint: remove stale `.bak_*` compiler backup files from the tracked/professional tree if they are not intentionally tracked.

## Sprint 24I repo cleanup: tracked backup artifacts removed

Completed cleanup of tracked backup/temp files from the professional repo tree.

Goal:
- Remove stale `.bak_*` implementation backups from tracked source.
- Prevent future local backup/patch artifacts from appearing in git status.
- Keep the repo professional for public/open-source review.

Removed tracked backup files:
- `fpgai/backends/hls/emit/top_cpp.py.bak_axis_precision_bits_source`
- `fpgai/backends/hls/emit/top_cpp.py.bak_axis_precision_packing`
- `fpgai/backends/hls/emit/top_cpp.py.bak_axis_precision_packing_v2`
- `fpgai/backends/hls/emit/top_cpp.py.bak_axis_precision_rawcfg_signature`
- `fpgai/backends/hls/emit/top_cpp.py.bak_precision_comments`
- `fpgai/backends/hls/emit/types_h.py.bak_precision_typedef_source`
- `fpgai/backends/hls/testbench.py.bak_precision_axis_tb`
- `fpgai/benchmark/pipeline.py.bak_precision_aware_thresholds`
- `fpgai/benchmark/pipeline.py.bak_precision_aware_thresholds_v2`
- `fpgai/benchmark/pipeline.py.bak_precision_aware_thresholds_v3`
- `fpgai/engine/compiler.py.bak_emit_tb_rawcfg`
- `fpgai/engine/compiler.py.bak_emit_tb_rawcfg_v2`
- `fpgai/engine/compiler.py.bak_emit_top_rawcfg`

Implemented:
- Added ignore rules for local backup/patch artifacts:
  - `*.bak`
  - `*.bak_*`
  - `*.orig`
  - `*.rej`
  - `*.tmp`
  - `*~`
  - `sprint*_inspection.txt`
- Removed local `__pycache__` directories outside `.venv`.

Validated:
- Tracked backup/temp file audit now returns no results:
  - `git ls-files | grep -E '(\\.bak|\\.bak_|~$|\\.tmp$|\\.orig$|\\.rej$|__pycache__|\\.pyc$)'`
- Regression tests passed:
  - `tests/test_config_truth.py`
  - `tests/test_dense_tiling_codegen.py`
  - `tests/test_compiler_artifact_meta.py`
  - `tests/test_cli_quiet_logging.py`
  - `tests/test_vivado_bridge_boards.py`
  - Result: `65 passed`.

Remaining:
- Clean duplicate/awkward Sprint 24F text in `docs/FPGAI_PROJECT_STATUS.md`.
- Review generated smoke outputs and decide which configs/results should be tracked versus ignored.

## Sprint 25C/25D update — end-to-end hardware knob traceability

Completed:
- Fixed `memory.weight_storage` so manual YAML BRAM/URAM selection reaches:
  - hardware knob contract,
  - compile/memory planning,
  - inference HLS top source,
  - training HLS top source,
  - `fpgai_params.cpp` trace comments.
- Fixed training HLS storage binding path:
  - `top_train_cpp.py` now emits storage bindings for static training weights, biases, and gradients.
- Fixed layerwise precision materialization:
  - `emit_types_h()` now uses resolved per-op precision attributes from `resolve_layerwise_precision()` instead of silently forcing global defaults.
- Kept stronger precision-mode traceability:
  - precision sweep materialization records `analysis.precision_sweep.selected_candidate`, `numerics.precision_mode`, and `numerics.defaults`.
- Added strict contract/source audit:
  - `fpgai.devtools.contract_source_audit`
  - parses `hardware_knob_contract.json`,
  - fails on empty checks,
  - verifies contract values against canonical HLS source/trace artifacts.

Validated:
- Full pytest passed after fixes.
- Sprint 25 end-to-end audit generated four compile cases:
  - `inference_pynq_safe`
  - `inference_kv260_aggressive`
  - `training_kv260_safe`
  - `training_kv260_aggressive`
- Contract/source audit passed with non-empty checks:
  - 4 cases,
  - 4 checks per case,
  - `memory.weight_storage`,
  - `optimization.pipeline.ii`,
  - `optimization.parallel.partition_factor`,
  - `optimization.parallel.unroll_factor`.

Remaining truth boundary:
- Vitis HLS and Vivado are not available on this machine PATH, so real HLS/Vivado report validation is still not proven here.
- Runtime board validation is still not proven in this sprint.
- Next sprint should run the same audit on a machine with Vitis/Vivado and attach real `csynth`, Vivado utilization/timing, bitstream, and board runtime artifacts.

## Sprint 26G — Full HLS completion for inference and training

Completed:
- Fixed inference URAM HLS generation by disabling invalid BIND_STORAGE pragmas on embedded initialized W*/B* parameter arrays.
- Fixed training HLS C++ generation by replacing literal `\n` source text with real generated newlines.
- Fixed training HLS synthesis by disabling file-scope BIND_STORAGE pragmas and keeping explicit trace comments instead.
- Reran both training designs successfully:
  - `training_kv260_safe_fx16_6`
  - `training_kv260_aggressive_fx8_3`
- Both training designs now complete Vitis HLS csynth and emit top-level `deeplearn_csynth` reports.

Current truth boundary:
- Full Stage 2 HLS csynth coverage should now be available for all 20 paper-matrix designs.
- Embedded URAM/training storage requests are represented in plans, reports, contracts, and generated trace comments.
- Real URAM-resident embedded parameter/training state storage still requires a synthesis-safe runtime-loaded or local-buffer implementation.

Final Sprint 27F paper artifact counts:
- 20 total paper designs.
- 20/20 prediction rows.
- 20/20 Vitis HLS rows.
- 15 HLS-only rows.
- 4 Vivado bitstream/XSA-ready rows.
- 1 Vivado board-capacity rejected row.
- 4 Vivado power/report rows.
- 0 missing manifests.
- Regression tests updated to match the current regenerated paper matrix.
- Final targeted validation: 14 tests passed.

Next:
- Rebuild Stage 2 prediction-vs-HLS and full-csynth subset tables.
- Build HLS calibration summary over the full 20-design dataset.
- Add regression tests for:
  - no literal `\\n#pragma` in generated training HLS
  - no file-scope BIND_STORAGE in training HLS
  - training HLS artifact status reaches full csynth in smoke mode

## Sprint 26H / 26H.1 closed — Estimator calibration against Vitis HLS csynth

Status: closed.

Final estimator label:
`operator_structural_v4_inference_hls_sharing_training_problem_shared`

Final validation:
- 20/20 paper-matrix designs compile successfully.
- 20/20 designs have Vitis HLS `full_csynth` reports.
- Stage 2 prediction-vs-HLS table rebuilt.
- Stage 2 HLS calibration summary rebuilt.
- Boundary: results are Vitis HLS csynth comparisons, not Vivado implementation or real board-runtime measurements.

Final V5 calibration summary:
- all LUT mean APE: 30.80%
- all DSP mean APE: 31.85%
- all BRAM18 mean APE: 1.31%
- inference BRAM18 mean APE: 0.00%
- inference DSP mean APE: 30.75%
- training DSP mean APE: 41.70%

Implemented estimator fixes:
- logical training resource overhead model,
- training DSP saturation,
- embedded small-parameter BRAM realization model,
- top-level residual BRAM model,
- inference precision/lane-aware HLS sharing model,
- training isolation so inference-specific sharing does not damage training,
- training keeps problem-size sharing.

Known remaining estimator limitations:
- high-parallel x8 LUT/DSP overestimation,
- combined aggressive fx8 DSP overestimation,
- training LUT underestimation,
- aggressive training DSP still imperfect,
- HLS latency/cycle parsing is still missing in the final table.

Next sprint:
Sprint 26I — Vivado implementation subset.

## Sprint 27A started — Paper artifact index

Status: first milestone complete.

Generated tool:
- `fpgai/devtools/build_paper_artifact_index.py`

Generated paper-result artifacts:
- `paper_results/index/paper_artifact_index.csv`
- `paper_results/index/paper_artifact_index.json`
- `paper_results/index/paper_artifact_index.md`

Current paper artifact coverage:
- 20/20 paper-matrix designs have prediction artifacts.
- 20/20 paper-matrix designs have Vitis HLS csynth artifacts.
- 5 Vivado implementation subset designs produced bitstream/XSA.
- 1 Vivado implementation subset design was rejected by board capacity.
- 14 designs are currently HLS-only.

Current artifact-index summary:
- `designs=20`
- `hls_only=14`
- `vivado_impl_bitstream_ready=5`
- `vivado_board_capacity_rejected=1`

Important classification:
- `training_kv260_safe_fx16_6` is classified as `vivado_board_capacity_rejected`.
- Vivado failure class: `vivado_impl_failed_board_capacity_lut_overutilized`.

Next sprint:
Sprint 27B — parse numeric HLS/Vivado resource, timing, and power reports into paper-ready CSV files.

## Sprint 27B — Numeric paper-result extraction

Status: functional extraction complete.

Generated tool:
- `fpgai/devtools/build_paper_numeric_results.py`

Generated paper-result artifacts:
- `paper_results/parsed/prediction_numeric_results.csv`
- `paper_results/parsed/hls_numeric_results.csv`
- `paper_results/parsed/vivado_numeric_results.csv`
- `paper_results/parsed/paper_numeric_joined.csv`

Validated coverage:
- 20/20 designs have prediction LUT/DSP/BRAM18 numeric rows.
- 20/20 designs have HLS numeric rows.
- 20/20 designs have HLS worst-case latency cycles parsed.
- 5/5 successful Vivado implementation designs have Vivado numeric utilization rows.
- 5/5 successful Vivado implementation designs have Vivado estimated power rows.
- 1/1 Vivado board-capacity rejection has parsed capacity-failure numbers.

Important parsed failure:
- `training_kv260_safe_fx16_6`
  - failure class: `vivado_impl_failed_board_capacity_lut_overutilized`
  - required Slice LUTs: 133729
  - available Slice LUTs: 117120
  - utilization: 114.18%

Regression guard:
- `tests/test_paper_result_artifacts.py` validates artifact-index and numeric-result coverage when generated paper results are present.

Boundary:
- Vivado power rows are Vivado estimated power, not measured board power.
- Real board runtime, real board power, and real energy remain future validation sprints.

Next sprint:
Sprint 27C — paper tables generated from `paper_results/parsed/paper_numeric_joined.csv`.

## Sprint 27C — Generated paper tables

Status: functional table generation complete.

Generated tool:
- `fpgai/devtools/build_paper_tables.py`

Generated paper-result tables:
- `paper_results/tables/table_1_artifact_coverage.csv`
- `paper_results/tables/table_1_artifact_coverage.md`
- `paper_results/tables/table_2_prediction_vs_hls.csv`
- `paper_results/tables/table_2_prediction_vs_hls.md`
- `paper_results/tables/table_3_hls_vs_vivado.csv`
- `paper_results/tables/table_3_hls_vs_vivado.md`
- `paper_results/tables/table_4_knob_effects.csv`
- `paper_results/tables/table_4_knob_effects.md`
- `paper_results/tables/table_5_training_capacity.csv`
- `paper_results/tables/table_5_training_capacity.md`

Validated table coverage:
- artifact coverage table rows: 8
- prediction-vs-HLS table rows: 20
- HLS-vs-Vivado table rows: 6
- KV260 inference knob-effect table rows: 16
- training-capacity table rows: 2

Important paper boundaries:
- Vivado power values are estimated power from Vivado reports, not measured board power.
- Bitstream/XSA exists for 5 implementation subset designs.
- `training_kv260_safe_fx16_6` is rejected by board capacity, not silently failed.
- Full board runtime, measured power, and energy are not claimed yet.

Next sprint:
Sprint 27D — generated paper figures from the parsed numeric CSVs.

## Sprint 27D — ArXiv comparison tables

Status: pivoted from PDF figures to arXiv-ready comparison tables.

Decision:
- PDF figures are optional and not the main arXiv artifact.
- The arXiv version should prioritize compact LaTeX tables for comparison, traceability, and honest claim support.

Main outputs should be:
- LaTeX tables (`.tex`)
- Markdown previews (`.md`)
- CSV source tables (`.csv`)

Required arXiv comparison tables:
- artifact coverage table
- prediction-vs-HLS comparison table
- HLS-vs-Vivado implementation table
- design knob comparison table
- training capacity table
- claim-support / limitation table

Boundary:
- Vivado power values are estimated power from Vivado reports, not measured board power.
- Real board runtime, measured board power, and measured energy are not claimed yet.

Next sprint:
Sprint 27D.1 — generate arXiv-ready LaTeX comparison tables.

## Sprint 27F — Tiling and memory materialization audit

Status: tiling fixed and validated; memory/URAM limitation remains open.

Tiling fixes:
- Regenerated `paper_design_matrix.json` before `generate_paper_configs`.
- Fixed paper tiling matrix so small/medium/large remain distinct for the tiny paper MLP.
- Patched Dense tiling planner aliases so generated `tm/tn/tk` values are accepted.
- Dense mapping:
  - `tm` -> Dense output tile
  - `tk` / `tn` -> Dense input tile

Validated tiling chain:
- YAML/config tiling values:
  - small: `1x1x1`
  - medium: `2x2x2`
  - large: `4x4x4`
- Planner tiles:
  - small layer 0: `{'in': 1, 'out': 1}`
  - medium layer 0: `{'in': 2, 'out': 2}`
  - large layer 0: `{'in': 4, 'out': 4}`
- HLS source:
  - small: `dense_out_in_tiled<8, 4, 1, 1, ...>`
  - medium: `dense_out_in_tiled<8, 4, 2, 2, ...>`
  - large: `dense_out_in_tiled<8, 4, 4, 4, ...>`
- HLS source hashes differ across small/medium/large.
- Vitis HLS passed for all three tiling designs.

Measured HLS effect:
- small: LUT 3899, FF 3567, DSP 14, BRAM18 5, latency 93
- medium: LUT 4275, FF 3633, DSP 22, BRAM18 5, latency 86
- large: LUT 4305, FF 3706, DSP 24, BRAM18 5, latency 81

Claim boundary:
- Tiling can now be claimed as materially emitted into HLS source and reflected in HLS reports for the paper MLP.
- URAM is still not proven as real embedded-weight URAM storage. Existing BRAM/URAM memory comparison must remain limited or labeled as requested/not-inferred until runtime-loaded mutable URAM buffers are implemented.

Final Sprint 27F paper artifact counts:
- 20 total paper designs.
- 20/20 prediction rows.
- 20/20 Vitis HLS rows.
- 15 HLS-only rows.
- 4 Vivado bitstream/XSA-ready rows.
- 1 Vivado board-capacity rejected row.
- 4 Vivado power/report rows.
- 0 missing manifests.
- Regression tests updated to match the current regenerated paper matrix.
- Final targeted validation: 14 tests passed.

Next:
- Add and keep `tests/test_tiling_materialization.py`.
- Rebuild paper numeric artifacts/tables so tiling rows use the corrected HLS reports.
- Continue Sprint 27G for real URAM storage or explicit unsupported/report-only classification.

## Sprint 27F — Tiling materialization

Status: closed for Dense/Gemm and Conv tiling.

Dense/Gemm tiling validation:
- Paper matrix tiling values were changed to remain distinct for the tiny MLP:
  - small: `1x1x1`
  - medium: `2x2x2`
  - large: `4x4x4`
- Regenerated `paper_design_matrix.json` before `generate_paper_configs`.
- Patched planner Dense aliases so generated `tm/tn/tk` values are accepted.
- Dense mapping:
  - `tm` -> Dense output tile
  - `tk` / `tn` -> Dense input tile

Validated Dense chain:
- small source: `dense_out_in_tiled<8, 4, 1, 1, ...>`
- medium source: `dense_out_in_tiled<8, 4, 2, 2, ...>`
- large source: `dense_out_in_tiled<8, 4, 4, 4, ...>`
- Source hashes differ across small/medium/large.
- Vitis HLS passed for all three designs.

Measured Dense HLS effect:
- small: LUT 3899, FF 3567, DSP 14, BRAM18 5, latency 93
- medium: LUT 4275, FF 3633, DSP 22, BRAM18 5, latency 86
- large: LUT 4305, FF 3706, DSP 24, BRAM18 5, latency 81

Conv tiling fixes:
- Patched Conv planner aliases so generated `tm/tn/tk` values are accepted.
- Conv mapping:
  - `tm` -> output-channel tile `oc`
  - `tn` -> input-channel tile `ic`
  - `tk` -> spatial tiles `oh` and `ow`
- Added regression tests for generated `tm/tn/tk` Conv aliases and layer-specific overrides.
- Existing Conv tiling codegen tests pass.

Validated Conv chain:
- `configs/suite/cnn_no_pool.yml` compiled successfully.
- `CONV_COMPILE_RETURN=0`
- HLS ran and passed.
- Generated HLS source contains:
  - `// FPGAI real convolution tiling helper.`
  - `conv2d_tiled<28, 28, 1, 28, 28, 4, 3, 1, 1, 8, 8, 8, 2, ...>`
  - `conv2d_tiled<28, 28, 4, 28, 28, 4, 3, 1, 1, 8, 8, 8, 2, ...>`
- Generated type comments record:
  - `planner_tile: {'oh': 8, 'ow': 8, 'oc': 8, 'ic': 2}`
- Hardware knob contract reports `optimization.tiling.conv` as `applied`.

Measured Conv HLS result:
- LUT 93849
- FF 159193
- DSP 3462
- BRAM18 39
- URAM 0
- WorstLatency 141221

Validated tests:
- `PY_COMPILE_PLANNER_RETURN=0`
- `PY_COMPILE_TEST_RETURN=0`
- `PYTEST_TILING_RETURN=0`
- `PYTEST_CONV_TILING_RETURN=0`
- Latest targeted result: 9 tests passed.

Claim boundary:
- Dense/Gemm tiling and Conv tiling may now be claimed as materially emitted into generated HLS source and validated by Vitis HLS reports.
- URAM remains open: embedded initialized weights are not yet proven to infer real URAM storage.

Final Sprint 27F paper artifact counts:
- 20 total paper designs.
- 20/20 prediction rows.
- 20/20 Vitis HLS rows.
- 15 HLS-only rows.
- 4 Vivado bitstream/XSA-ready rows.
- 1 Vivado board-capacity rejected row.
- 4 Vivado power/report rows.
- 0 missing manifests.
- Regression tests updated to match the current regenerated paper matrix.
- Final targeted validation: 14 tests passed.

Next:
- Rebuild paper numeric artifacts/tables so tiling rows use the corrected Dense HLS reports.
- Continue Sprint 27G for real URAM storage or explicit unsupported/report-only classification.

