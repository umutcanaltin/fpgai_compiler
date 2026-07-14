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


## Sprint 27G.2 — Runtime package model-weight payloads

Implemented and validated runtime package support for model-weight payloads.

Validated behavior:
- BRAM / embedded weights:
  - compiler passes weights_mode=embedded into runtime package generation
  - runtime_weight_payload_required=false
  - runtime_weight_payload_present=false
  - no runtime_package/weights/weights.bin
  - no runtime_package/weights/weight_layout.json
- URAM / runtime-loaded weights:
  - compiler passes weights_mode=uram into runtime package generation
  - runtime_weight_payload_required=true
  - runtime_weight_payload_present=true
  - runtime_package/weights/weights.bin exists
  - runtime_package/weights/weight_layout.json exists
  - total_words=46
  - payload format is packed32
  - layout follows generated HLS parameter order W0, B0, W1, B1

Validated compiler-path configs:
- paper_experiments/full_pipeline_gate/sprint26_paper_matrix/configs/kv260_memory_uram.yml
- paper_experiments/full_pipeline_gate/sprint26_paper_matrix/configs/kv260_memory_bram.yml

Next sprint:
- Implement and validate DDR model-weight storage using the same runtime package payload path.
- DDR acceptance requires generated HLS weights_mem m_axi, runtime package weights payload/layout, HLS pass, and hardware knob contract applied status.

## Sprint 27G.3 — DDR model-weight backend

Implemented and validated DDR model-weight storage path.

Validated behavior:
- memory.storage.weights=ddr resolves to compiler weights_mode=ddr.
- Generated HLS top has:
  - const ap_uint<32>* weights_mem
  - m_axi weights_mem interface on bundle gmem_weights
  - Runtime DDR weight load path
  - fpgai_load_ddr_vector calls
- Runtime package has:
  - runtime_package/weights/weights.bin
  - runtime_package/weights/weight_layout.json
  - manifest runtime_weights.weights_mode=ddr
  - manifest required=true and present=true
  - total_words=46
- DDR uses the same packed32 runtime payload layout as URAM.
- DDR does not emit URAM BIND_STORAGE pragmas.
- Vitis HLS passed for kv260_memory_ddr.

Measured DDR HLS result:
- LUT 11750
- FF 6932
- DSP 34
- BRAM18 9
- URAM 0
- WorstLatency 190

Next:
- Clean up/normalize PS/PL input/output schema while keeping backward compatibility with data_movement.ps_pl and data_movement.pl_ps.

## Sprint 27G.4 — PS/PL data_movement schema normalization

Implemented and validated read-compatible normalization for the new PS/PL data_movement schema while preserving legacy paths.

Supported legacy schema:
- data_movement.ps_pl.input
- data_movement.pl_ps.output
- data_movement.ps_pl.weights.mode

Supported normalized schema:
- data_movement.input.load.interface
- data_movement.input.load.source
- data_movement.output.store.interface
- data_movement.output.store.destination
- data_movement.weights.load.interface
- data_movement.weights.load.source
- data_movement.weights.store.interface
- data_movement.weights.store.destination

Validated behavior:
- Compiler weight-mode resolver accepts data_movement.weights.load.interface.
- memory.storage.weights still has priority over data_movement weight transport.
- Communication planner accepts data_movement.input.load and data_movement.output.store.
- New schema input/output tensor names and size_words are used by communication planning.
- New schema DDR config compiles successfully.
- Generated HLS emits DDR weights_mem m_axi path.
- Runtime package emits weights.bin and weight_layout.json.
- Vitis HLS passed for kv260_memory_ddr_new_schema.

Validation:
- 16 focused tests passed.
- kv260_memory_ddr_new_schema compile returned 0.
- Runtime package manifest reports:
  - runtime_weights.weights_mode=ddr
  - required=true
  - present=true
  - total_words=46

Current memory + PS/PL status:
- BRAM model weights:
  - embedded/local HLS path
  - no runtime weight payload
  - HLS validated
- URAM model weights:
  - runtime-loaded local URAM storage
  - weights_mem m_axi import path
  - runtime package weight payload/layout
  - HLS validated
- DDR model weights:
  - external/runtime DDR weight path
  - weights_mem m_axi import path
  - runtime package weight payload/layout
  - HLS validated
- Legacy and normalized PS/PL schemas both compile.

Claim boundary:
- Safe to claim YAML-controlled BRAM/URAM/DDR model-weight storage affects generated HLS source, HLS reports, runtime package manifests, and tests.
- Safe to claim normalized PS/PL schema is backward compatible for current compiler/planner paths.
- Do not claim board runtime, measured board power, measured board energy, or real FPGA execution yet.

Next:
- Run full scenario/knob traceability audit:
  - YAML knob
  - compiler effective value
  - prediction artifact effect
  - generated HLS source effect
  - Vitis HLS actual report effect
  - runtime package effect
  - Vivado/report effect when available
  - status: applied / report_only / estimate_only / missing / unsupported

## Sprint 27H.1 — normalized schema report cleanup

Implemented normalized data_movement reporting for hardware knob contracts and design-space materialization metadata.

Validated behavior:
- New-schema DDR compile returns 0.
- Hardware knob contract now reports normalized data movement rows:
  - data_movement.input.load
  - data_movement.output.store
  - data_movement.weights.load.interface
  - data_movement.weights.store.interface
- Contract rows are marked applied when the normalized YAML path is present.
- Hardware knob contract increased from 18 to 22 rows for the new-schema DDR config.
- Design-space materialization metadata includes both legacy and normalized data_movement paths for backward compatibility.
- Normalized data_movement paths appear in:
  - manifest.json
  - estimate_vs_hls/results.json
  - design_space/results.json
  - reports/hardware_knob_contract.json
  - reports/hardware_knob_contract.md

Validation:
- kv260_memory_ddr_new_schema compile returned 0.
- HLS passed.
- Runtime package created.
- Runtime weights remain weights_mode=ddr, required=true, present=true.
- Focused runtime/memory tests pass.

Claim boundary:
- Safe to claim reports now expose normalized schema paths in the hardware knob contract.
- Safe to claim design-space materialization metadata knows both legacy and normalized data_movement paths.
- Still do not claim Vivado implementation or board runtime for this matrix unless those artifacts are generated.

Next:
- Generate the full knob/effect traceability table from current run artifacts.

## Sprint 27H.2 — run-level knob/effect traceability table

Generated and committed run-level artifact traceability tables:
- paper_results/knob_effect_traceability.csv
- paper_results/knob_effect_traceability.md

Validated scope:
- 22 paper-matrix runs summarized.
- 22/22 have hardware knob contracts.
- 22/22 have generated HLS source markers.
- 22/22 have Vitis HLS csynth reports.
- 4/22 have Vivado bridge/bitstream/project artifacts.
- 0/22 are claimed as real-board runtime measurements.

Key validated scenarios:
- BRAM model weights:
  - runtime_weights_mode=embedded
  - runtime payload not required
  - HLS URAM=0
- DDR model weights:
  - runtime_weights_mode=ddr
  - runtime payload required/present
  - total_words=46
  - HLS uses weights_mem m_axi
  - HLS URAM=0
- URAM model weights:
  - runtime_weights_mode=uram
  - runtime payload required/present
  - total_words=46
  - HLS uses weights_mem m_axi and URAM bind markers
  - HLS URAM=18
- Normalized schema DDR:
  - normalized_schema_reported=true
  - runtime_weights_mode=ddr
  - HLS/runtime behavior matches DDR legacy config

Claim boundary:
- Safe to claim generated HLS source and Vitis HLS csynth effects for all 22 rows.
- Safe to claim Vivado artifacts only for the 4 rows marked Vivado=true in the traceability table.
- Do not claim real-board execution, measured power, measured energy, or board accuracy from this table.

Next:
- Generate per-knob traceability table:
  - knob
  - values tested
  - prediction effect
  - generated HLS source effect
  - Vitis HLS actual effect
  - runtime package effect
  - Vivado effect
  - claim status

## Sprint 27H.4 — YAML to generated HLS C++ traceability audit

Generated YAML-to-HLS-C++ traceability artifacts:
- paper_results/yaml_to_hls_cpp_traceability.csv
- paper_results/yaml_to_hls_cpp_traceability.md

Validated scope:
- 198 traceability records generated.
- 0 records marked check_needed.
- Every checked major knob is classified as applied, applied_or_not_applicable, not_required, or report_only_or_backend.

Validated logical effects:
- Pipeline knobs:
  - optimization.pipeline.ii maps to generated PIPELINE pragmas.
  - HLS csynth latency/resource results are recorded per run.
- Parallel knobs:
  - optimization.parallel.unroll_factor maps to generated UNROLL pragmas.
  - optimization.parallel.partition_factor maps to generated ARRAY_PARTITION pragmas.
  - HLS csynth resource/latency differences are recorded per run.
- Tiling knobs:
  - optimization.tiling.dense maps to dense_out_in_tiled markers.
  - optimization.tiling.conv is applied for Conv/training rows and marked not applicable for non-Conv rows.
- Memory storage:
  - BRAM/embedded rows do not expose external weights_mem and do not allocate URAM.
  - DDR rows expose weights_mem and m_axi port=weights_mem, and HLS URAM remains 0.
  - URAM rows expose weights_mem, impl=uram, and HLS URAM is greater than 0.
- Data movement:
  - AXIS/AXI interface markers are present.
  - Normalized DDR schema maps to the same generated DDR HLS path as legacy DDR.
- Precision:
  - Parameter artifacts are present.
  - Further precision-specific comparison should inspect generated typedef/parameter widths directly across fx8/fx12/fx16 rows.
- Board:
  - targets.platform.board is treated as report/backend target metadata unless Vivado/bitstream artifacts are generated.

Claim boundary:
- Safe to claim YAML knobs now have artifact-level traceability to generated HLS C++ markers and/or explicit report/backend-only boundaries.
- Safe to claim memory BRAM/DDR/URAM choices are logically different in generated C++ and HLS csynth reports.
- Still do not claim real-board execution, measured power, measured energy, or board accuracy.

Next:
- Add a precision-width specific audit to compare generated typedef/parameter widths across fx8/fx12/fx16 rows.
- Then move traceability generation into the existing reporting library so the tables are reproducible from the CLI.

## Sprint 28B/28C — Memory semantics cleanup

Problem:
- The Sprint 27J validation passed generated-code checks, but the memory rows were not safe to describe as a pure BRAM-vs-DDR-vs-URAM comparison.
- The current DDR implementation loads runtime weights from `weights_mem` into full local W/B arrays before compute.
- Therefore current DDR is `ddr_preload_full`, not scalable `ddr_tiled`.

Implemented:
- Added memory semantics classifier:
  - `embedded_constants`
  - `ddr_preload_full`
  - `uram_preload_full`
  - `ddr_tiled`
  - `invalid_or_ambiguous`
- Added tests for generated Sprint 27J memory rows.
- Generated paper-safe memory and precision tables under:
  - `paper_results/sprint28c/memory_semantics_table.md`
  - `paper_results/sprint28c/precision_effect_table.md`
  - `paper_results/sprint28c/paper_claim_boundary.md`

Validated classifications:
- `kv260_memory_bram`: `embedded_constants`
- `kv260_memory_ddr`: `ddr_preload_full`
- `kv260_memory_ddr_new_schema`: `ddr_preload_full`
- `kv260_memory_uram`: `uram_preload_full`
- precision rows: `embedded_constants`

Claim boundary:
- Safe: embedded constants, DDR preload-full, URAM preload-full, generated HLS, runtime packages, and Vitis HLS reports.
- Not safe: scalable DDR-resident/tiled large-network execution.
- Not safe: pure BRAM-vs-DDR-vs-URAM storage-only comparison.

## Sprint 29A draft — storage/import/export movement semantics

Implemented in the uploaded ZIP working copy:
- Public weight storage contract moved toward `bram`, `uram`, and `ddr` only.
- Data movement direction names now support `import` and `export` in addition to legacy `load`/`store` aliases.
- Input/output movement now supports the new schema:
  - `data_movement.inputs.import`
  - `data_movement.outputs.export`
  - `interface`, `transport`, and `policy` fields.
- Weight movement resolver now separates:
  - storage location
  - import/export capability
  - HLS backend compatibility mode
  - runtime command semantics.
- `m_axi/full` is now represented as an import/export capability, not as automatic reload before every compute.
- Runtime package weight metadata now reports canonical modes such as:
  - `bram_static`
  - `bram_import_full`
  - `bram_import_export_full`
  - `uram_import_full`
  - `uram_import_export_full`
  - `ddr_tiled` / `ddr_tiled_mutable` as future modes.
- Runtime package now records:
  - `import_required`
  - `export_supported`
  - `reload_before_each_compute: false`.
- DDR storage now rejects until true DDR-tiled Dense/Conv HLS exists. Full import is no longer treated as DDR storage; it belongs to BRAM/URAM storage with `m_axi/full` import.
- Training URAM/DDR rejection now happens before fake generated HLS/runtime artifacts.
- `fpgai/__init__.py` and `fpgai/engine/__init__.py` now use lazy exports so runtime/report utility tests can import without pulling heavy ONNX compiler dependencies.

Validation in this sandbox:
- Passed:
  - `tests/test_memory_storage_effect.py`
  - `tests/test_runtime_package.py`
  - `tests/test_memory_semantics_classifier.py`
  - training reject tests for URAM and DDR.
- Command:
  - `python -m pytest -q tests/test_memory_storage_effect.py tests/test_runtime_package.py tests/test_memory_semantics_classifier.py tests/test_training_memory_storage_contract.py::test_training_uram_weight_storage_is_rejected_until_real_hls_backend_exists tests/test_training_memory_storage_contract.py::test_training_ddr_weight_storage_is_rejected_until_real_hls_backend_exists`
  - Result: `18 passed, 5 skipped`.
- `test_training_bram_weight_storage_still_compiles` was not runnable in this sandbox because ONNX is not installed. It should be run in the repo venv.

Truth boundary after this draft:
- This sprint updates compiler/report/runtime contracts and rejects impossible DDR/URAM training/storage cases honestly.
- It does not yet implement true Dense DDR-tiled HLS.
- It does not yet implement true `uram_static` compile-time initialized URAM arrays.
- Existing compile-time constants without detected BRAM/URAM binding are classified as `legacy_compile_time_constants`, not falsely as exact BRAM/URAM storage.

Next implementation step:
- Implement real BRAM/URAM static generated HLS storage or keep exact rejection/reporting until those generated arrays are synthesis-safe and classifier-detectable.
- Then implement Dense `ddr_tiled` in existing `top_cpp.py` path.

## Sprint 29B — BRAM/URAM static weight storage draft

Implemented in the inspected ZIP copy.

Semantics:

- `memory.storage.weights=bram` with `data_movement.weights.import=compile_time/static` resolves to `bram_static`.
- `memory.storage.weights=uram` with `data_movement.weights.import=compile_time/static` resolves to `uram_static`.
- Generated HLS keeps initial parameter values as compile-time constants in `fpgai_params.cpp`, then creates function-scope static `W*/B*` arrays inside `deeplearn.cpp` and binds them using `#pragma HLS BIND_STORAGE` with `impl=bram` or `impl=uram`.
- Static BRAM/URAM modes require no runtime weight payload and do not reload weights before every compute.

Validation run in sandbox:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py::test_training_uram_weight_storage_is_rejected_until_real_hls_backend_exists \
  tests/test_training_memory_storage_contract.py::test_training_ddr_weight_storage_is_rejected_until_real_hls_backend_exists \
  tests/test_memory_semantics_classifier.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py
```

Result: `23 passed, 5 skipped`.

## Sprint 29C — Dense DDR-tiled inference draft

Implemented in the inspected ZIP copy.

Semantics:

- Public `memory.storage.weights=ddr` with `data_movement.weights.import=m_axi/tiled` now resolves to `ddr_tiled`.
- Dense DDR-tiled inference generates an HLS `weights_mem` m_axi port.
- Dense compute calls are rewritten to `dense_out_in_ddr_tiled`, which imports tile-sized weight buffers from `weights_mem` during compute.
- The top source no longer uses `fpgai_load_ddr_vector` or full local W/B arrays for `ddr_tiled`.
- Runtime package creates the packed weight payload from `*_init` arrays and records `weights_mode=ddr_tiled`, `import_required=true`, and `reload_before_each_compute=false`.
- Conv with `ddr_tiled` is still rejected clearly until Conv DDR-tiled codegen is implemented.

Validation run in sandbox:

```bash
python -m pytest -q \
  tests/test_memory_storage_effect.py \
  tests/test_memory_semantics_classifier.py \
  tests/test_runtime_package.py \
  tests/test_training_memory_storage_contract.py::test_training_uram_weight_storage_is_rejected_until_real_hls_backend_exists \
  tests/test_training_memory_storage_contract.py::test_training_ddr_weight_storage_is_rejected_until_real_hls_backend_exists \
  tests/test_mixed_precision_codegen.py::test_weight_delivery_mode_changes_generated_hls_interfaces \
  tests/test_mixed_precision_codegen.py::test_weight_storage_binding_changes_generated_hls_pragmas
```

Result: `29 passed, 5 skipped`.

Truth boundary after this draft:

- Dense DDR-tiled inference source generation is implemented structurally.
- Conv DDR-tiled inference is not implemented yet and rejects.
- Training DDR-tiled mutable import/update/export is not implemented yet and rejects through the training storage contract.
- HLS csynth/Vivado validation for a large Dense model still needs a follow-up experiment sprint.

## Sprint 29D — BRAM/URAM runtime import/export command modes draft

Implemented in the inspected ZIP copy.

Semantics:

- `bram_import_full` and `uram_import_full` now generate explicit runtime command modes instead of importing weights unconditionally before every compute.
- Generated HLS exposes `mode` on AXI-Lite for runtime-imported BRAM/URAM weights.
- `FPGAI_MODE_IMPORT_WEIGHTS` imports the full runtime payload from `weights_mem` into local BRAM/URAM arrays and returns.
- `FPGAI_MODE_RUN_INFERENCE` computes with already-imported local weights.
- `FPGAI_MODE_EXPORT_WEIGHTS` is emitted only for `bram_import_export_full` and `uram_import_export_full`, and copies current local weights back to `weights_mem`.
- Runtime-imported BRAM/URAM modes create function-scope static `W*/B*` arrays and bind them with `impl=bram` or `impl=uram`.
- `reload_before_each_compute=false` remains the contract: repeated inference can reuse imported weights until the runtime explicitly imports again.

Validation run in sandbox:

```bash
python -m pytest -q \
  tests/test_memory_storage_effect.py \
  tests/test_memory_semantics_classifier.py \
  tests/test_runtime_package.py \
  tests/test_training_memory_storage_contract.py::test_training_uram_weight_storage_is_rejected_until_real_hls_backend_exists \
  tests/test_training_memory_storage_contract.py::test_training_ddr_weight_storage_is_rejected_until_real_hls_backend_exists
```

Result: `32 passed, 5 skipped`.

Truth boundary after this draft:

- BRAM/URAM import/export command hardware paths are implemented for inference HLS source generation.
- Export returns the current local weights; it becomes training-meaningful after mutable training updates are implemented.
- Training URAM/DDR mutable storage remains rejected until `top_train_cpp.py` has real mutable load/update/export support.
- Conv DDR-tiled inference remains rejected until the Conv tiled backend is implemented.

## Sprint 29E — Input/output m_axi full movement

Implemented input/output array data movement for full `m_axi` import/export in the existing HLS top emitter.

Supported generated HLS behavior:

- `data_movement.inputs.import.interface=m_axi`, `policy=full` rewrites the top input from `in_stream` to `const ap_uint<32>* input_mem` and imports the full input array from DDR/runtime memory.
- `data_movement.outputs.export.interface=m_axi`, `policy=full` rewrites the top output from `out_stream` to `ap_uint<32>* output_mem` and exports the full output array to DDR/runtime memory.
- Existing `axi_stream + dma + full` behavior remains the default.
- Tiled input/output movement is intentionally rejected until real tiled I/O codegen is implemented.

Runtime package manifests now record `runtime_io.inputs.import` and `runtime_io.outputs.export` summaries.

## Sprint 29F — Conv DDR-tiled inference

Implemented Conv DDR-tiled inference source generation in the existing HLS top emitter.

Supported generated HLS behavior:

- `memory.storage.weights=ddr` with `data_movement.weights.import.interface=m_axi`, `policy=tiled` now supports Conv as well as Dense inference.
- Generated HLS exposes `weights_mem` as an `m_axi` port for DDR-resident weights.
- Conv calls are rewritten to `conv2d_ddr_tiled`.
- Conv DDR-tiled code allocates tile-sized `conv_weight_tile[TILE_OC][TILE_IC][K][K]`, input tile, and accumulator tile buffers.
- The full Conv weight tensor is not copied into a local W/B array.
- `fpgai_load_ddr_vector` is not used for DDR-tiled Conv; weights are read by tile from `weights_mem` during compute.
- Mixed Conv + Dense DDR-tiled networks keep a single packed `weights_mem` layout with correct per-layer weight/bias offsets.

Truth boundary after this draft:

- Conv and Dense DDR-tiled inference source generation is implemented structurally.
- HLS csynth/Vivado validation for large Conv/Dense DDR-tiled designs still needs a follow-up experiment sprint.
- Tiled input/output movement remains rejected until real tiled I/O codegen is implemented.
- Training URAM/DDR mutable storage remains rejected until `top_train_cpp.py` has real mutable load/update/export support.

## Sprint 29G — activation storage modes

Implemented first-class activation storage semantics for generated inference HLS:

- `memory.storage.activations: bram` resolves to `activation_bram`.
- `memory.storage.activations: uram` resolves to `activation_uram` on URAM-capable boards.
- URAM activation storage rejects on boards with zero URAM.
- Generated `deeplearn.cpp` rewrites local activation buffer `BIND_STORAGE` pragmas for `layer_in` and `layer_*_out` to the requested BRAM/URAM implementation.
- `memory_plan.notes` and `runtime_package/package_manifest.json` now report activation storage separately from weight storage.
- The generated-source memory classifier reports activation buffer bindings so DDR-tiled weight designs can still explain BRAM/URAM use from activations/tile buffers.

Still unsupported: activation DDR, activation LUTRAM/register storage, activation streaming/tiled I/O, training URAM/DD tiled mutable storage.

## Sprint 29H — training BRAM mutable storage and testbench comparison contract

Implemented training BRAM mutable storage in the existing training HLS emitter/testbench path.

Supported generated HLS behavior:

- `memory.storage.weights=bram` training weights are mutable HLS arrays used by forward, backward, and SGD update code.
- Generated training source now inserts real `#pragma HLS BIND_STORAGE ... impl=bram` pragmas for `W_*`, `B_*`, `dW_*`, and `dB_*` arrays instead of comment-only storage claims.
- Training command mode names are emitted in source:
  - `FPGAI_MODE_EXPORT_WEIGHTS_STREAM = 1` keeps the existing C-simulation/testbench weight snapshot path.
  - `FPGAI_MODE_RUN_TRAINING = 2` runs the forward/backward/update step.
  - `FPGAI_MODE_IMPORT_WEIGHTS = 3` and `FPGAI_MODE_EXPORT_WEIGHTS = 4` are emitted for BRAM `m_axi/full` import/export configurations.
- BRAM `m_axi/full` import/export training configurations now expose `weights_mem` as an `m_axi` port and implement explicit import/export command branches.
- Import/export is command-driven; weights are not reloaded before every training or inference call.

Testbench / Python-reference validation contract:

- The existing training C-simulation testbench path is preserved.
- The testbench still emits `weights_before.bin`, `grads.bin`, and `weights_after.bin`.
- The compiler still runs the Python training reference and compares HLS/C-simulation artifacts against:
  - `weights_before_ref.bin`
  - `grads_ref.bin`
  - `weights_after_ref.bin`
- For BRAM `m_axi/full` import configurations, the generated testbench packs the runtime weight payload into `weights_mem` and calls the training top with the m_axi pointer.

Validation run in sandbox:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py
```

Result in sandbox: `42 passed, 10 skipped`.

Truth boundary after this draft:

- Training BRAM mutable storage and BRAM import/export command source generation are implemented structurally.
- Full local HLS C-simulation with Python comparison requires the repo environment with `onnx` and Vitis HLS available; sandbox tests skip ONNX-dependent compile runs.
- Training URAM mutable storage and training DDR tiled mutable storage remain rejected until implemented.

## Sprint 29I — training URAM mutable storage

Implemented training URAM mutable storage in the existing training HLS emitter/testbench path.

Supported generated HLS behavior:

- `memory.storage.weights=uram` training weights are now allowed on URAM-capable boards.
- URAM training rejects on boards with zero URAM blocks.
- Generated training source inserts real `#pragma HLS BIND_STORAGE ... impl=uram` pragmas for mutable `W_*` and `B_*` arrays.
- Gradient arrays can remain in BRAM via `training.storage.gradients=bram`, so URAM weights and BRAM gradients are represented separately.
- URAM `m_axi/full` import/export configurations expose `weights_mem` as an `m_axi` port and implement explicit command branches:
  - `FPGAI_MODE_IMPORT_WEIGHTS`
  - `FPGAI_MODE_RUN_TRAINING`
  - `FPGAI_MODE_EXPORT_WEIGHTS`
- Import/export is command-driven; weights are not reloaded before every training call.

Testbench / Python-reference validation contract:

- The existing training C-simulation comparison path is preserved.
- The testbench still emits `weights_before.bin`, `grads.bin`, and `weights_after.bin`.
- The compiler still compares those against Python reference artifacts when the full ONNX/Vitis-capable environment is available.

Validation run in sandbox:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py
```

Result in sandbox: `44 passed, 10 skipped`.

Truth boundary after this draft:

- Training BRAM and URAM mutable storage are implemented structurally with real HLS storage bindings and testbench/Python comparison artifacts.
- Training DDR tiled mutable storage remains rejected until the external tiled load/update/export backend is implemented.
- Optimizer-state DDR movement and tiled input/output movement remain unsupported until their dedicated sprints.


## Sprint 29J — Training DDR tiled mutable Dense-only

Implemented a first real generated-HLS path for training `memory.storage.weights=ddr` with `data_movement.weights.import/export` set to `m_axi/tiled`. The training top now exposes `weights_mem`, emits DDR-tiled training markers, tile buffers (`weight_tile`, `grad_tile`), imports Dense parameter tiles from DDR before local training updates, and exports updated parameter tiles back to DDR after the update. The testbench runtime mode now treats `ddr_tiled` and `ddr_tiled_mutable` as m_axi weight-runtime modes so C-simulation continues to write `weights_before.bin`, `grads.bin`, and `weights_after.bin` for Python-reference comparison.

Boundary: this sprint is Dense-only for training DDR tiled. Conv or unsupported training ops must remain rejected until a Conv tiled backward/update backend exists.

## Sprint 29K — Training Conv DDR tiled mutable

Implemented the Conv extension of the training DDR tiled mutable path. The training DDR backend now accepts Dense/Conv graphs, keeps `weights_mem` as the DDR source of truth, emits Conv-specific tile buffers (`conv_weight_tile`, `conv_grad_tile`), and preserves the existing training testbench/Python comparison artifacts (`weights_before.bin`, `grads.bin`, `weights_after.bin`). Unsupported non-Dense/non-Conv training operators still reject clearly.

Validation in sandbox:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py
```

Result in sandbox: `44 passed, 11 skipped` because ONNX-dependent compile tests skip here. Local repo with ONNX should run those compile/testbench paths.

## Sprint 29L — Input/output m_axi tiled movement

Implemented `m_axi/tiled` input/output movement in the existing inference HLS top emitter and runtime manifest path.

Supported generated HLS behavior:

- `data_movement.inputs.import.interface=m_axi` with `policy=tiled` rewrites the top-level input port to `const ap_uint<32>* input_mem`.
- `data_movement.outputs.export.interface=m_axi` with `policy=tiled` rewrites the top-level output port to `ap_uint<32>* output_mem`.
- Generated source emits tile buffers (`input_tile`, `output_tile`) and tile-sized transfer loops.
- Existing AXI-stream/DMA full input/output behavior remains the default.
- Existing `m_axi/full` input/output behavior remains supported.
- AXI-stream tiled movement is still rejected until a real stream-tiled backend is implemented.

Runtime package behavior:

- `runtime_io.inputs.import.resolved=m_axi_import_tiled` for tiled m_axi input imports.
- `runtime_io.outputs.export.resolved=m_axi_export_tiled` for tiled m_axi output exports.

Truth boundary:

- This sprint implements tiled external transfer/staging for input/output arrays in the inference top path.
- Deeper compute-fused activation tiling and training tiled input/output remain future dedicated work.

## Sprint 29M — User-facing weight mode expansion

Implemented high-level user intent expansion for weight movement without changing the public storage vocabulary.

Public storage remains:

- `memory.storage.weights=bram`
- `memory.storage.weights=uram`
- `memory.storage.weights=ddr`

`weights.mode` is now an intent shortcut, not a storage location:

- `weights.mode=embedded` expands to compile-time/static import and no export for BRAM/URAM storage.
- `weights.mode=import` expands to command-driven `m_axi/full` import and no export for BRAM/URAM storage.
- `weights.mode=import_export` expands to command-driven `m_axi/full` import/export for BRAM/URAM storage.
- `weights.mode=tiled` expands to `m_axi/tiled` DDR external-memory execution.
- `weights.mode=tiled_mutable` expands to `m_axi/tiled` DDR import/export for `pipeline.mode=training_on_device`.

Priority rule:

```text
manual detailed data_movement.weights import/export
  > weights.mode
  > training.weight_initialization.mode
  > compiler defaults
```

Training initialization shortcut:

- `training.weight_initialization.mode=compile_time` expands to static compile-time initialization.
- `training.weight_initialization.mode=import` expands to command-driven runtime import.
- `zero`, `xavier`, `he`, and random seeded initialization modes reject clearly until implemented.

Truth boundary:

- `embedded` is not reintroduced as a storage option; it is only a user-facing mode.
- DDR still means tiled external-memory execution, not a full preload into BRAM/URAM.
- `m_axi/full` import remains command-capable import; it does not reload weights before every compute.
- Detailed `data_movement` remains the source of truth when present.

Validation in sandbox:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py
```

Result in sandbox: `53 passed, 11 skipped` because ONNX/environment-dependent compile tests skip here.

## Sprint 29N — Build/artifact stage selection

Implemented a user-facing `build.stages` contract without adding a detached codegen backend.

Supported YAML shape:

```yaml
build:
  stages:
    cpp: true
    testbench: true
    hls_project: false
    hls_synthesis: false
    vivado_project: false
    vivado_implementation: false
    bitstream: false
    runtime_package: true
    reports: true
```

Behavior:

- `cpp=true` generates existing HLS-compatible C++ source/include artifacts through the existing HLS emit path.
- `testbench=true` emits the existing C++ testbench artifacts and requires `cpp=true`.
- `hls_project=false` keeps C++-only mode and does not emit `hls/run_hls.tcl`.
- `hls_synthesis=true` requires `hls_project=true` and gates Vitis HLS execution.
- `runtime_package=true` emits `runtime_package/package_manifest.json` and records selected build stages.
- `reports=false` skips best-effort HLS report metadata emission.
- Vivado/bitstream stages are recorded and dependency-validated, but actual Vivado bridge execution remains in the dedicated Vivado bridge flow.

Dependency validation:

```text
testbench          requires cpp
hls_project        requires cpp
hls_synthesis      requires hls_project
vivado_project     requires hls_synthesis or build.existing_hls_ip=true
vivado_implementation requires vivado_project
bitstream          requires vivado_project and vivado_implementation
```

Legacy configs without `build.stages` keep prior behavior:

- HLS source/project generation follows `backends.hls.enabled`.
- Vitis HLS execution follows `toolchain.vitis_hls.enabled`.
- Host C++ generation follows `backends.host_cpp.enabled`.
- Runtime package and report metadata remain enabled by default.

Validation in sandbox:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py \
  tests/test_build_stages.py
```

Result in sandbox: `60 passed, 12 skipped` because ONNX/environment-dependent compile tests skip here.

## Sprint 29O/29P/30E bundle

Implemented as one validation-saving bundle:

- Sprint 29O runtime command sequence selection.
  - Adds `runtime.sequence` with command validation against generated memory/pipeline semantics.
  - Emits `reports/runtime_sequence.json`, `reports/runtime_sequence.md`, and `runtime_package/run_sequence.json`.
  - Records runtime sequence in `manifest.json` and `runtime_package/package_manifest.json`.
- Sprint 29P generated C++ readability modes.
  - Adds `codegen.readability` values: `compact`, `normal`, `high`, `debug`.
  - Default is `high` and generated HLS C++ now includes an honest resolved-decision banner.
- Sprint 30E resolved config/config contract foundation.
  - Emits `reports/resolved_config.json`, `reports/resolved_config.yml`, and `reports/config_contract.json`.
  - Records build stages, runtime sequence, codegen readability, memory notes, and communication edges.

Validation command used:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py \
  tests/test_build_stages.py
```

Observed sandbox result: `62 passed, 12 skipped`.

## Sprint 29Q/29R/30A bundle

Implemented as one validation-saving bundle:

- Sprint 29Q training tiled input/output/label movement foundation.
  - Adds training `m_axi` tiled interfaces for `inputs.import`, `labels.import`, and `outputs.export`.
  - Generated training HLS source exposes `input_mem`, `label_mem`, and `output_mem` m_axi ports when requested.
  - Emits `reports/training_io_movement.json` and `reports/training_io_movement.md`.
  - Preserves the existing training Python/testbench comparison artifacts.
- Sprint 29R gradient export full mode.
  - Supports `data_movement.gradients.export.interface=m_axi` with `policy=full`.
  - Generated training HLS source exposes `gradients_mem` and `FPGAI_MODE_EXPORT_GRADIENTS`.
  - Emits `reports/gradient_export.json` and `reports/gradient_export.md`.
  - Tiled gradient export rejects clearly until implemented.
- Sprint 30A classifier/report extension foundation.
  - Runtime sequence support now enables `export_gradients` only when generated gradient export is present.
  - The runtime package records the supported command in `runtime_sequence.supported_commands`.

Validation command used:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py \
  tests/test_build_stages.py \
  tests/test_layer_registry_and_compatibility.py \
  tests/test_training_movement_gradient_export.py
```

Observed sandbox result: `64 passed, 16 skipped`.

## Sprint 29S/29T/29U bundle

Implemented as one validation-saving bundle:

- Sprint 29S optimizer-state storage/movement foundation.
  - Adds `training.storage.optimizer_state` contract with supported values `none`, `bram`, and `uram` for generated HLS reservation.
  - `ddr` optimizer-state storage rejects clearly until tiled optimizer-state backing is implemented.
  - Adds `data_movement.optimizer_state.import/export` support for `m_axi + ps_runtime + full` as an explicit interface contract.
  - Emits `reports/training_optimizer_state.json` and `reports/training_optimizer_state.md`.
- Sprint 29T optimizer selection foundation.
  - Recognizes `sgd`, `momentum`, and `adam` in the config contract.
  - `sgd` remains the implemented generated HLS update path.
  - `momentum` and `adam` reject clearly until generated update kernels and numeric validation are implemented.
- Sprint 29U loss/label contract cleanup.
  - Recognizes `mse` and `cross_entropy` in the config contract.
  - `mse` remains the implemented generated HLS/training-reference path.
  - `cross_entropy` rejects clearly until the generated loss kernel and training reference are implemented.
  - Emits `reports/training_loss_contract.json` and `reports/training_loss_contract.md`.

Validation command used:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py \
  tests/test_build_stages.py \
  tests/test_layer_registry_and_compatibility.py \
  tests/test_training_movement_gradient_export.py \
  tests/test_training_optimizer_loss_contract.py
```

Observed sandbox result: `64 passed, 21 skipped`.

## Sprint 29V/29W/29X bundle

Implemented foundation contracts for batch/gradient accumulation semantics, AXI-stream tiled I/O reporting, and hardware knob/board-fit reporting.

Artifacts now emitted under `reports/`:

- `training_batch_accumulation.json` / `.md`
- `stream_tiled_io.json` / `.md`
- `hardware_knob_contract.json` / `.md`

Truth boundary: these reports make the selected compiler contract explicit. Paper-safe claims still require numeric validation, HLS/Vivado, or FPGA execution artifacts at the required verification level.

## Sprint 29Y/29Z/30G bundle

Implemented Vivado/runtime/truthfulness foundation contracts.

Artifacts now emitted under `reports/`:

- `vivado_bd_contract.json` / `.md`
  - Records expected BD wiring from generated HLS interfaces.
  - Tracks m_axi ports, AXIS ports, required DMA/interconnect/control blocks, and build-stage status.
  - Truth boundary: this is a contract, not Vivado implementation proof.
- `feature_contract.json`
  - Records source generation, numeric validation, HLS synthesis, runtime packaging, and paper-safe status.
- `claim_audit.md`
  - Human-readable claim audit for paper-table filtering.

Runtime package now emits:

- `runtime_package/runtime_api.py`
  - Contains API scaffold functions: `import_weights`, `run_inference`, `run_training`, `export_weights`, `export_gradients`, and `run_sequence`.
  - Truth boundary: generated scaffold validates package/command contract; board-specific DMA/MMIO backend is still required for real FPGA execution.

Validation command used:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py \
  tests/test_build_stages.py \
  tests/test_layer_registry_and_compatibility.py \
  tests/test_training_movement_gradient_export.py \
  tests/test_training_optimizer_loss_contract.py \
  tests/test_execution_semantics_and_board_contract.py \
  tests/test_vivado_runtime_truth_contract.py
```

Observed sandbox result: `67 passed, 25 skipped`.

## Sprint 29W implementation pass — AXI-stream tiled inference I/O

Converted one previously contract-only/rejected path into real generated HLS source for inference:

- `data_movement.inputs.import.interface=axi_stream`, `transport=dma`, `policy=tiled`
  now generates tiled AXI-stream input code.
- `data_movement.outputs.export.interface=axi_stream`, `transport=dma`, `policy=tiled`
  now generates tiled AXI-stream output code.
- Generated source now contains:
  - `FPGAI_AXIS_INPUT_TILE_SIZE`
  - `FPGAI_AXIS_OUTPUT_TILE_SIZE`
  - `input_tile[...]`
  - `output_tile[...]`
  - AXI stream reads/writes with TLAST generation for tiled output.

Important remaining debt is still tracked, not hidden:

- Training AXI-stream tiled I/O still requires compute-fused training tile scheduling.
- Numeric validation for tiled stream inference should be connected in the numeric-validation sprint.
- HLS/Vivado/board execution proof still requires the later validation/BD/runtime sprints.

Validation command used:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py \
  tests/test_build_stages.py \
  tests/test_layer_registry_and_compatibility.py \
  tests/test_training_movement_gradient_export.py \
  tests/test_training_optimizer_loss_contract.py \
  tests/test_execution_semantics_and_board_contract.py \
  tests/test_vivado_runtime_truth_contract.py
```

Observed sandbox result: `67 passed, 25 skipped`.

## Sprint 29R/29S implementation pass — tiled gradient export and DDR optimizer-state backing

Implemented after the runtime/contract foundation:

- `data_movement.gradients.export.interface=m_axi, policy=tiled` now resolves to `m_axi_export_tiled` instead of rejecting.
- Generated training HLS emits `gradients_mem`, `FPGAI_MODE_EXPORT_GRADIENTS`, `FPGAI_GRADIENT_EXPORT_TILE_SIZE`, and a local `gradient_export_tile` used by tiled gradient export loops.
- `training.storage.optimizer_state=ddr` now resolves as supported when paired with `data_movement.optimizer_state.import/export.policy=tiled`.
- Generated training HLS emits `optimizer_state_mem` plus `optimizer_state_tile` for DDR-backed optimizer-state movement.

Still implementation debt, not final:

- Momentum and Adam update kernels remain rejected until real generated update logic and numeric validation are implemented.
- DDR optimizer-state backing currently provides the interface/tile backing required by future optimizers; full Momentum/Adam state update semantics must be implemented in the optimizer sprint.

## Sprint 30J-real / Sprint Q / Sprint A1 foundation — numeric validation and explainable HLS artifacts

Implemented on top of the latest Sprint 29R/29S state:

- Numeric validation now performs a real float32 file comparison when inference `outputs_ref` and `outputs_hw` artifacts are supplied.
- `reports/numeric_validation.json` now records inference comparison metrics:
  - `mse`
  - `mae`
  - `max_abs_error`
  - `cosine_similarity`
  - `passed`
- Training numeric validation now records explicit training checks for gradients, updated weights, and weight deltas when `training_compare` artifacts exist.
- Tiled/full gradient export requests are now recorded in numeric validation as `gradient_export` evidence.
  - If dedicated `gradients_mem` capture is not available yet, the report says `generated_not_captured_by_testbench` rather than pretending export-specific numeric proof exists.
- Generated HLS projects now emit explainability/review artifacts:
  - `reports/generated_hls_explanation.json`
  - `reports/generated_hls_explanation.md`
  - `reports/hardware_design_decisions.json`
  - `reports/hardware_design_decisions.md`
  - `reports/codegen_review_checklist.md`
- Manifest now records `generated_hls_explanation_artifacts`.

Important truth boundary:

- Inference numeric correctness is only marked passed when actual reference/HW output files are supplied and compared.
- Training correctness is only marked passed when the existing Python/testbench comparison artifacts exist.
- Tiled gradient export HLS source exists, but export-specific `gradients_mem` capture still needs the later testbench/runtime implementation pass before it can be claimed as independently numerically proven.

Validation command used in sandbox:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py \
  tests/test_build_stages.py \
  tests/test_layer_registry_and_compatibility.py \
  tests/test_training_movement_gradient_export.py \
  tests/test_training_optimizer_loss_contract.py \
  tests/test_execution_semantics_and_board_contract.py \
  tests/test_vivado_runtime_truth_contract.py \
  tests/test_generated_hls_explanation_and_numeric.py
```

Observed sandbox result: `68 passed, 28 skipped`.

## Sprint 29T implementation pass — Momentum optimizer kernel

Implemented after Sprint 30J/Q/A1:

- `training.optimizer.type: momentum` now generates a real HLS update kernel for Dense/Conv trainable weights and biases.
- Generated source includes persistent velocity arrays named `FPGAI_MOMENTUM_W_*` and `FPGAI_MOMENTUM_B_*`.
- Generated update rule is explicit in source: `V = momentum * V - learning_rate * dParam; Param = Param + V`.
- `training_optimizer_state.json` reports Momentum `hls_update_status=implemented` and `numeric_validation_status=implemented` for the generated update path.
- Adam remains the next optimizer implementation debt and still rejects clearly until its M/V state update kernel is implemented.

Remaining related debt:

- Extend Momentum numeric validation to compare exported optimizer state arrays against a Python reference.
- Implement Adam generated update kernel and numeric validation.
- Connect batch/gradient accumulation modes to Momentum update validation.

## Sprint 29T Adam implementation pass

Implemented generated Adam optimizer update kernel support for `training.optimizer.type: adam`.

Current generated HLS behavior:

- Emits `FPGAI Adam optimizer update kernel` section.
- Emits persistent first/second moment arrays `FPGAI_ADAM_M_*` and `FPGAI_ADAM_V_*`.
- Replaces generated SGD update calls with Adam update loops.
- Records Adam optimizer hyperparameters in `reports/training_optimizer_state.json`.

Remaining validation debt:

- Adam optimizer-state tensor export/reference comparison must be added to full numeric validation.
- Adam with batch/gradient accumulation still needs dedicated numeric validation.

## Sprint 29V implementation pass — native batch/gradient accumulation runtime modes

Implemented after Momentum/Adam optimizer codegen fixes:

- Native accumulation now exposes explicit HLS mode constants:
  - `FPGAI_MODE_ACCUMULATE_GRADIENTS = 3`
  - `FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS = 4`
  - `FPGAI_MODE_RESET_ACCUMULATORS = 5`
- Generated training HLS now labels the reset/accumulate/apply paths with readable runtime-command comments.
- `runtime.sequence` now accepts:
  - `reset_accumulators`
  - `accumulate_gradients`
  - `apply_accumulated_gradients`
- `reports/training_batch_accumulation.json` records the HLS mode mapping and supported runtime commands for native accumulation.
- `runtime_package/runtime_api.py` exposes API scaffold functions for native accumulation commands.

Truth boundary:

- The native HLS modes and command mapping are generated and source-validated.
- Paper-safe numeric claims still require the training comparison artifacts and `numeric_validation.json` to pass for the selected model, optimizer, batch size, and accumulation schedule.


## Sprint 29U implementation pass - Cross-entropy loss kernel

Implemented real generated HLS support for `training.loss.type: cross_entropy`. The training top now emits a stable softmax/logit cross-entropy loss path, computes `loss_value`, and sets the final-layer gradient to `probability - target`. The loss contract report marks cross-entropy as implemented, while paper-safe correctness still depends on numeric validation for the selected model/config.


## Sprint 29T optimizer-state export capture

- Added generated HLS mode `FPGAI_MODE_EXPORT_OPTIMIZER_STATE = 9`.
- Momentum/Adam persistent optimizer state can now be serialized to `optimizer_state_mem` when `data_movement.optimizer_state.export` uses `m_axi`.
- Runtime API scaffold exposes `export_optimizer_state()`.
- `numeric_validation.json` now distinguishes generated export-capture support from missing runtime/testbench comparison files.
- Remaining debt: board/testbench capture must write `optimizer_state_after.bin` and compare it to `optimizer_state_after_ref.bin` for paper-safe optimizer-state correctness.

## Sprint 29T+ optimizer-state capture validation update

- Added runtime-package optimizer-state capture metadata and API helper support.
- Runtime package now copies `optimizer_state_after.bin` and `training_reference/optimizer_state_after_ref.bin` when present.
- Generated `runtime_api.py` now exposes `capture_optimizer_state(payload, out_path=...)` and `export_optimizer_state(board_payload=..., capture_path=...)` so board/runtime backends can persist mode-9 optimizer-state payloads into `optimizer_state_after.bin`.
- `numeric_validation.json` now auto-discovers captured optimizer-state files and compares packed Momentum/Adam state when both ref/got files exist.
- Remaining debt: board-specific PYNQ/KV260 runtime must call HLS mode 9 through DMA/MMIO and pass the returned bytes to `export_optimizer_state(board_payload=...)` during real FPGA execution.
## Sprint 29R+ gradient export capture validation

- Added runtime/API capture path for `gradients_after.bin` / `gradients_export.bin`.
- Numeric validation now compares dedicated gradient-export payloads against `training_reference/grads_ref.bin` when both exist.
- Clean compile preserves externally captured gradient export validation files.
- Remaining debt: board/CSim backend must call `FPGAI_MODE_EXPORT_GRADIENTS` and write the captured payload automatically.


## Sprint 29R/29T CSim automatic capture update

- Training CSim testbench now detects `data_movement.gradients.export` with `m_axi` and calls generated mode `FPGAI_MODE_EXPORT_GRADIENTS = 8` after training.
- The testbench writes dedicated gradient-export capture files: `gradients_after.bin` and `gradients_export.bin`.
- Training CSim testbench now detects Momentum/Adam optimizer-state export with `m_axi` and calls generated mode `FPGAI_MODE_EXPORT_OPTIMIZER_STATE = 9` after training.
- The testbench writes `optimizer_state_after.bin` so numeric validation can compare captured optimizer state when a Python reference is available.
- This closes the CSim capture gap for generated export modes; remaining debt is board runtime capture through DMA/MMIO on PYNQ/KV260/KR260.

## Sprint B implementation pass — PYNQ/KV260 runtime buffer allocation and binding

Implemented generated runtime buffer metadata and fake-board-testable PYNQ/KV260 buffer allocation support.

Current generated runtime package now emits:

- `runtime_package/buffer_plan.json`
- `runtime_package/runtime_execution_plan.json`
- `runtime_package/board_runtime.py` allocation helpers
- `runtime_package/runtime_api.py` allocation/binding helpers

Generated runtime support added:

- `allocate_buffers_from_plan(...)` in `board_runtime.py`
- `allocate_runtime_buffers(...)` in `runtime_api.py`
- `bind_allocated_buffers(...)` in `runtime_api.py`
- `bind_backend(..., buffers=...)` buffer handoff
- sync-before/sync-after execution-plan handling for generated runtime commands
- fake `allocate_fn` injection for non-board unit tests
- PYNQ `allocate()` use only when allocation is requested without injection

Buffer plan currently covers the main runtime PS/PL exchange buffers:

- `input`
- `output`
- `weights` when runtime import/export is required
- `labels` for training packages
- `gradients_mem` for mode-8 gradient export
- `optimizer_state_mem` for mode-9 optimizer-state export

Truth boundary:

- The generated runtime package can now allocate, bind, and synchronize PYNQ/KV260-style buffers from compiler-generated metadata.
- This is still not a real FPGA execution claim. Real board validation still requires Vivado handoff artifacts, a deployed overlay, board-specific buffer addresses/interfaces, and captured execution reports.

Validation:

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py \
  tests/test_build_stages.py \
  tests/test_layer_registry_and_compatibility.py \
  tests/test_training_movement_gradient_export.py \
  tests/test_training_optimizer_loss_contract.py \
  tests/test_execution_semantics_and_board_contract.py \
  tests/test_vivado_runtime_truth_contract.py \
  tests/test_generated_hls_explanation_and_numeric.py
```

Result in this patch workspace:

```text
77 passed, 35 skipped
```

## Sprint C implementation pass — runtime execution reports

Implemented generated runtime execution reporting in the existing runtime package generator.

Current generated runtime package now writes these files when `runtime_api.run_sequence()` is executed:

- `runtime_package/runtime_execution_report.json`
- `runtime_package/runtime_execution_report.md`

Generated runtime report support added:

- command-by-command runtime sequence logging
- mode-number logging from `runtime_execution_plan.json`
- sync-before and sync-after buffer logging
- capture-file logging for gradient and optimizer-state exports
- backend metadata logging, including backend type, board, top/IP name, and bound buffers
- per-command latency measurement with `time.perf_counter()`
- strict failure behavior that writes the report before re-raising the runtime error
- non-strict failure behavior with `run_sequence(strict=False, return_report=True)` for inspection/debug flows
- `run_sequence(return_report=True)` to return the full report object while still writing JSON/Markdown artifacts

Truth boundary:

- The generated runtime package now produces auditable runtime execution reports for board/fake-board runtime calls.
- This is still not a claim that real FPGA execution passed. Real board validation still requires Vivado handoff artifacts, deployed bitstream/overlay, real board buffer binding, and Sprint T FPGA execution capture.

Code quality note:

- Sprint C stayed in the existing runtime package path (`fpgai/runtime/package.py`) and extended generated `runtime_api.py` rather than adding detached scripts.
- Tests import and execute the generated runtime API, so the generated Python source is syntax-checked and behavior-checked.

Validation in this patch workspace:

```text
79 passed, 35 skipped
```

## Sprint C++ implementation pass — generated C++ readability and resource hygiene

Implemented generated C++ quality reporting and source-level semantic checks in the existing HLS explanation/reporting path.

New reports emitted under `reports/`:

- `generated_cpp_readability.json`
- `generated_cpp_readability.md`
- `generated_cpp_validation.json`
- `generated_cpp_validation.md`

Implemented checks:

- generated top source exists
- top function is present
- requested runtime/source features are present
- unrequested runtime/source features are absent
- placeholder/future-work markers are not present in generated C++
- readability level is resolved and reflected in comment density/section summaries
- generated C++ validation status is honest and limited to static generated-source contract validation

Important resource/latency hygiene rule added:

- If the user does not request an import/export/runtime feature, the generated C++ must not contain that unused mode/path/comment. This keeps generated HLS source explainable and avoids unnecessary latency/resource risk from inactive branches leaking into synthesis.

Concrete generator fixes:

- Inference runtime weight import without export no longer emits `FPGAI_MODE_EXPORT_WEIGHTS`.
- Training runtime weight import without export no longer emits `FPGAI_MODE_EXPORT_WEIGHTS`.
- Existing gradient and optimizer-state export generation remains command/config driven and is now checked by reports/tests.

Truth boundary:

- Generated C++ is now structurally explainable, readability-checked, and tested against selected YAML decisions.
- This is not yet a claim that every generated source is HLS-synthesized or board-validated. HLS/Vivado/FPGA validation remains in later sprints.

Validation in this patch workspace:

```text
79 passed, 38 skipped
```

## Sprint D implementation pass — direct Vivado BD Tcl handoff generation

Implemented direct compiler-integrated Vivado handoff generation using the existing Vivado bridge/build-stage path.

Generated when `build.stages.vivado_project=true`:

- `vivado/project.tcl`
- `vivado/bd.tcl`
- `vivado/run_vivado.tcl`
- `reports/vivado_bd_validation.json`
- `reports/vivado_bd_validation.md`

Implemented behavior:

- `cpp`/reports-only builds do not emit Vivado Tcl files.
- Vivado Tcl is emitted only when the resolved build stages request `vivado_project`.
- Generated Tcl is board-aware:
  - PYNQ-Z2 uses `processing_system7`.
  - KV260/KR260 use `zynq_ultra_ps_e`.
- Generated BD Tcl always includes the required PS block, HLS IP lookup/instance path, and AXI-Lite control interconnect path.
- AXI DMA is emitted only when generated HLS source exposes AXI-stream ports.
- m_axi memory interconnect is emitted only when generated HLS source exposes m_axi ports.
- Unrequested gradient/optimizer-state paths are not written into the Vivado Tcl.
- `vivado_bd_validation.json` records structural checks and the truth boundary.

Truth boundary:

- Sprint D validates Tcl generation and static block-design structure only.
- It is not a claim that Vivado synthesis, implementation, bitstream generation, or FPGA execution passed.
- Sprint E must add Vivado execution/import truth handling, and Sprint T must add real board execution validation.

Validation in this patch workspace:

```text
focused Vivado/build/runtime tests passed
```

### Sprint D fix — DMA generation must follow requested data movement

Fixed after integration testing showed embedded inference Vivado BD Tcl still emitted `axi_dma_0` because the bridge treated any generated AXI-stream-looking source port as a DMA requirement.

Correct rule:

- AXI DMA is generated only when the resolved/user data movement requests `axi_stream` or `transport: dma`.
- Generated source port names alone are not enough to instantiate DMA, because unused DMA costs resources and can misrepresent the user-selected runtime/movement plan.
- Runtime sequence command parsing now recognizes string commands and single-key command dictionaries, not only dictionaries with an explicit `command` key.

Truth boundary remains unchanged: Sprint D is structural Vivado Tcl generation/validation only, not Vivado implementation or bitstream proof.

## Sprint E implementation pass — Vivado implementation / bitstream truth reports

Implemented honest Vivado execution/import truth reporting in the existing Vivado bridge path.

Generated reports:

- `reports/vivado_validation_report.json`
- `reports/vivado_validation_report.md`
- `reports/vivado_implementation_report.json`
- `reports/vivado_implementation_report.md`
- `reports/bitstream_report.json`
- `reports/bitstream_report.md`

Implemented behavior:

- CPP-only or Vivado-not-requested builds report `not_requested` instead of creating fake Vivado success.
- Vivado project validation reports `tool_missing` when the configured Vivado executable is unavailable.
- Vivado implementation reports `tool_missing` when Vivado is unavailable and `artifact_missing` when implementation was requested but no real implementation artifact/report exists.
- Bitstream reports `not_requested`, `tool_missing`, or `artifact_missing` unless a real `.bit`/`.xsa` artifact exists and Vivado implementation is already passed.
- Reports include `claimed_success: false` unless real Vivado/artifact evidence is present.
- Manifest now records `vivado_truth_artifacts` so downstream paper/claim audit code can trace the verification level.

Truth boundary:

- Sprint E does not run Vivado by itself and does not claim implementation or bitstream success without actual tool/artifact evidence.
- Board-fit capacity enforcement is still Sprint F.
- Real FPGA execution validation is still Sprint T.

## Sprint F update — Board-fit enforcement

Sprint F adds real compiler-side board-fit enforcement using the existing Vivado board registry and hardware feasibility reporter.

Implemented:

- `reports/board_fit.json` and `reports/board_fit.md` now expose top-level deployment status, limiting dimension, and stage gating.
- Board-fit now derives requested storage/interface requirements from YAML instead of only reading estimator totals.
- URAM-backed storage is rejected for PYNQ-Z2 because the board profile has zero URAM.
- DMA, AXI-stream, m_axi, DDR, gradient export, optimizer-state export, and weight import/export requirements are counted only when requested by YAML/runtime sequence.
- Vivado implementation and bitstream reports now return `blocked_by_board_fit` when board-fit does not allow the selected board/stage.
- Vivado project Tcl generation remains allowed for structural handoff/debug even when implementation/bitstream are blocked.

Truth boundary:

- Board-fit enforcement is based on selected board profiles, estimator/imported resource data, and resolved YAML movement/storage requirements.
- It blocks invalid implementation/bitstream claims, but it is not a replacement for calibrated HLS/Vivado timing/resource reports or real board execution.

## Sprint G update — HLS synthesis truth and estimate-vs-HLS reports

Sprint G adds real HLS synthesis artifact import and estimate-vs-HLS truth reporting in the compiler/report flow.

Generated reports:

- `reports/hls_synthesis_report.json`
- `reports/hls_synthesis_report.md`
- `reports/estimate_vs_hls.json`
- `reports/estimate_vs_hls.md`

Implemented behavior:

- CPP-only and non-synthesis builds report `not_requested` and do not claim HLS success.
- HLS synthesis requests with a missing Vitis/Vivado HLS executable report `tool_missing` instead of crashing or pretending success.
- Missing csynth artifacts report `artifact_missing`.
- Real `*_csynth.xml`, `csynth.xml`, `*_csynth.rpt`, or `csynth.rpt` artifacts are parsed when present.
- Parsed values include LUT, FF, DSP, BRAM18, URAM when available, latency cycles, and latency milliseconds.
- `estimate_vs_hls.json` compares design-space estimates against parsed HLS values and sets `paper_safe: true` only when a real parsed HLS report exists.
- Manifest now records `hls_truth_artifacts` for downstream paper/claim audit traceability.

Truth boundary:

- Sprint G proves FPGAI can import and compare real HLS synthesis artifacts when they exist.
- It does not claim the estimator is calibrated or accurate across all models. Calibration still requires larger HLS/Vivado sweeps.
- It does not claim Vivado implementation, bitstream generation, or real FPGA execution.

## Sprint H update — Batch / gradient accumulation numeric validation foundation

Sprint H extends the existing numeric validation report instead of adding a detached checker.

Implemented:

- `reports/numeric_validation.json` now includes a `batch_accumulation` section.
- `reports/numeric_validation.md` records the batch-accumulation validation status.
- Native accumulation requests are detected from `training.batch_size`, `training.gradient_accumulation`, and `runtime.sequence`.
- Runtime commands are traced for `reset_accumulators`, `accumulate_gradients`, and `apply_accumulated_gradients`.
- The report records `batch_size`, `accumulation_steps`, accumulation policy, optimizer apply count, and required runtime-command coverage.
- For repeated-microbatch average accumulation, the Python reference gradient is materialized as `training_reference/accumulated_grads_ref.bin` and weights-after reference as `training_reference/weights_after_accum_ref.bin`.
- Missing HLS/runtime accumulated-gradient captures are reported as `artifact_missing`, never as a pass.
- Plain training without accumulation reports `batch_accumulation.status = not_requested` and does not emit unused accumulated reference files.

Truth boundary:

- Sprint H establishes traceable batch-accumulation numeric-validation artifacts and honest missing-capture reporting.
- A paper-safe accumulated-training correctness claim still requires captured HLS/runtime accumulated gradients and weights-after artifacts to compare against the generated Python reference.
- Full optimizer-specific accumulated updates for Momentum/Adam and loss-specific validation continue in Sprint I/J.

## Sprint I update — Optimizer-state numeric correctness foundation

Sprint I strengthens optimizer numeric validation for persistent Momentum/Adam state while keeping the existing compiler/report flow.

Implemented:

- Python training reference now records optimizer metadata in `training_reference/summary.json`.
- Momentum reference state is materialized as zero `velocity_before` and `velocity_after = momentum * velocity_before - learning_rate * gradient`.
- Adam reference state is materialized as zero first/second moments before the step and updated first/second moments after the step.
- Reference files are emitted when the optimizer has persistent state:
  - `training_reference/optimizer_state_before_ref.bin`
  - `training_reference/optimizer_state_after_ref.bin`
- Adam `bias_correction` is explicit in the reference and numeric report. The reference applies bias correction only when requested by YAML.
- `reports/numeric_validation.json.training.reference` now records optimizer type, bias-correction setting, and optimizer-state reference files.
- `reports/numeric_validation.json.optimizer_state_validation` now reports `artifact_missing` when a reference exists but no HLS/runtime optimizer-state capture exists.
- SGD optimizer state reports `not_applicable` instead of pretending a persistent state tensor exists.
- Existing preserved runtime/CSim capture handling now includes optimizer-state before/after reference and captured files.

Truth boundary:

- Momentum/Adam optimizer-state references are now generated and compared when captured artifacts exist.
- Missing optimizer-state captures are reported honestly as missing evidence, not as a pass.
- A paper-safe Momentum/Adam optimizer correctness claim still requires matching captured HLS/runtime optimizer-state and weights-after artifacts.

## Sprint J update — Cross-entropy numeric-validation foundation

Sprint J strengthens the cross-entropy training path by adding Python reference artifacts and explicit numeric-validation truth reporting.

Implemented:

- Python training reference now respects `training.loss.type` for `cross_entropy`.
- Stable-softmax cross-entropy reference is generated from the model output logits/proxy logits.
- Cross-entropy reference artifacts are emitted when requested:
  - `training_reference/logits_ref.bin`
  - `training_reference/softmax_ref.bin`
  - `training_reference/cross_entropy_loss_ref.json`
  - `training_reference/dlogits_ref.bin`
- `reports/numeric_validation.json.training.reference` now records loss type and cross-entropy reference artifact paths.
- `reports/numeric_validation.json.loss_validation` reports cross-entropy validation status, reference availability, stable-softmax metadata, and capture comparisons.
- MSE training reports `loss_validation.status = not_requested` and does not claim cross-entropy evidence.
- Missing HLS/runtime dlogits/softmax captures are reported as `artifact_missing`, never as a pass.

Truth boundary:

- Cross-entropy reference artifacts and validation status are now tracked by the normal numeric-validation report.
- A paper-safe cross-entropy correctness claim still requires matching captured HLS/runtime loss-gradient evidence, gradients, and weights-after artifacts.

## Sprint K update — Training tiled-I/O compute-fused validation foundation

Sprint K strengthens training tiled I/O by connecting tiled movement requests to Python reference artifacts and numeric-validation truth reporting.

Implemented:

- Training tiled-I/O requests are detected from `data_movement.inputs.import`, `data_movement.labels.import`, and `data_movement.outputs.export` when `policy: tiled` is selected with `m_axi` or `axi_stream`.
- Python training reference now emits tiled reference artifacts only when tiled training I/O is requested:
  - `training_reference/tiled_inputs_ref.bin`
  - `training_reference/tiled_labels_ref.bin`
  - `training_reference/tiled_outputs_ref.bin`
  - `training_reference/tiled_gradients_ref.bin`
  - `training_reference/tiled_weights_after_ref.bin`
- `reports/numeric_validation.json.training.reference` records tiled reference paths.
- `reports/numeric_validation.json.training_tiled_io` records interface type, tile sizes, input/label/output tiled status, AXI-stream TLAST requirement, reference availability, capture availability, and comparison results.
- Missing HLS/runtime tiled output/gradient/weights-after captures are reported as `artifact_missing`, never as a pass.
- Plain training without tiled I/O reports `training_tiled_io.status = not_requested` and does not emit tiled reference artifacts.

Truth boundary:

- Sprint K establishes structural/numeric traceability for training tiled I/O and honest missing-capture reporting.
- A paper-safe tiled-training correctness claim still requires captured HLS/runtime tiled outputs, gradients, and weights-after artifacts to compare against the Python tiled reference.

## Sprint L update — Layer backend status and full-priority layer support contract

Sprint L strengthens the layer backend contract so FPGAI treats layers as first-class compiler artifacts, not informal compatibility notes.

Implemented:

- Priority layer registry now includes the user-requested layer family:
  - Dense / Linear
  - Conv / Conv2D
  - DepthwiseConv2D
  - PointwiseConv2D
  - ReLU / LeakyReLU
  - MaxPool
  - AvgPool / AveragePool
  - Flatten / Reshape
  - Softmax
  - BatchNormalization / BatchNorm
  - Add / residual-style elementwise Add
  - GlobalAveragePool
- Inference and training capability tables now include the priority aliases and specialization layers instead of leaving them as placeholder/unknown registry entries.
- `reports/layer_backend_status.json` and `reports/layer_backend_status.md` are emitted next to `model_compatibility` and `layer_knob_contract`.
- The new layer backend status report records, per encountered layer:
  - active pipeline support
  - inference status
  - training status
  - shape inference status
  - HLS codegen status
  - memory-planning status
  - precision status
  - numeric-validation readiness
  - training forward/backward status where applicable
  - unsupported reason if any
- Unknown operators remain unsupported and must not be silently accepted.
- The report policy records that all layers are required, unsupported layers block compile, unused layer kernels must be absent, and training requires backward support.

Truth boundary:

- Sprint L adds an auditable layer-backend status system and expands the priority layer support contract.
- Unsupported unknown operators still reject/report clearly.
- This sprint does not claim every knob is fully validated across every layer; that remains Sprint M.

## Sprint M update — Per-layer knob application contract

Sprint M strengthens `layer_knob_contract` from a coarse registry table into a per-layer, per-knob traceability report.

Implemented:

- `reports/layer_knob_contract.json` schema version 2 records each important knob for every encountered layer.
- Each knob now resolves to one of:
  - `applied`
  - `not_applicable`
  - `not_requested`
  - `compiler_default`
  - `rejected`
- The report records evidence and reasons for every knob/layer decision.
- Manual YAML sources are tracked for:
  - precision
  - pipelining
  - parallelization
  - tiling
  - weight storage
  - activation storage
  - gradient storage
  - optimizer-state storage
  - data movement
  - runtime sequence
  - build stages
  - board fit
  - training
- Weight-storage knobs are `applied` only for layers with trainable/parameter tensors and `not_applicable` for activation-only layers such as ReLU/MaxPool.
- Gradient/optimizer-state storage knobs are only relevant in training mode and report `not_applicable` in inference mode.
- Unknown/unsupported operators mark every knob as `rejected` with the backend reason instead of silently accepting unsupported combinations.
- The contract explicitly records the resource/latency hygiene rule: unrequested import/export/runtime/tiling/stream paths must be absent from generated C++ and Vivado/runtime artifacts.

Truth boundary:

- Sprint M proves YAML knob decisions are tracked per layer and reported as applied, not applicable, defaulted, or rejected.
- It does not yet prove HLS resource/latency deltas for every knob. Precision-effect and parallel/pipeline/tiling HLS-effect validation remain Sprint N/O.

## Sprint N update — Precision effect validation

Sprint N adds a first-class precision-effect report that verifies precision is not only a YAML/report value but is materialized in generated HLS artifacts.

Implemented:

- `reports/precision_effect.json`
- `reports/precision_effect.md`
- `manifest.json.precision_effect_artifacts`
- Generated-source evidence scans for `hls/include/fpgai_types.h` and generated top sources.
- Manual precision YAML sources are recorded from `numerics.precision_mode`, `numerics.defaults.*`, `numerics.layers`, and precision-sweep selection.
- The report records resolved activation/weight/bias/accumulator bits from `precision_layout.json`.
- The report verifies generated `ap_fixed<...>` typedef evidence agrees with the resolved precision layout.
- The report separates:
  - generated precision materialization
  - optional quantization/numeric metrics
  - optional precision sweep evidence
  - HLS-backed paper-safe resource/timing claims
- `paper_safe_hls_claim` remains false unless `estimate_vs_hls.status == compared` from parsed real HLS artifacts.

Truth boundary:

- Sprint N proves precision decisions affect generated type/layout artifacts and are traceable in reports.
- It does not claim real HLS resource/timing precision effects unless Sprint G HLS truth reports contain a real estimate-vs-HLS comparison.
- Parallelization/pipelining HLS-effect validation remains Sprint O.

## Sprint O update — Parallelization and pipelining effect validation

Sprint O adds a first-class report for source-level parallelization and pipelining materialization.

Implemented:

- `reports/parallel_pipeline_effect.json`
- `reports/parallel_pipeline_effect.md`
- `manifest.json.parallel_pipeline_effect_artifacts`
- Manual YAML source tracking for `hls.pipeline_ii`, `hls.dense.*`, `hls.conv.*`, `hls.activation.*`, `optimization.parallel.*`, `parallelization`, and `pipelining`.
- Generated-source evidence scans for `FPGAI_PIPELINE_II`, dense/conv unroll macros, `#pragma HLS PIPELINE`, `#pragma HLS UNROLL`, and `#pragma HLS ARRAY_PARTITION`.
- Per-knob statuses: `applied`, `not_requested`, `compiler_default`, or `rejected`.
- HLS-backed paper-safe claims remain false unless real Sprint G estimate-vs-HLS comparison evidence exists.

Truth boundary:

- Sprint O proves parallelization and pipelining decisions are materialized in generated HLS source and are traceable in reports.
- It does not claim real latency/resource improvement unless parsed HLS reports exist and `estimate_vs_hls.status == compared`.

## P3D-D dataset metadata and quality hardening

- Runtime buffer plans preserve logical dataset batch shapes while retaining flat physical word counts.
- Dataset-backed `run_inference` uses the resolved sample count as its repeat count.
- Numeric validation checks HLS dataset execution records for sample/invocation/output consistency.
- Classification validation emits confusion-matrix and per-class-accuracy artifacts.
- Focused P3D-D regression suite passed.

## P3D-E torchvision dataset adapters

- `validation.dataset.source: torchvision` supports `MNIST` and `FashionMNIST`.
- Torch and torchvision are optional dependencies exposed through the `datasets` extra.
- Dataset download is disabled unless YAML explicitly sets `download: true`.
- Deterministic selection modes include `first`, `random`, and `balanced_per_class`.
- Balanced selection records exact source indices and class distribution in `dataset_manifest.json`.
- Preprocessing records normalization, flattening, optional channel insertion, and optional mean/std transforms.
- Complete non-frozen configurations are provided for balanced 100-sample MNIST and FashionMNIST adapter validation.
- FashionMNIST quality claims remain disallowed until a matching FashionMNIST-trained ONNX model is supplied.

## P3D-F1 — Dataset-backed HLS training execution

Status: implemented and focused-test validated.

- Training compilation now consumes normalized `validation.dataset` inputs instead of dummy input data when a dataset is configured.
- Classification labels are lowered to one-hot float32 target records matching the model output width; regression targets are preserved.
- The generated training CSim testbench receives per-record input/target widths and reports dataset input records, target records, and records consumed.
- A `reports/training_dataset_contract.json` artifact records sample count, record widths, input/target artifacts, and the current reference scope.
- Current scope boundary: HLS training executes the configured dataset batch, while the software training reference remains a first-sample/single-step reference. Dataset-wide optimizer reference execution is scheduled for P3D-F2 and must precede dataset-wide training-correctness claims.

## P3D-F2 — Dataset-wide training numerical validation

- Dataset-backed accumulated SGD reference consumes every ordered sample.
- Canonical training execution, comparison, and learning-behavior artifacts are emitted.
- Initial/final dataset loss, gradient norms, and weight-update norm are reported.
- Numerical correctness is independent from one-update loss direction; convergence is not claimed.

### P3D-F2 v2 — training reference parameter binding repair

- Dataset-wide training reference supports named tensors and direct operation-attribute parameters.
- Removed the brittle requirement that Dense/Conv/BatchNorm parameters must be explicit op inputs.
- Added regression coverage for attr-backed Dense parameters matching imported `dense0` models.
- Focused training dataset validation: 5 passed.

## P3D-F2.1 — Dataset training comparison validation

Implemented canonical tolerance-based validation for dataset-backed training comparisons.
`reports/training_dataset_comparison.json` now records complete gradient, weight-delta,
final-weight, and execution-count checks. The learning-behavior report derives
`numeric_validation_status` from this comparison status rather than from comparison-object
presence. Missing or empty comparison artifacts produce `pending_comparison`; failed
checks produce `failed_tolerance`; only complete passing checks produce `passed`.

## P3D-F2.2 — hardware-domain dataset training reference

- Added fixed-point dataset training reference artifacts using declared weight, bias, gradient, accumulator, and update precisions.
- The reference quantizes each sample gradient, accumulates in the configured accumulator type, averages and casts to gradient storage, applies the quantized learning rate, and casts final parameters to their storage roles.
- Dataset training validation now uses HLS-versus-hardware-domain metrics for the acceptance decision.
- Float-reference comparison remains available as a secondary quantization diagnostic and does not independently fail hardware numerical validation.
- Missing hardware-domain artifacts produce `pending_comparison`; tolerances were not relaxed.

## P3D-F2.3 — Training semantic traceability

Implemented observable stage-by-stage dataset training trace artifacts. The HLS testbench now exports the accumulated gradient before optimizer reduction and the reduced/exported gradient, while the hardware-domain reference emits matching stage artifacts. The compiler writes `reports/training_gradient_semantics.json` with stage metrics and the first observable divergence stage. Pre-cast update state is explicitly reported as not observable rather than inferred. Focused validation: 11 tests passed.

## P3D-F2.4 — Per-sample and per-layer training gradient traceability

- Added HLS CSim export of each accumulated-gradient snapshot and its derived per-sample fixed-point contribution.
- Added hardware-domain reference artifacts for every per-sample gradient and accumulator state.
- Added a parameter layer map with flat offsets, counts, shapes, and parameter roles.
- Added `reports/training_per_sample_gradient_trace.json` with first divergent sample, parameter index, layer, role, values, and error.
- Focused training dataset validation: 13 passed.

## P3D-F2.5 bias-gradient semantics diagnostics

Dataset-backed training validation now emits gradient comparisons aggregated by layer and parameter role. Bias gradients receive per-sample scale diagnostics against unity, batch-size, and inverse-batch-size hypotheses. These diagnostics are traceability artifacts and do not alter numerical acceptance tolerances.
