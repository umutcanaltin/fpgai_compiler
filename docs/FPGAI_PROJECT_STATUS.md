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
