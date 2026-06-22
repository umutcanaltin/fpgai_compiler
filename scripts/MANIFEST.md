# FPGAI Script Manifest

This manifest classifies current root-level `scripts/` files so the repository can move from sprint-driven research development to a professional config-first open-source workflow.

## Policy

Users should primarily interact with FPGAI through the installed `fpgai` CLI and YAML configuration files. Files in `scripts/` are transitional utilities only. New long-term functionality should be implemented inside the `fpgai/` package and exposed through CLI subcommands.

## Categories

- `PUBLIC_TRANSITION`: usable public-facing tool today, but should eventually become a `fpgai` CLI subcommand.
- `PAPER_EVIDENCE`: paper/reproducibility artifact generation; should move to `fpgai/evidence/`.
- `DEV_DIAGNOSTIC`: developer debugging or validation helper; should move to `tools/dev/` or tests.
- `LEGACY_EXPERIMENT`: historical experiment/evidence script; keep temporarily, then archive or remove after replacement.
- `DELETE_CANDIDATE`: sprint patch or obsolete one-off helper; delete after confirming no imports/references.

## Current script classification

| Script | Category | Target location / action | Notes |
|---|---|---|---|
| `analyze_fpgai_experiments.py` | PAPER_EVIDENCE | `fpgai/evidence/experiments.py` | General experiment analysis; useful but should be packaged. |
| `build_bitstream.py` | PUBLIC_TRANSITION | `fpgai cli vivado build-bitstream` | Public hardware build wrapper candidate. |
| `classify_hardware_feasibility.py` | PAPER_EVIDENCE | `fpgai/evidence/feasibility.py` | Useful paper/support utility. |
| `collect_claim_support_v2.py` | PAPER_EVIDENCE | `fpgai/evidence/claims.py` | Keep until evidence CLI exists. |
| `collect_comm_ablation.py` | PAPER_EVIDENCE | `fpgai/evidence/communication.py` | Must clearly label modeled/static transfer evidence. |
| `collect_estimator_accuracy.py` | PAPER_EVIDENCE | `fpgai/evidence/estimator_accuracy.py` | Keep but do not use placeholder 0/1 predictions as claims. |
| `collect_hardware_knob_tables.py` | PAPER_EVIDENCE | `fpgai/evidence/hardware_knobs.py` | Paper table generator. |
| `collect_hls_calibration_samples.py` | DEV_DIAGNOSTIC | `tools/dev/` or tests | Calibration helper. |
| `collect_paper_evidence.py` | PAPER_EVIDENCE | `fpgai/evidence/collect.py` | Candidate unified evidence collector. |
| `collect_training_convergence_evidence.py` | PAPER_EVIDENCE | `fpgai/evidence/training.py` | Paper table generator. |
| `collect_vivado_impl_evidence.py` | PAPER_EVIDENCE | `fpgai/evidence/vivado_impl.py` | Paper table generator. |
| `compare_fpgai_experiments.py` | PAPER_EVIDENCE | `fpgai/evidence/compare.py` | Useful comparison helper. |
| `compare_hardware_knobs.py` | PAPER_EVIDENCE | `fpgai/evidence/hardware_knobs.py` | Merge with hardware knob collector. |
| `compare_host_vs_onnx.py` | DEV_DIAGNOSTIC | `tools/dev/` or tests | Correctness helper. |
| `compare_vitis_vs_host_vs_onnx.py` | DEV_DIAGNOSTIC | `tools/dev/` or tests | Correctness helper. |
| `compile.py` | PUBLIC_TRANSITION | `fpgai compile` | Should be replaced by package CLI usage. |
| `create_model_suite.py` | DEV_DIAGNOSTIC | `tools/dev/` or `examples/` generator | Useful, but not public top-level script. |
| `create_sprint9_precision_effect_sweep.py` | DELETE_CANDIDATE | delete after reference check | One-off sprint generator. |
| `diagnose_training_csim.py` | DEV_DIAGNOSTIC | `tools/dev/` | Training debug helper. |
| `extract_hls_calibration_dataset.py` | DEV_DIAGNOSTIC | `tools/dev/` or `fpgai/analysis/` CLI | Calibration utility. |
| `extract_memory_binding_evidence.py` | LEGACY_EXPERIMENT | archive/remove after evidence CLI replacement | Historical sprint evidence. |
| `extract_parallel_policy_evidence.py` | LEGACY_EXPERIMENT | archive/remove after evidence CLI replacement | Historical sprint evidence. |
| `extract_training_accelerator_evidence.py` | LEGACY_EXPERIMENT | archive/remove after evidence CLI replacement | Historical training sprint evidence. |
| `extract_training_accumulated_batch_evidence.py` | LEGACY_EXPERIMENT | archive/remove after evidence CLI replacement | Historical training sprint evidence. |
| `extract_training_batch_multistep_evidence.py` | LEGACY_EXPERIMENT | archive/remove after evidence CLI replacement | Historical training sprint evidence. |
| `extract_training_multi_epoch_convergence_evidence.py` | LEGACY_EXPERIMENT | archive/remove after evidence CLI replacement | Replaced by `collect_training_convergence_evidence.py`. |
| `extract_training_native_accumulated_batch_evidence.py` | LEGACY_EXPERIMENT | archive/remove after evidence CLI replacement | Historical training sprint evidence. |
| `extract_vivado_bridge_evidence.py` | PAPER_EVIDENCE | `fpgai/evidence/vivado_bridge.py` | Useful until merged with Vivado implementation collector. |
| `generate_kv260_runtime.py` | PUBLIC_TRANSITION | `fpgai runtime generate --board kv260` | Potential public runtime tool. |
| `generate_models.py` | DEV_DIAGNOSTIC | `tools/dev/` or `examples/` | Model generation helper. |
| `generate_paper_artifacts.py` | PAPER_EVIDENCE | `fpgai/evidence/paper_artifacts.py` | Paper figure/table generator. |
| `inspect_pipeline_policy_artifacts.py` | DEV_DIAGNOSTIC | `tools/dev/` | Debug helper. |
| `make_estimator_tables.py` | PAPER_EVIDENCE | merge into `fpgai/evidence/estimator_accuracy.py` | Avoid duplicate estimator logic. |
| `make_input_bin.py` | PUBLIC_TRANSITION | `fpgai data make-input` or examples | Host/runtime helper. |
| `make_weights_bin.py` | PUBLIC_TRANSITION | `fpgai data make-weights` or examples | Host/runtime helper. |
| `paper_frontier.py` | PAPER_EVIDENCE | `fpgai/evidence/plots.py` | Paper plot script. |
| `paper_training_precision.py` | PAPER_EVIDENCE | `fpgai/evidence/plots.py` | Paper plot script. |
| `paper_training_table.py` | PAPER_EVIDENCE | `fpgai/evidence/training.py` | Paper table script. |
| `patch_sprint13b_compiler.py` | DELETE_CANDIDATE | delete after reference check | Sprint patch, not reusable. |
| `patch_sprint13b_compiler_runtime_preload.py` | DELETE_CANDIDATE | delete after reference check | Sprint patch, not reusable. |
| `patch_sprint13b_compiler_runtime_preload_v6.py` | DELETE_CANDIDATE | delete after reference check | Sprint patch, not reusable. |
| `patch_sprint13b_compiler_v2.py` | DELETE_CANDIDATE | delete after reference check | Sprint patch, not reusable. |
| `patch_sprint13b_compiler_v3.py` | DELETE_CANDIDATE | delete after reference check | Sprint patch, not reusable. |
| `patch_sprint13b_compiler_v4.py` | DELETE_CANDIDATE | delete after reference check | Sprint patch, not reusable. |
| `patch_sprint13e_top_native_accum.py` | DELETE_CANDIDATE | delete after reference check | Sprint patch, not reusable. |
| `patch_sprint13f_top_loss_eval.py` | DELETE_CANDIDATE | delete after reference check | Sprint patch, not reusable. |
| `patch_sprint15c_pipeline_policy.py` | DELETE_CANDIDATE | delete after reference check | Sprint patch, not reusable. |
| `plot_policy_results.py` | PAPER_EVIDENCE | `fpgai/evidence/plots.py` | Paper/experiment plot script. |
| `plot_training_loss_curves.py` | PAPER_EVIDENCE | `fpgai/evidence/plots.py` | Paper plot script. |
| `probe_fpgai_config_schema.py` | DEV_DIAGNOSTIC | `tools/dev/` | Useful while stabilizing schema. |
| `report_safe_clock_recommendations.py` | PAPER_EVIDENCE | `fpgai/evidence/safe_clock.py` | Useful paper/support utility. |
| `run_all_training_experiments.py` | LEGACY_EXPERIMENT | replace with `fpgai sweep` | Too broad as public script. |
| `run_fpgai_experiments.py` | PUBLIC_TRANSITION | `fpgai sweep` | Main public sweep functionality should move into package CLI. |
| `run_hls_calibration_validation.py` | DEV_DIAGNOSTIC | `tools/dev/` or `fpgai calibration validate` | Calibration helper. |
| `run_hostcpp.py` | PUBLIC_TRANSITION | `fpgai runtime hostcpp` | Runtime helper candidate. |
| `run_inference_policy_sweep.py` | LEGACY_EXPERIMENT | replace with YAML sweep + `fpgai sweep` | Historical sweep wrapper. |
| `run_paper_experiments.py` | PAPER_EVIDENCE | `fpgai paper run` | Paper workflow command candidate. |
| `run_policy_sweep.py` | LEGACY_EXPERIMENT | replace with YAML sweep + `fpgai sweep` | Historical sweep wrapper. |
| `run_training_policy_sweep.py` | LEGACY_EXPERIMENT | replace with YAML sweep + `fpgai sweep` | Historical sweep wrapper. |
| `run_vitis_csim.py` | PUBLIC_TRANSITION | `fpgai hls csim` | Public HLS helper candidate. |
| `run_vivado_bridge.py` | PUBLIC_TRANSITION | `fpgai vivado bridge` | Public Vivado bridge command candidate. |
| `summarize_hls_calibration_claims.py` | PAPER_EVIDENCE | `fpgai/evidence/calibration.py` | Paper/support utility. |
| `summarize_policy_results.py` | PAPER_EVIDENCE | `fpgai/evidence/policy.py` | Paper/support utility. |
| `test_weights_pack.py` | DEV_DIAGNOSTIC | move into `tests/` | Should become a pytest test. |
| `validate_sprint4_yaml_pipeline.py` | DELETE_CANDIDATE | delete after reference check | Historical sprint validator. |
| `verify.py` | PUBLIC_TRANSITION | `fpgai verify` | Public verification helper candidate. |

## First deletion candidates

These are safe-looking deletion candidates, but must be removed only after a reference check:

```text
scripts/patch_sprint13b_compiler.py
scripts/patch_sprint13b_compiler_runtime_preload.py
scripts/patch_sprint13b_compiler_runtime_preload_v6.py
scripts/patch_sprint13b_compiler_v2.py
scripts/patch_sprint13b_compiler_v3.py
scripts/patch_sprint13b_compiler_v4.py
scripts/patch_sprint13e_top_native_accum.py
scripts/patch_sprint13f_top_loss_eval.py
scripts/patch_sprint15c_pipeline_policy.py
scripts/create_sprint9_precision_effect_sweep.py
scripts/validate_sprint4_yaml_pipeline.py
```

## Reference-check command before deletion

```bash
grep -R "patch_sprint\|create_sprint9\|validate_sprint4" -n . \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=experiments \
  --exclude-dir=build \
  --exclude='scripts/MANIFEST.md'
```
