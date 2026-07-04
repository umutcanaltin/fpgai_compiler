# R2C Tool Ownership Audit

## Summary
- devtool_merge_or_delete_candidate: 8
- keep_active_or_migrate_references: 1
- manual_cli_tool_review: 33
- manual_review: 2
- reporting_merge_candidate: 10

## `fpgai/devtools/build_hls_csynth_table.py`
- recommendation: `devtool_merge_or_delete_candidate`
- lines: 295
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/devtools/build_hls_paper_subset_table.py`
- recommendation: `devtool_merge_or_delete_candidate`
- lines: 133
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/devtools/build_paper_stage1_tables.py`
- recommendation: `devtool_merge_or_delete_candidate`
- lines: 273
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/devtools/collect_paper_prediction_codegen.py`
- recommendation: `devtool_merge_or_delete_candidate`
- lines: 259
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/devtools/diagnose_training_csim.py`
- recommendation: `devtool_merge_or_delete_candidate`
- lines: 54
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/devtools/generate_paper_configs.py`
- recommendation: `devtool_merge_or_delete_candidate`
- lines: 260
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/devtools/paper_experiment_matrix.py`
- recommendation: `devtool_merge_or_delete_candidate`
- lines: 266
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/devtools/run_paper_prediction_codegen.py`
- recommendation: `devtool_merge_or_delete_candidate`
- lines: 292
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/devtools/end_to_end_audit.py`
- recommendation: `keep_active_or_migrate_references`
- lines: 388
- has_main: True
- uses_argparse: True
- python_m_refs: 1
  - `KNOBS_TUTORIAL.md`
- import_refs: 0
- old_terms: ['truth']
- doc: End-to-end FPGAI audit runner.  This module is intentionally under fpgai.devtools instead of a loose script. It creates a small truth-audit matrix that checks whether YAML hardware settings materialize into compile plans, predictions, HLS/Vivado arti

## `fpgai/analysis/output_check.py`
- recommendation: `manual_cli_tool_review`
- lines: 419
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/devtools/audit_knob_materialization.py`
- recommendation: `manual_cli_tool_review`
- lines: 256
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Audit whether paper-matrix tiling and memory knobs materialize in generated artifacts.

## `fpgai/devtools/build_paper_artifact_index.py`
- recommendation: `manual_cli_tool_review`
- lines: 311
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Build a paper artifact index from FPGAI experiment outputs.  This tool intentionally reports artifact availability/status only. Resource/timing numeric parsing is handled by later paper-result builders.

## `fpgai/devtools/build_paper_figures.py`
- recommendation: `manual_cli_tool_review`
- lines: 234
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Build paper-ready figures from parsed FPGAI numeric artifacts.

## `fpgai/devtools/build_paper_numeric_results.py`
- recommendation: `manual_cli_tool_review`
- lines: 492
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Build numeric paper-result CSVs from FPGAI prediction, HLS, and Vivado artifacts.

## `fpgai/devtools/build_paper_tables.py`
- recommendation: `manual_cli_tool_review`
- lines: 448
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Build paper-ready result tables from parsed FPGAI numeric artifacts.

## `fpgai/devtools/canonical_hls_source_audit.py`
- recommendation: `manual_cli_tool_review`
- lines: 194
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Audit YAML hardware effects in canonical FPGAI-generated HLS source only.  This intentionally ignores Vitis-generated/internal directories such as: - fpgai_hls_proj/ - .autopilot/ - impl/ - syn/ - csim/  It only scans: - hls/src - hls/include - hls/r

## `fpgai/devtools/hls_source_effect_audit.py`
- recommendation: `manual_cli_tool_review`
- lines: 151
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Audit whether hardware knobs are visible in generated HLS source artifacts.

## `fpgai/devtools/probe_config_schema.py`
- recommendation: `manual_cli_tool_review`
- lines: 25
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/analyze_experiments.py`
- recommendation: `manual_cli_tool_review`
- lines: 59
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Analyze FPGAI experiment outputs and generate paper-ready artifacts.

## `fpgai/reporting/claim_traceability.py`
- recommendation: `manual_cli_tool_review`
- lines: 269
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: ['evidence', 'safe_wording']
- doc: Collect reviewer-safe claim support for FPGAI.  This script creates:   reports/reproducibility/claim_support.csv   reports/reproducibility/claim_support.md  Rule:   Do not invent claims. A claim is marked READY only if at least one   supporting artif

## `fpgai/reporting/communication_ablation.py`
- recommendation: `manual_cli_tool_review`
- lines: 286
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: ['evidence']
- doc: Collect communication-aware ablation reports from FPGAI artifacts.  This collector is conservative: it does not claim hardware runtime speedup. It scans existing build artifacts for communication/memory plans and estimates transfer-volume trade-offs 

## `fpgai/reporting/compare_experiments.py`
- recommendation: `manual_cli_tool_review`
- lines: 37
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Compare multiple FPGAI experiment result directories.

## `fpgai/reporting/estimator_accuracy.py`
- recommendation: `manual_cli_tool_review`
- lines: 394
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: ['evidence']
- doc: Collect resource-estimator accuracy reports safely.  Key safety rule:   Do NOT treat placeholder values such as 0 or 1 as valid resource predictions.  This collector compares Vivado implementation resources against FPGAI estimator artifacts only when

## `fpgai/reporting/hardware_knob_tables.py`
- recommendation: `manual_cli_tool_review`
- lines: 301
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Build paper-facing hardware knob tables from Vivado implementation reports.  Inputs:   reports/vivado_impl_summary/vivado_impl_summary.csv  Outputs:   reports/hardware_knobs/precision_table.csv/.md   reports/hardware_knobs/pipeline_table.csv/.md   re

## `fpgai/reporting/hls_calibration_samples.py`
- recommendation: `manual_cli_tool_review`
- lines: 347
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Collect and aggregate FPGAI HLS calibration sample datasets.  Sprint 10B helper.  Two collection modes are supported:  1. Existing dataset mode: collect already-written ``estimate_vs_hls.json`` files. 2. Expanded project-root mode: find every ``build

## `fpgai/reporting/hls_calibration_summary.py`
- recommendation: `manual_cli_tool_review`
- lines: 249
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Summarize HLS calibration validation results into paper-safe claims.  This script reads calibration_validation.json from Sprint 10 and produces a small JSON/Markdown summary that distinguishes:   - same-operator held-out validation (leave-one-sample/

## `fpgai/reporting/hls_calibration_validation.py`
- recommendation: `manual_cli_tool_review`
- lines: 56
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/paper_artifacts.py`
- recommendation: `manual_cli_tool_review`
- lines: 475
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: ['evidence']
- doc: Collect FPGAI experiment evidence into paper-ready tables.  This script is intentionally read-only with respect to experiment artifacts. It scans completed experiment folders and writes Markdown/CSV/JSON summaries.  Default inputs:   experiments/spri

## `fpgai/reporting/paper_training_precision.py`
- recommendation: `manual_cli_tool_review`
- lines: 252
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/paper_training_table.py`
- recommendation: `manual_cli_tool_review`
- lines: 298
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/plot_policy_results.py`
- recommendation: `manual_cli_tool_review`
- lines: 289
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/plot_training_loss_curves.py`
- recommendation: `manual_cli_tool_review`
- lines: 75
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Plot training loss curves/table summary.  Input:   reports/training_convergence/training_convergence.csv  Output:   reports/training_convergence/loss_curve.png  If only initial/final loss are available, this plots a two-point loss line per design.

## `fpgai/reporting/policy_summary.py`
- recommendation: `manual_cli_tool_review`
- lines: 98
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/safe_clock_recommendations.py`
- recommendation: `manual_cli_tool_review`
- lines: 197
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: ['evidence']
- doc: Generate safe-clock recommendations from FPGAI Vivado evidence.  Reads <experiment>/vivado_bridge_evidence/evidence.json and writes:   <experiment>/safe_clock_report/safe_clock_report.csv   <experiment>/safe_clock_report/safe_clock_report.md   <exper

## `fpgai/reporting/training_convergence.py`
- recommendation: `manual_cli_tool_review`
- lines: 361
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Collect training convergence reports from existing FPGAI experiment outputs.  Outputs:   reports/training_convergence/training_convergence.csv   reports/training_convergence/training_convergence.md   reports/training_convergence/training_convergence.

## `fpgai/reporting/vivado_impl_artifacts.py`
- recommendation: `manual_cli_tool_review`
- lines: 448
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: ['evidence']
- doc: Collect Vivado implementation reports from FPGAI experiment artifacts.  Vivado implementation report goals:   - Do not rerun Vivado.   - Summarize only existing artifacts.   - Prefer Vivado report files over broad JSON crawling.   - Avoid parsing HLS

## `fpgai/runtime/binary_io.py`
- recommendation: `manual_cli_tool_review`
- lines: 94
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/runtime/bitstream.py`
- recommendation: `manual_cli_tool_review`
- lines: 36
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/runtime/hostcpp.py`
- recommendation: `manual_cli_tool_review`
- lines: 32
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/runtime/kv260.py`
- recommendation: `manual_cli_tool_review`
- lines: 34
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/validation/onnx_compare.py`
- recommendation: `manual_cli_tool_review`
- lines: 291
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/validation/verify_flow.py`
- recommendation: `manual_cli_tool_review`
- lines: 187
- has_main: True
- uses_argparse: True
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/experiments/model_generation.py`
- recommendation: `manual_review`
- lines: 73
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/experiments/model_suite.py`
- recommendation: `manual_review`
- lines: 319
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/compare_hardware_knobs.py`
- recommendation: `reporting_merge_candidate`
- lines: 172
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: ['evidence']
- doc: Compare FPGAI hardware knob evidence.  Reads vivado_bridge_evidence/evidence.json and optional results.json from an experiment directory, classifies design names into precision/parallel/pipeline families, and writes a CSV + Markdown comparison table.

## `fpgai/reporting/inspect_pipeline_policy_artifacts.py`
- recommendation: `reporting_merge_candidate`
- lines: 63
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/memory_binding_artifacts.py`
- recommendation: `reporting_merge_candidate`
- lines: 100
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []
- doc: Extract memory binding artifacts from generated HLS artifacts.

## `fpgai/reporting/parallel_policy_artifacts.py`
- recommendation: `reporting_merge_candidate`
- lines: 251
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: ['evidence']

## `fpgai/reporting/training_accelerator_artifacts.py`
- recommendation: `reporting_merge_candidate`
- lines: 205
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/training_accumulated_batch_artifacts.py`
- recommendation: `reporting_merge_candidate`
- lines: 218
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: ['truth']
- doc: Extract accumulated mini-batch training artifacts.  This extractor is intentionally conservative: - It only marks accumulated_batch=True when the design name/config indicates   an accumulated-batch design, the HLS artifacts exist, and the manifest ha

## `fpgai/reporting/training_batch_multistep_artifacts.py`
- recommendation: `reporting_merge_candidate`
- lines: 140
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/training_multi_epoch_convergence_artifacts.py`
- recommendation: `reporting_merge_candidate`
- lines: 246
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/training_native_accumulated_batch_artifacts.py`
- recommendation: `reporting_merge_candidate`
- lines: 108
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: []

## `fpgai/reporting/vivado_bridge_artifacts.py`
- recommendation: `reporting_merge_candidate`
- lines: 401
- has_main: True
- uses_argparse: False
- python_m_refs: 0
- import_refs: 0
- old_terms: ['evidence']
- doc: Extract Vivado/Vitis HLS reports from FPGAI experiment artifacts.  This extractor is intentionally defensive because different sprints may produce slightly different directory layouts:    artifacts/<design>/build/vivado_bridge/...   artifacts/<design
