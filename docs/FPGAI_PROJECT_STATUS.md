# FPGAI Project Status

This document records the current implementation state of the FPGAI repository. It is intentionally concise so the public repository does not look like an unfinished work log.

## Working rules

- Prefer YAML-first workflows through the public CLI.
- Inspect and reuse existing modules before adding new files.
- Keep public claims tied to implemented behavior and generated artifacts.
- Use tests during development; keep only useful public regression tests.
- Use “experiments”, “reports”, “artifacts”, “validation”, and “traceability” wording in public docs.

## Current target

FPGAI should provide a clear end-to-end compiler workflow:

```text
inspect config/model
→ estimate model/resource/timing before HLS
→ optionally run design-space exploration
→ generate host/HLS artifacts
→ optionally run Vitis HLS
→ optionally generate/run Vivado bridge artifacts
→ emit runtime package metadata
→ report truthful artifact status in CLI output and manifest
```

## Current implementation state

Implemented and tested:

- Config-first CLI workflows for compile, inspect, benchmark, sweep, experiment inspection, and report generation.
- Quick compile flow that does not require Vitis HLS or Vivado.
- Pre-HLS model profile, resource prediction, timing prediction, and prediction summary artifacts.
- Manifest-backed pipeline stage reporting.
- Design-space metadata in the compile manifest when DSE is enabled.
- HLS artifact grouping in the compile manifest.
- Board-aware Vivado bridge generation for:
  - `pynq_z2` using `processing_system7`;
  - `kv260` using `zynq_ultra_ps_e`;
  - `kr260` using `zynq_ultra_ps_e`.
- Runtime package emission from compile outputs:
  - `runtime_package/package_manifest.json`;
  - copied runtime-facing artifacts when present;
  - truthful bitstream/HWH/XSA presence flags.
- Quiet and normal compile summaries that surface manifest-backed artifact status.

## Truth boundaries

The repository should not imply more than the generated artifacts prove.

- Resource and timing predictions are pre-HLS estimates unless compared against HLS/Vivado reports.
- Vivado bridge generation is separate from the main compile command.
- Bitstream, HWH, and XSA files are only reported as present when actual files exist.
- Physical-board runtime benchmarking is not the default public workflow.
- Training-code generation exists, but full numerical training convergence remains a validation area.

## Important current artifacts

A normal compile can emit:

```text
<out_dir>/manifest.json
<out_dir>/reports/model_profile.json
<out_dir>/reports/resource_prediction.json
<out_dir>/reports/timing_prediction.json
<out_dir>/reports/prediction_summary.md
<out_dir>/hls_artifact_metadata.json
<out_dir>/runtime_package/package_manifest.json
```

Vivado bridge generation can emit:

```text
<out_dir>/vivado_bridge/scripts/export_hls_ip.tcl
<out_dir>/vivado_bridge/scripts/create_bd.tcl
<out_dir>/vivado_bridge/scripts/run_vivado.tcl
<out_dir>/vivado_bridge/vivado_bridge_manifest.json
```

## Latest update

- Public status document was condensed into a professional current-state summary.
- Old work-log language was removed from this file.
- Current feature boundaries are documented in terms of implemented artifacts and validation status.

## Next cleanup target

Review public docs and reporting modules for stale wording such as old work-log labels, old output names, and outdated runtime/Vivado limitations.
