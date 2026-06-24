# FPGAI CLI Workflows

This document defines the public command-line workflows for FPGAI. The `fpgai` CLI is the user-facing interface. Files under `scripts/` are transitional compatibility or developer tools and should not be the normal workflow in documentation.

## 1. Inspect a single compile config

Use inspect before compile to check the model/configuration without running the compiler backend.

```bash
fpgai inspect --config configs/examples/inference_compile.yml
```

`fpgai inspect` is pure inspection. It must not run Vitis HLS, Vivado, benchmark, or compile.

Inspect can also write pre-HLS model/resource/timing prediction artifacts:

```bash
fpgai inspect \
  --config configs/examples/inference_compile.yml \
  --out reports/inference_prediction
```

This writes:

```text
reports/inference_prediction/model_profile.json
reports/inference_prediction/resource_prediction.json
reports/inference_prediction/timing_prediction.json
reports/inference_prediction/prediction_summary.md
```

These files are estimates generated before HLS/Vivado.

## 2. Quick compile without HLS/Vivado

Use the quick compile example to validate the front half of the pipeline without requiring Vitis HLS or Vivado:

```bash
fpgai compile --config configs/examples/quick_compile.yml --quiet
```

This writes the manifest, IR/planning artifacts, model profile, HLS artifact metadata, runtime package metadata, and pre-HLS resource/timing prediction artifacts under:

```text
build/fpgai_quick_compile/
```

A normal compile also writes prediction artifacts under:

```text
<project.out_dir>/reports/model_profile.json
<project.out_dir>/reports/resource_prediction.json
<project.out_dir>/reports/timing_prediction.json
<project.out_dir>/reports/prediction_summary.md
```

## 3. Compile and benchmark a full inference config

Use this for one model/configuration.

```bash
fpgai compile --config configs/examples/inference_compile.yml
fpgai benchmark --config configs/examples/inference_compile.yml
```

When enabled in the YAML config, compile can generate host C++ artifacts, HLS artifacts, design-space reports, and HLS/Vitis summaries.

## 4. Compile with design-space exploration

The full inference example enables design-space exploration:

```bash
fpgai compile --config configs/examples/inference_compile.yml --quiet
```

When `analysis.design_space.enabled: true`, compile writes estimate-based DSE artifacts under:

```text
<project.out_dir>/design_space/results.json
<project.out_dir>/design_space/results.csv
<project.out_dir>/design_space/layer_breakdown.csv
<project.out_dir>/design_space/summary.txt
```

The compile manifest records DSE artifact paths, analytical model names, recommendation policy, recommendation scope, and recommended candidates. DSE is recommendation-only: it evaluates configured YAML candidates and does not run an exhaustive search optimizer. Recommendations are pre-HLS estimates unless compared with HLS/Vivado reports separately.

## 5. HLS/Vitis artifacts

When HLS is enabled and Vitis HLS runs, compile records grouped HLS artifacts in `manifest.json` under `hls_artifacts`.

The grouped HLS manifest block includes:

```text
hls_project_dir
stdout_log
stderr_log
csynth_report
schedule_summary
artifact_metadata
ii_comparison
```

The corresponding files are emitted under the compile output directory when available:

```text
hls_artifact_metadata.json
hls_schedule_summary.json
hls_ii_comparison.json
```

These artifacts describe generated HLS files, discovered schedule reports, and requested-versus-achieved II comparisons when vendor reports are available.

## 6. Vivado bridge board support

Vivado bridge generation is separate from the main compile command.

FPGAI supports board-aware Vivado bridge generation for these board keys:

```text
pynq_z2
kv260
kr260
```

The generated block-design Tcl is board-specific:

```text
pynq_z2  -> processing_system7 / Zynq-7000
kv260    -> zynq_ultra_ps_e / Zynq UltraScale+ MPSoC
kr260    -> zynq_ultra_ps_e / Zynq UltraScale+ MPSoC
```

Generate bridge artifacts for a compiled design:

```bash
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board pynq_z2 --export-hls-ip
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board kv260 --export-hls-ip
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board kr260 --export-hls-ip
```

Request Vivado synthesis:

```bash
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board pynq_z2 --export-hls-ip --run-vivado-synth
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board kv260 --export-hls-ip --run-vivado-synth
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board kr260 --export-hls-ip --run-vivado-synth
```

Request Vivado implementation/bitstream generation:

```bash
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board pynq_z2 --export-hls-ip --run-vivado-impl
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board kv260 --export-hls-ip --run-vivado-impl
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board kr260 --export-hls-ip --run-vivado-impl
```

Bitstream and XSA status are only treated as present when generated files are found in the Vivado bridge artifacts. Board-specific Tcl generation is tested in the repository; actual Vivado implementation success must be validated with the installed Vivado version, board files, and target hardware.

## 7. Runtime package

A compile run emits a runtime package manifest under:

```text
<project.out_dir>/runtime_package/package_manifest.json
```

The runtime package copies runtime-facing artifacts that already exist, such as:

```text
runtime_package/manifest.json
runtime_package/inputs/input.bin
runtime_package/hls/hls_artifact_metadata.json
runtime_package/hls/hls_schedule_summary.json
runtime_package/hls/hls_ii_comparison.json
runtime_package/hardware/*.bit
runtime_package/hardware/*.hwh
runtime_package/hardware/*.xsa
```

The package is truth-preserving. It does not run Vivado or deploy to hardware. It records whether bitstream, HWH, and XSA files are present:

```text
bitstream_present
hwh_present
xsa_present
deployable_overlay_present
```

For quick compile flows without Vivado implementation, these hardware fields are expected to be false.

You can also create the package directly from an output directory:

```bash
python -m fpgai.runtime.package <project.out_dir> --board kv260 --pipeline-mode inference --top-name deeplearn
```

## 8. Sweep config

Use this for precision/policy/design-space sweeps.

```bash
fpgai sweep inspect --config configs/sweeps/inference_precision.yml

fpgai sweep run \
  --config configs/sweeps/inference_precision.yml \
  --out experiments/inference_precision \
  --max-design-points 1 \
  --timeout-sec 1200
```

Sweep output goes under `experiments/`. Generated experiment artifacts should not normally be committed.

## 9. Experiment config

Use this workflow for reproducibility and paper-result generation.

Inspect an experiment configuration before running it:

```bash
fpgai experiment inspect --config configs/experiments/arxiv_paper.yml
```

Run the experiment configuration and write generated outputs under `paper_experiments/`:

```bash
fpgai experiment run \
  --config configs/experiments/arxiv_paper.yml \
  --out paper_experiments/arxiv
```

Generated experiment outputs should go under `paper_experiments/` or `experiments/`, not into tracked source directories.

## 10. Report generation

Use `fpgai report` to turn existing experiment outputs into reviewer-facing summaries, tables, plots, and traceability reports. These commands reuse the implementation under `fpgai/reporting/`; users do not need to call internal scripts directly.

Build a summary report from an experiment output directory:

```bash
fpgai report build \
  --input experiments/inference_precision \
  --out reports/inference_precision
```

Generate paper tables and figures from a sweep/result CSV:

```bash
fpgai report paper-artifacts \
  --csv experiments/inference_precision/policy_sweep_results.csv \
  --out reports/inference_precision/paper_artifacts
```

Generate Pareto/frontier artifacts:

```bash
fpgai report frontier \
  --csv experiments/inference_precision/policy_sweep_results.csv \
  --out reports/inference_precision/frontier
```

Generate estimator-vs-real resource and latency tables:

```bash
fpgai report estimator \
  --csv experiments/inference_precision/policy_sweep_results.csv \
  --out reports/inference_precision/estimator
```

Report outputs should go under `reports/` and should not normally be committed.

## 11. Logging policy

Default compile and benchmark commands should print concise FPGAI summaries and paths to logs. Full Vitis/Vivado output should only be shown with `--verbose`.

```bash
fpgai compile --config configs/examples/inference_compile.yml
fpgai compile --config configs/examples/inference_compile.yml --verbose
fpgai compile --config configs/examples/inference_compile.yml --quiet
```
