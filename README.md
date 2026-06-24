# FPGAI

FPGAI is a research compiler framework for turning ONNX neural-network models into FPGA/SoC accelerator projects. It focuses on YAML-driven compilation, pre-HLS model/resource/timing prediction, Vitis HLS project generation, design-space exploration, Vivado bridge artifacts, runtime package metadata, and reproducible experiment workflows.

## Who is this for?

- Researchers building FPGA AI compiler flows.
- Engineers experimenting with Xilinx FPGA/SoC deployment.
- Contributors adding operators, optimization policies, backends, and report tools.
- Authors preparing reproducible FPGA/compiler experiments.

## Current public workflow

FPGAI is used through the `fpgai` command-line interface. Normal users should not call internal files under `scripts/` directly.

Install locally:

```bash
pip install -e .
```

Inspect a single compile config:

```bash
fpgai inspect --config configs/examples/inference_compile.yml
```

Run a quick compile without Vitis HLS or Vivado:

```bash
fpgai compile --config configs/examples/quick_compile.yml
```

The quick compile example runs the YAML-first front half of the pipeline. It emits a manifest, IR/planning artifacts, model profile, pre-HLS resource/timing prediction artifacts, HLS artifact metadata, and runtime package metadata under `build/fpgai_quick_compile/`.

Run a benchmark flow:

```bash
fpgai benchmark --config configs/examples/inference_compile.yml
```

Inspect a sweep config:

```bash
fpgai sweep inspect --config configs/sweeps/inference_precision.yml
```

Run a small sweep:

```bash
fpgai sweep run \
  --config configs/sweeps/inference_precision.yml \
  --out experiments/inference_precision \
  --max-design-points 1 \
  --timeout-sec 1200
```

Inspect an experiment config:

```bash
fpgai experiment inspect --config configs/experiments/arxiv_paper.yml
```

## Main generated artifacts

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

The compile summary is manifest-backed. It reports prediction artifacts, HLS artifact status, Vivado bridge status, runtime package status, and compact pipeline stage statuses.

## Vivado bridge support

Vivado bridge generation is separate from the main compile command. Supported board keys are:

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

Generate Vivado bridge artifacts for a compiled output:

```bash
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board pynq_z2 --export-hls-ip
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board kv260 --export-hls-ip
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board kr260 --export-hls-ip
```

Request Vivado synthesis or implementation when Vivado and board files are installed:

```bash
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board pynq_z2 --export-hls-ip --run-vivado-synth
python -m fpgai.backends.vivado.run_bridge <experiment_or_build_dir> --board kv260 --export-hls-ip --run-vivado-impl
```

Bitstream, HWH, and XSA files are only reported as present when generated files exist.

## Runtime package

Compile emits runtime package metadata under:

```text
<out_dir>/runtime_package/package_manifest.json
```

The runtime package copies runtime-facing artifacts that already exist and records truthful hardware handoff status:

```text
bitstream_present
hwh_present
xsa_present
deployable_overlay_present
```

For quick compile flows without Vivado implementation, these hardware fields are expected to be false.

## YAML config types

FPGAI uses separate YAML schemas for separate workflows.

| Location | Purpose | Inspect command |
|---|---|---|
| `configs/examples/*.yml` | Single compile/benchmark configs | `fpgai inspect --config ...` |
| `configs/sweeps/*.yml` | Sweep and design-space configs | `fpgai sweep inspect --config ...` |
| `configs/experiments/*.yml` | Reproducibility experiment configs | `fpgai experiment inspect --config ...` |

## Repository structure

```text
fpgai/                Python package and compiler implementation
configs/examples/     Public single-run example configs
configs/sweeps/       Sweep/matrix configs
configs/experiments/  Reproducibility experiment configs
docs/                 User and developer documentation
models/               Small ONNX models used by examples/tests
tests/                Unit and integration tests
scripts/              Transitional compatibility/developer tools only
```

Generated build outputs are written under `build/` or the configured `project.out_dir`. Generated sweep outputs are written under `experiments/`. These generated outputs should normally not be committed.

## Supported flow today

FPGAI currently supports:

- ONNX model import and graph inspection.
- YAML-driven compile configuration.
- Pre-HLS model/resource/timing prediction artifacts.
- Vitis HLS project generation.
- HLS CSim/synthesis flow when enabled in the config.
- Correctness benchmarking against ONNX Runtime for supported inference paths.
- Precision, pipeline, and parallel-policy sweeps through `fpgai sweep run`.
- Design-space metadata and recommendations when DSE is enabled.
- Board-aware Vivado bridge generation for PYNQ-Z2, KV260, and KR260.
- Runtime package metadata emitted from compile outputs.
- Quiet CLI logging by default, with full tool output available through `--verbose`.

## Important limitations

FPGAI is an active research compiler. Public claims should follow generated artifacts.

- Physical-board runtime benchmarking is not the default public workflow.
- Training-code generation exists, but full numerical training convergence remains a validation area.
- Resource and timing predictions are exported as explicit pre-HLS estimate artifacts; calibration and HLS/Vivado comparison reports remain separate validation steps.
- Communication optimization currently distinguishes modeled transfer planning from measured board-level DMA runtime.

## Development and contribution

Start with the CLI workflows above. For development, run:

```bash
pytest -q
```

When adding a feature, include:

- YAML config coverage when applicable.
- Unit tests or integration smoke tests.
- Documentation under `docs/`.
- A clear statement of generated artifacts and limitations.

## License

FPGAI is released for academic, educational, and non-commercial research use under the terms in `LICENSE.md`. Commercial use, redistribution for commercial purposes, and proprietary integration require separate written permission.
