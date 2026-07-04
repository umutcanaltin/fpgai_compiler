# Example artifact smoke validation

Artifact smoke validation checks FPGAI compile output directories after examples are compiled. It does not run Vitis HLS, Vivado, or FPGA board execution. It reads generated artifacts and truth reports, then separates compiler-estimated evidence from HLS/Vivado/bitstream/FPGA truth.

## Batch 2 smoke command

```bash
python -m fpgai.cli compile --config examples/inference/mnist_mlp_embedded.yml
python -m fpgai.cli compile --config examples/inference/mnist_mlp_import_weights.yml
python -m fpgai.cli compile --config examples/training/mnist_mlp_training_sgd.yml
python -m fpgai.cli compile --config examples/build/cpp_only.yml
python -m fpgai.cli compile --config examples/build/hls_project.yml
python -m fpgai.cli compile --config examples/build/vivado_project.yml
python -m fpgai.reporting.artifact_smoke   build/examples/mnist_mlp_embedded   build/examples/mnist_mlp_import_weights   build/examples/mnist_mlp_training_sgd   build/examples/cpp_only   build/examples/hls_project   build/examples/vivado_project   --output build/examples/artifact_smoke_suite_batch2.json
```

## What must be present

For normal compiler-estimated examples, the audit expects:

- `manifest.json`
- `hls/` and `hls/src/`
- `runtime_package/package_manifest.json` when runtime package is requested
- config, generated C++, data movement, movement validation, board-fit, HLS truth, Vivado truth, and bitstream truth reports

When `build.stages.vivado_project=true`, the audit also requires:

- `vivado/project.tcl`
- `vivado/bd.tcl`

## Truth boundary

A passed artifact smoke report proves generated artifacts and reports are structurally present and internally consistent. It does not prove HLS synthesis, Vivado implementation, bitstream generation, or real FPGA execution unless the corresponding truth report has real evidence.

## Training truth reports

Training compile outputs must emit the same HLS/Vivado/bitstream truth-status report files as inference outputs. If the stage is not requested, the report should exist with `not_requested` or equivalent non-paper-safe status rather than being missing.
