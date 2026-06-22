# Paper Artifact Policy

Paper claims must be evidence-first.

A claim is allowed only when it is supported by at least one of:

1. generated code,
2. HLS CSim output,
3. Vivado reports,
4. physical-board runtime logs,
5. reviewer-facing CSV/JSON/Markdown evidence generated from repository artifacts.

## Claim levels

| Level | Meaning |
|---|---|
| supported | Direct artifact evidence exists |
| partial | Some evidence exists, but the limitation must be stated |
| unsupported | Do not use in the paper |

## Current known limitations

- HLS training loss curves are not physical FPGA loss curves.
- Communication ablation currently reports transfer-volume modeling/static artifact evidence, not measured board DMA speedup.
- Estimator accuracy is partial unless non-placeholder LUT/FF/BRAM/DSP predictions are available.
- Safe-clock guidance is reported but FPGAI does not automatically rewrite clock constraints.
