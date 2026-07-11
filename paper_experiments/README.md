# FPGAI paper experiments

This directory freezes the paper experiment setup before adding more plots.

The matrix is organized into two sections:

1. **Inference first**: precision, parallelism, pipeline, weight import/embedded mode, deployability, and real board-runtime candidate rows.
2. **Training second**: optimizer, loss, tiling, batch/accumulation, deployability, and real board-runtime training-curve candidate rows.

The YAML rows intentionally separate artifact claim levels:

- `level_2_vivado_implementation`: compile through HLS and Vivado implementation, but do not claim bitstream/runtime.
- `level_3_bitstream_package`: generate `.bit`, `.hwh`, `.xsa`, Python driver, buffer plan, and runtime-package validation.
- `level_4_board_execution`: reserved rows for real KV260 runtime data. These rows must not claim runtime success until a board-runtime report exists.

Run the setup report:

```bash
python -m fpgai.paper.experiment_setup paper_experiments/paper_experiment_matrix.yml --output-dir paper_results/experiment_setup
```

Then compile selected rows with the command plan emitted by the setup report, and regenerate plots:

```bash
python -m fpgai.paper.plots build --output-dir paper_results/plots
```
