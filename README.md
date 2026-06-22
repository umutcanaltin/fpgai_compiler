# FPGAI

FPGAI is an open-source compiler framework for turning ONNX neural-network models into FPGA/SoC accelerator projects. It focuses on YAML-driven compilation, Vitis HLS project generation, correctness benchmarking, design-space sweeps, and reproducible paper experiments.

## Who is this for?

- Researchers building FPGA AI compiler flows.
- Engineers experimenting with Xilinx FPGA/SoC deployment.
- Contributors adding operators, optimization policies, backends, and experiment/report tools.
- Authors preparing reproducible FPGA/compiler papers.

## Current public workflow

FPGAI is used through the `fpgai` command-line interface. Normal users should not call internal files under `scripts/` directly.

```bash
pip install -e .
```

Inspect a single compile config:

```bash
fpgai inspect --config configs/examples/inference_compile.yml
```

Compile a model:

```bash
fpgai compile --config configs/examples/inference_compile.yml
```

Run a benchmark flow:

```bash
fpgai benchmark --config configs/examples/inference_compile.yml
```

Inspect a sweep config:

```bash
fpgai sweep inspect --config configs/sweeps/inference_precision.yml
```

Run a sweep:

```bash
fpgai sweep run \
  --config configs/sweeps/inference_precision.yml \
  --out experiments/inference_precision \
  --max-design-points 1 \
  --timeout-sec 1200
```

Inspect a paper-experiment config:

```bash
fpgai experiment inspect --config configs/experiments/arxiv_paper.yml
```

## YAML config types

FPGAI uses separate YAML schemas for separate workflows.

| Location | Purpose | Inspect command |
|---|---|---|
| `configs/examples/*.yml` | Single compile/benchmark configs | `fpgai inspect --config ...` |
| `configs/sweeps/*.yml` | Design-space sweep configs | `fpgai sweep inspect --config ...` |
| `configs/experiments/*.yml` | Paper/reproducibility experiment configs | `fpgai experiment inspect --config ...` |

## Repository structure

```text
fpgai/                  Python package and compiler implementation
configs/examples/       Public single-run example configs
configs/sweeps/         Sweep/matrix configs
configs/experiments/          Paper experiments configs
docs/                   User and developer documentation
models/                 Small ONNX models used by examples/tests
tests/                  Unit and integration tests
scripts/                Transitional compatibility/developer tools only
```

Generated build outputs are written under `build/` or the configured `project.out_dir`. Generated sweep outputs are written under `experiments/`. These generated outputs should normally not be committed.

## Supported flow today

FPGAI currently supports:

- ONNX model import and graph inspection.
- YAML-driven compile configuration.
- Vitis HLS project generation.
- HLS CSim / synthesis flow where enabled in the config.
- Correctness benchmarking against ONNX Runtime for supported inference paths.
- Precision, pipeline, and parallel-policy sweeps through `fpgai sweep run`.
- Schema-specific inspection for compile, sweep, and paper-experiment YAML files.
- Quiet CLI logging by default, with full tool output available through `--verbose`.

## Important limitations

FPGAI is an active research compiler. Some flows are still being strengthened. In particular:

- Physical-board runtime benchmarking is not yet the default public workflow.
- Training-code generation exists, but full numerical training correctness is still being hardened.
- Resource estimation is being moved toward explicit exported predictions and calibrated reports.
- Communication optimization currently distinguishes modeled transfer planning from measured board-level DMA runtime.

These are implementation targets, not abandoned features.

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

See `LICENSE.md`.
