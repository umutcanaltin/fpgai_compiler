# FPGAI Compiler

FPGAI is an academic research compiler framework for generating FPGA-oriented neural network accelerators from high-level model descriptions. The project focuses on resource-aware compilation for FPGA-SoC targets, with an emphasis on Vitis HLS code generation, configurable arithmetic precision, memory-placement strategies, parallelization policies, and experimental training-accelerator flows.

The framework is intended for research in embedded AI acceleration, FPGA compilation, design-space exploration, and hardware/software co-design.

## Overview

FPGAI provides a Python-based compilation and experiment workflow for transforming neural network models into HLS projects. The compiler is designed to make hardware-related choices explicit and reproducible, including numeric precision, weight storage strategy, storage binding, and layer-level parallelization.

The current implementation is centered on AMD/Xilinx FPGA-SoC workflows using Vitis HLS. Generated projects can be used for C simulation, synthesis-oriented inspection, and automated experiment collection.

## Key Features

- ONNX model import and intermediate representation construction
- Vitis HLS C++ accelerator generation
- Fixed-point precision configuration
- Embedded, streamed, and external-memory weight handling paths
- BRAM and URAM storage binding support
- Configurable parallelization and pipelining parameters
- HLS testbench generation for inference and training-oriented experiments
- Automated experiment sweeps through YAML configuration files
- Evidence and result extraction scripts for reproducible evaluation

## Repository Structure

```text
fpgai/                  Core compiler package
  backends/hls/          HLS code generation, testbenches, and TCL emitters
  engine/                Compilation orchestration
  frontend/              Model import and frontend utilities
  ir/                    Intermediate representation utilities

configs/                Example compiler and experiment configurations
configs/sweeps/         Sweep definitions for automated experiments
scripts/                Experiment runners and evidence extraction tools
models/                 Example neural network models
tests/                  Unit and regression tests
```

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Install development and test dependencies as needed:

```bash
pip install pytest numpy onnx onnxruntime pyyaml
```

For HLS compilation and C simulation, install a compatible AMD/Xilinx Vitis HLS environment and make sure `vitis_hls` is available on your `PATH`.

## Basic Usage

Run the test suite:

```bash
pytest
```

Run an experiment sweep:

```bash
PYTHONPATH="$PWD" python -B scripts/run_fpgai_experiments.py \
  --sweep configs/sweeps/<sweep-file>.yml \
  --out experiments/<experiment-name> \
  --max-design-points 4 \
  --timeout-sec 1800
```

Inspect experiment results:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("experiments/<experiment-name>/results.json")
data = json.loads(p.read_text())
print("passed:", data.get("passed_count"))
print("failed:", data.get("failed_count"))
for r in data.get("results", []):
    print(r.get("design_name"), r.get("status"), r.get("returncode"), r.get("error"))
PY
```

## Experiment Workflow

A typical FPGAI workflow is:

1. Select an ONNX model and target configuration.
2. Define the precision, memory, and parallelization policy in a YAML sweep.
3. Generate HLS code and testbench artifacts.
4. Run Vitis HLS C simulation or synthesis-oriented checks.
5. Extract metrics and generated-code evidence from the experiment directory.

Experiment outputs are written under `experiments/` and should normally not be committed to the repository.

## Development Notes

Before committing changes, run syntax checks and targeted tests:

```bash
python -B -m py_compile $(find fpgai scripts -name "*.py")
pytest
```

For large HLS experiments, start with a small number of design points before running a full sweep.

## Scope

FPGAI is research software. The project is intended to support reproducible compiler and accelerator-design experiments rather than serve as a general-purpose production deployment tool.

The current codebase is primarily oriented toward Vitis HLS and AMD/Xilinx FPGA-SoC targets. Support for additional FPGA vendors, broader ONNX operator coverage, board-level deployment flows, and extended training studies may require additional implementation and validation.

## Citation

If you use FPGAI in academic work, please cite the associated publication or repository when available. Citation metadata can be added in a future `CITATION.cff` file.

## License

This repository is distributed under the **FPGAI Academic Research Use License v1.0**. The software is available for academic and non-commercial research use only. Commercial use, product integration, paid services, and redistribution for commercial advantage require prior written permission from the copyright holder.

See [LICENSE.md](LICENSE.md) for the full license terms.

## Contact

For academic collaboration, research use, or commercial licensing inquiries, please contact the repository maintainer.
