# FPGAI

**FPGAI** is a resource-aware FPGA-SoC compilation framework for neural-network inference and training experiments. It imports ONNX models, builds an internal graph representation, generates Vitis HLS accelerator projects, materializes compiler policies into generated HLS code, and produces reproducible correctness and evidence tables for FPGA/compiler research.

> **License note:** FPGAI is released for **academic and non-commercial research use only**. Commercial use, product integration, paid services, redistribution for commercial purposes, and use in proprietary products require prior written permission from the author. See [`LICENSE.md`](LICENSE.md).

---

## What FPGAI is for

FPGAI is designed for researchers and engineers who want to study how neural-network accelerator design choices affect correctness, latency, resource usage, and training behavior on FPGA-SoC targets.

The project currently focuses on:

- ONNX-to-Vitis-HLS project generation.
- Fixed-point neural-network inference experiments.
- Runtime weight-loading strategies.
- Memory binding strategies such as BRAM and URAM.
- Parallelization-policy materialization.
- HLS CSim correctness validation.
- One-step and small multi-step training accelerator smoke tests.
- Reproducible paper-style experiment automation.

FPGAI should be treated as a research compiler, not a general production deployment SDK.

---

## Current verified capabilities

The following capabilities have automated evidence in the current development flow.

### Inference and HLS generation

- ONNX model import for the evaluated CNN/MLP-style workloads.
- Internal graph/IR construction and compile planning.
- Vitis HLS project generation.
- Auto-generated HLS top functions and testbenches.
- HLS CSim execution through generated TCL scripts.
- Correctness benchmarking against Python/ONNX reference outputs.
- Intermediate layer debugging and first-bad-layer style analysis.

### Weight and memory strategies

FPGAI supports evaluated weight-placement modes:

- `embedded`: weights compiled into generated C++ artifacts.
- `stream`: weights provided at runtime through a preload path/stream.
- `ddr`: external-memory style runtime weight access for supported experiments.

FPGAI also supports HLS storage binding evidence for:

- BRAM binding.
- URAM binding.

### Parallelization policies

The compiler can materialize policy-driven HLS parameters for evaluated designs, including:

- resource-first policy,
- balanced policy,
- throughput-first policy,
- latency-first policy,
- layer-specific unroll and partition factors,
- PE/SIMD-style generated evidence comments.

### Training accelerator experiments

FPGAI currently supports validated HLS CSim smoke tests for CNN training accelerators:

- one-step training validation,
- embedded-weight training CSim,
- streamed runtime-weight training CSim,
- HLS-side artifact emission:
  - `weights_before.bin`,
  - `grads.bin`,
  - `weights_after.bin`,
- comparison against Python reference training artifacts,
- small multi-step and batch-replay smoke tests.

The current training evidence supports **small HLS training smoke tests**, not full autonomous multi-epoch training convergence on arbitrary datasets.

---

## What FPGAI does not yet claim

To avoid overclaiming, the current version does **not** claim:

- full arbitrary ONNX support,
- arbitrary model training,
- full multi-epoch FPGA training convergence,
- globally optimal hardware search,
- full production deployment on physical boards,
- complete equivalence with FINN or hls4ml baselines,
- vendor-independent FPGA support.

These are roadmap items and should only be claimed after generated-code evidence and passing experiment evidence exist.

---

## Repository structure

```text
fpgai/
  benchmark/        Benchmarking, reference comparison, metrics
  backends/         HLS and host-code generation
  config/           YAML config loading and validation
  engine/           Compiler pipeline, planning, memory, training flow
  frontend/         ONNX import
  ir/               Graph IR and compiler passes
  util/             Helper utilities

configs/
  sweeps/           Reproducible experiment sweep definitions
  suite/            Example configs and experiment setups

scripts/
  run_fpgai_experiments.py
  extract_*_evidence.py
  collect_paper_evidence.py
  diagnostic and utility scripts

models/
  Example ONNX models

tests/
  Unit and integration tests

README.md
LICENSE.md
requirements.txt
pyproject.toml
setup.py
```

---

## Requirements

Typical development environment:

- Python 3.10+
- `numpy`
- `onnx`
- `onnxruntime`
- `pyyaml`
- `matplotlib` for plotting and paper figures
- Xilinx Vitis HLS, tested around 2023.2-style flows
- Vivado/Vitis installation when generating or running HLS projects

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the requirements file is incomplete for a fresh environment, install the core dependencies manually:

```bash
pip install numpy onnx onnxruntime pyyaml matplotlib
```

---

## Basic usage

Run the compiler with a YAML config:

```bash
python3 main.py --config fpgai.yml
```

Typical pipeline:

1. Load the ONNX model.
2. Build the internal graph/IR.
3. Generate compile, memory, and communication plans.
4. Generate Vitis HLS source and TCL artifacts.
5. Optionally run Vitis HLS CSim.
6. Optionally benchmark generated outputs against reference outputs.

---

## Example configuration fields

A typical configuration contains:

```yaml
version: 1

project:
  name: fpgai_example
  out_dir: build/fpgai_example
  clean: true

pipeline:
  mode: inference
  outputs:
    top_kernel_name: deeplearn

model:
  format: onnx
  path: models/cnn_mnist.onnx

targets:
  platform:
    board: kv260
    part: xck26-sfvc784-2LV-c
    clocks:
      - name: pl_clk0
        target_mhz: 200

numerics:
  defaults:
    activation: { type: ap_fixed, total_bits: 16, int_bits: 6 }
    weight:     { type: ap_fixed, total_bits: 16, int_bits: 6 }
    bias:       { type: ap_fixed, total_bits: 24, int_bits: 10 }
    accum:      { type: ap_fixed, total_bits: 24, int_bits: 10 }

data_movement:
  ps_pl:
    weights:
      mode: embedded   # embedded | stream | ddr

backends:
  hls:
    enabled: true
    vitis:
      enabled: true
      mode: csim
      exe: vitis_hls

benchmark:
  enabled: true
  fail_on_mismatch: true
```

---

## Output artifacts

Generated outputs are written under `project.out_dir`.

Important artifacts include:

```text
build/.../
  manifest.json
  input.bin
  target.bin
  ir/
    descriptors.json
    compile_plan.json
    memory_plan.json
    comm_plan.json
  hls/
    src/
      deeplearn.cpp
      tb.cpp
    include/
    run_hls.tcl
    logs/
    metadata/
  bench/
    metrics.json
    summary.txt
    reference_input.npy
    reference_output.npy
    hls_output.npy
  training_reference/
    weights_before_ref.bin
    grads_ref.bin
    weights_after_ref.bin
    summary.json
    summary.txt
  training_compare/
    results.json
    summary.txt
```

---

## Reproducible experiment workflow

FPGAI uses sweep configs and extraction scripts to keep claims evidence-based.

General workflow:

```bash
rm -rf experiments/<experiment_name>

PYTHONPATH="$PWD" python -B scripts/run_fpgai_experiments.py \
  --sweep configs/sweeps/<sweep>.yml \
  --out experiments/<experiment_name> \
  --max-design-points <N> \
  --timeout-sec 1800
```

Inspect results:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/<experiment_name>/results.json")
data = json.loads(p.read_text())
print("passed:", data.get("passed_count"))
print("failed:", data.get("failed_count"))
for r in data.get("results", []):
    print(r.get("design_name"), r.get("status"), r.get("returncode"), r.get("error"))
PY
```

Extract evidence:

```bash
python scripts/extract_training_accelerator_evidence.py experiments/<experiment_name>
python scripts/extract_training_batch_multistep_evidence.py experiments/<experiment_name>
python scripts/collect_paper_evidence.py
```

---

## Verified sprint evidence summary

The recent evidence-driven development flow validated:

### Sprint 12A — external DDR runtime memory mode

Validated embedded, streamed, and external-DDR runtime weight strategies end-to-end for the evaluated CNN configurations.

### Sprint 12B — BRAM/URAM storage binding

Validated HLS-level BRAM and URAM storage binding using generated `BIND_STORAGE` directives and Vitis HLS runs.

### Sprint 12C — parallel policy materialization

Validated resource-first, balanced, throughput-first, and latency-first parallelization policies with distinct generated HLS parameters.

### Sprint 12D — paper evidence collector

Generated paper-ready evidence tables for correctness, memory modes, memory binding, parallel policies, HLS metrics, and claim support.

### Sprint 13A — one-step training accelerator validation

Validated one-step training accelerator generation for embedded and streamed runtime-weight CNN configurations.

### Sprint 13B — streamed training CSim preload and comparison

Fixed streamed training CSim so runtime weights are loaded from `training_reference/weights_before_ref.bin`, enabling training comparison artifacts for stream mode.

### Sprint 13C — batch/multi-step training smoke tests

Validated small multi-step and batch-replay HLS training smoke tests for embedded and streamed runtime-weight modes.

---

## Training experiment commands

One-step training accelerator evidence:

```bash
rm -rf experiments/sprint13b_training_stream_compare_v7

PYTHONPATH="$PWD" python -B scripts/run_fpgai_experiments.py \
  --sweep configs/sweeps/sprint13a_training_accelerator.yml \
  --out experiments/sprint13b_training_stream_compare_v7 \
  --max-design-points 4 \
  --timeout-sec 1800

python scripts/extract_training_accelerator_evidence.py \
  experiments/sprint13b_training_stream_compare_v7
```

Batch/multi-step smoke evidence:

```bash
rm -rf experiments/sprint13c_training_batch_multistep_v3

PYTHONPATH="$PWD" python -B scripts/run_fpgai_experiments.py \
  --sweep configs/sweeps/sprint13c_training_batch_multistep.yml \
  --out experiments/sprint13c_training_batch_multistep_v3 \
  --max-design-points 6 \
  --timeout-sec 1800

python scripts/extract_training_batch_multistep_evidence.py \
  experiments/sprint13c_training_batch_multistep_v3
```

Expected Sprint 13C evidence:

```text
hls_ok=True
training_compare=True
has_multistep_summary=True
hls_weights_before_bin=True
hls_grads_bin=True
hls_weights_after_bin=True
```

---

## Current roadmap

Near-term roadmap:

1. True accumulated mini-batch SGD.
2. Tiny multi-epoch convergence smoke tests.
3. Operator coverage audit for the evaluated ONNX subset.
4. HLS resource/latency report normalization.
5. Combined precision/memory/parallel design-space sweeps.
6. FINN/hls4ml baseline comparison setup.
7. Physical KV260 deployment evidence.

Longer-term roadmap:

- Broader ONNX operator support.
- More complete training support.
- Physical board runtime integration.
- Vendor abstraction beyond Xilinx/Vitis.
- Resource-aware hardware search with stronger optimization guarantees.

---

## Development principle

FPGAI development follows this rule:

> Do not claim a feature unless there is generated-code evidence and passing experiment evidence.

For each new feature:

1. Add one claim.
2. Generate HLS artifacts.
3. Run a small sweep first.
4. Inspect `results.json`, HLS logs, and generated `tb.cpp`/`deeplearn.cpp`.
5. Extract `.json`, `.csv`, and `.md` evidence.
6. Only then update README/paper claims.

---

## Citation / attribution

If you use FPGAI in academic work, please cite the related FPGAI paper/preprint when available and acknowledge this repository.

Suggested placeholder citation:

```bibtex
@misc{altin_fpgai,
  title        = {FPGAI: A Resource-Aware Compilation and Execution Framework for Neural Network Inference and Training on FPGA-SoCs},
  author       = {Altin, Umut Can},
  year         = {2026},
  note         = {Research software, academic use only}
}
```

---

## License

This project is licensed for **academic and non-commercial research use only**. See [`LICENSE.md`](LICENSE.md).

For commercial licensing, industrial evaluation, paid services, product integration, or redistribution outside academic research, contact the author for written permission.
