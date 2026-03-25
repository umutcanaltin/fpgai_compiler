# FPGAI

FPGAI is an open-source compiler framework that converts ONNX neural network models into deployable FPGA/SoC accelerator projects, with automatic HLS generation, runtime packaging, correctness benchmarking, and experiment automation for compiler-policy exploration.

The project is designed for:

- researchers building custom FPGA AI pipelines,
- engineers deploying models on Xilinx SoCs,
- contributors adding new operators, dataflows, benchmarking features, and compiler optimization policies,
- authors preparing reproducible FPGA/compiler papers with automated result generation.

---

## Features

### Core compiler flow

- ONNX model import
- layer-wise graph analysis
- resource-aware compile planning
- memory placement planning
- communication planning
- automatic Vitis HLS project generation
- embedded or streamed weight support
- auto-generated HLS testbench
- host/runtime artifact generation

### Correctness and debugging

- end-to-end correctness benchmarking against ONNX ground truth
- intermediate layer-by-layer error tracing
- first bad layer detection
- numeric error metrics
- benchmark summaries and JSON outputs for automation

### Precision and compiler optimization workflow

- configurable fixed-point numerics
- per-type control for activation, weight, bias, and accumulator precision
- automated experiment scripting for multiple models
- support for precision-policy sweeps
- support for parallelization-policy sweeps
- CSV aggregation of experiment results
- automatic plot generation for paper figures
- summary scripts for selecting best policies

### Extensibility

- extensible operator flow for contributor-added layers
- modular frontend / engine / backend layout
- reusable benchmarking pipeline
- clear generated artifact structure

---

## Project Structure

```text
fpgai/
  benchmark/        # benchmark and comparison pipeline
  backends/         # HLS / host code generation
  config/           # yaml config loading and validation
  engine/           # analysis, planning, memory, communication, compile flow
  frontend/         # ONNX import
  ir/               # graph IR and passes
  util/             # helper utilities

configs/
  suite/            # example configs and experiment setups

scripts/
  ...               # utility scripts, sweeps, plotting, summaries

build/
  ...               # generated artifacts

models/
  ...               # ONNX models

main.py             # main compiler entrypoint
fpgai.yml           # user configuration
README.md           # project documentation
```

---

## Requirements

Typical environment:

- Python 3.10+
- `numpy`
- `onnx`
- `onnxruntime`
- `pyyaml`
- `matplotlib` for plotting scripts
- Xilinx Vitis HLS 2023.2
- Vivado 2023.2

If you use benchmark and HLS generation, make sure Vitis HLS is installed and accessible.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/umutcanaltin/fpgai_compiler.git
cd fpgai_compiler
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you do not yet have a complete `requirements.txt`, install at least:

```bash
pip install numpy onnx onnxruntime pyyaml matplotlib
```

---

## Basic Usage

Run the compiler with a YAML configuration:

```bash
python3 main.py --config fpgai.yml
```

This will:

1. load the ONNX model,
2. analyze and plan the graph,
3. generate HLS and host artifacts,
4. optionally run Vitis HLS,
5. optionally run correctness benchmark.

---

## Main Execution Modes

FPGAI can be used in several different ways.

### 1. Compile-only mode

Disable benchmark in YAML:

```yaml
benchmark:
  enabled: false
```

Then run:

```bash
python3 main.py --config fpgai.yml
```

This will generate HLS and host artifacts and optionally run Vitis HLS.

### 2. Compile + benchmark mode

Enable benchmark in YAML:

```yaml
benchmark:
  enabled: true
```

Then run:

```bash
python3 main.py --config fpgai.yml
```

This will:

1. generate ONNX reference input/output,
2. compile and run HLS on the same input,
3. compare final outputs,
4. optionally compare intermediate layer outputs.

### 3. Precision exploration workflow

Use this when comparing fixed-point settings.

Typical use:

- change `numerics.defaults`,
- compile and benchmark,
- compare errors and hardware metrics across runs.

### 4. Compiler-policy sweep workflow

Use this when exploring compiler strategies such as:

- precision policies,
- parallelization policies,
- resource-first vs latency-first tradeoffs.

This mode is useful for:

- ASAP / FPGA / compiler papers,
- compiler optimization studies,
- reproducible design-space exploration.

### 5. Contributor / operator development mode

Use this when adding or debugging new operators.

Recommended process:

- compile-only first,
- final benchmark next,
- intermediate benchmark last,
- fix the first bad layer before changing thresholds.

---

## Configuration

The compiler is controlled by `fpgai.yml`.

Example:

```yaml
version: 1

project:
  name: fpgai_example_dense
  out_dir: build/fpgai_example_dense
  clean: true

pipeline:
  mode: inference
  outputs:
    top_kernel_name: deeplearn

targets:
  platform:
    board: kv260
    part: xck26-sfvc784-2LV-c
    clocks:
      - name: pl_clk0
        target_mhz: 200

operators:
  supported:
    - Dense
    - Conv
    - MaxPool
    - AvgPool
    - Add
    - Relu
    - LeakyRelu
    - Sigmoid
    - Softmax
    - BatchNormalization
    - Flatten
    - Reshape

  defaults:
    activation_insert:
      kind: none
      alpha: 0.1
      except_last: true

model:
  format: onnx
  path: models/cnn_mnist.onnx

numerics:
  defaults:
    activation: { type: ap_fixed, total_bits: 16, int_bits: 6 }
    weight:     { type: ap_fixed, total_bits: 16, int_bits: 6 }
    bias:       { type: ap_fixed, total_bits: 24, int_bits: 10 }
    accum:      { type: ap_fixed, total_bits: 24, int_bits: 10 }

data_movement:
  ps_pl:
    compression:
      enabled: true
    weights:
      mode: embedded   # embedded | stream | ddr

backends:
  hls:
    enabled: true
    vitis:
      enabled: true
      mode: csim
      exe: vitis_hls

toolchain:
  vitis_hls:
    enabled: true
    settings64: /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh

benchmark:
  enabled: true
  fail_on_mismatch: true
  seed: 0

  compare:
    atol: 0.08
    rtol: 0.08
    max_abs_error: 0.08
    mean_abs_error: 0.03
    rmse: 0.04
    require_argmax_match: false
    min_cosine_similarity: 0.95

  intermediate:
    enabled: true
    fail_on_layer_mismatch: false
    stop_on_first_bad_layer: false

debug:
  verbose: false
```

---

## Output Artifacts

Generated files are written under `project.out_dir`.

Important outputs include:

```text
build/.../
  manifest.json
  input.bin
  ir/
    descriptors.json
    compile_plan.json
    memory_plan.json
    comm_plan.json
  hls/
    src/
    include/
    run_hls.tcl
    logs/
    metadata/
  hostcpp/
  bench/
    metrics.json
    summary.txt
    reference_input.npy
    reference_output.npy
    hls_output.npy
    intermediate/
      intermediate_metrics.json
      intermediate_summary.txt
```

Additional experiment outputs may include:

- sweep CSV files,
- generated plots,
- report summaries,
- HLS synthesis metric extracts.

---

## Benchmarking

FPGAI can automatically compare generated HLS results against ONNX reference outputs.

### Final-output benchmark

This compares:

- ONNX output
- HLS CSIM output

Metrics include:

- max absolute error
- mean absolute error
- RMSE
- cosine similarity
- optional argmax agreement

### Intermediate benchmark

This compares intermediate outputs layer by layer and reports:

- first bad layer
- worst layers by error
- skipped layout-sensitive layers if needed

This is useful for debugging where divergence starts.

Run benchmark by enabling it in YAML:

```yaml
benchmark:
  enabled: true
```

Then run:

```bash
python3 main.py --config fpgai.yml
```

---

## Interpreting Benchmark Results

### Final benchmark passes

This means end-to-end generated inference is numerically close enough to ONNX based on your thresholds.

### Final benchmark fails

Check:

```text
build/.../bench/summary.txt
build/.../bench/metrics.json
```

Common causes:

- overly strict thresholds,
- wrong weight source in stream mode,
- final softmax sensitivity,
- data layout mismatch before Dense.

### Intermediate benchmark shows first bad layer

This usually means:

- layout mismatch before that layer,
- arithmetic precision issue at that layer,
- or operator implementation bug.

Example:

```text
first_bad_layer : dense0
```

This suggests early feature extraction is correct, and divergence begins when features are flattened or consumed by Dense.

---

## Intermediate Benchmark Notes

Current intermediate comparison is layout-aware for common CNN inference paths.

### Compared directly

- Dense
- Softmax

### Compared after ONNX layout normalization

- Conv
- Relu
- MaxPool
- AvgPool

### Currently skipped or treated specially

- Flatten
- Reshape

These can be layout-sensitive because ONNX tensors are usually NCHW, while internal HLS buffers may use HWC-flat ordering.

This is expected and not necessarily a bug.

---

## Weight Modes

FPGAI currently supports multiple weight modes:

### `embedded`

Weights are emitted directly into generated C++ parameter files.

Use this first when validating correctness.

### `stream`

Weights are provided at runtime through a preload stream.

Useful for runtime-loaded models, but benchmarking requires the real weights to be loaded in correct order.

### `ddr`

Reserved or evolving mode for external memory-backed parameter movement.

---

## Recommended Validation Workflow

### Step 1: start with embedded weights

Set:

```yaml
data_movement:
  ps_pl:
    weights:
      mode: embedded
```

This is the easiest mode for correctness validation.

### Step 2: run benchmark

```bash
python3 main.py --config fpgai.yml
```

### Step 3: inspect final benchmark

```text
build/.../bench/summary.txt
build/.../bench/metrics.json
```

### Step 4: inspect intermediate benchmark

```text
build/.../bench/intermediate/intermediate_summary.txt
build/.../bench/intermediate/intermediate_metrics.json
```

### Step 5: fix the first bad layer

Do not start from the final output only. Use the first bad intermediate layer to identify where divergence begins.

---

## Policy Sweep and Plotting Workflow

FPGAI can also be used as a compiler experimentation platform for paper-ready evaluation.

This is especially useful for:

- precision studies,
- latency / resource tradeoff analysis,
- policy-driven compiler optimization,
- ASAP 2026 paper figures and tables.

### Typical experiment dimensions

The sweep scripts can explore:

- model choice,
- precision policy,
- parallelization policy.

Typical precision policies:

- `Uniform-8`
- `Uniform-12`
- `Uniform-16`
- `Mixed-Conservative`
- `Mixed-Aggressive`

Typical parallelization policies:

- `Resource-First`
- `Balanced`
- `Latency-First`

### Run policy sweep

If you add the policy sweep script:

```text
scripts/run_policy_sweep.py
```

run experiments like this:

```bash
python3 scripts/run_policy_sweep.py \
  --config fpgai.yml \
  --models models/suite/mlp_mnist.onnx models/cnn_mnist.onnx \
  --precision-policies Uniform-8 Uniform-12 Uniform-16 Mixed-Conservative Mixed-Aggressive \
  --parallel-policies Resource-First Balanced Latency-First
```

This script will:

- generate temporary configs,
- run the compiler for each experiment point,
- collect benchmark and HLS data,
- write an aggregated CSV.

Typical output:

```text
build/policy_sweeps/policy_sweep_results.csv
```

### Generate plots

If you add:

```text
scripts/plot_policy_results.py
```

run:

```bash
python3 scripts/plot_policy_results.py \
  --csv build/policy_sweeps/policy_sweep_results.csv
```

Typical output directory:

```text
build/policy_sweeps/plots/
```

Typical plots include:

- RMSE vs precision policy,
- latency vs parallel policy,
- LUT vs precision policy,
- DSP vs precision policy,
- BRAM vs precision policy,
- Pareto plot: latency vs RMSE,
- heatmaps for latency / RMSE / DSP.

### Summarize best policies

If you add:

```text
scripts/summarize_policy_results.py
```

run:

```bash
python3 scripts/summarize_policy_results.py \
  --csv build/policy_sweeps/policy_sweep_results.csv
```

This is useful for selecting:

- the best-accuracy policy,
- the lowest-latency policy,
- a balanced tradeoff point.

---

## Example Debugging Interpretation

Suppose you get:

```text
conv0   passed
act0    passed
pool0   passed
dense0  failed
act1    slightly off
```

This usually means:

- Conv, activation, and pooling are correct,
- the issue starts at flatten/reshape or dense input ordering,
- softmax drift is a consequence of earlier dense mismatch.

This is much more useful than only seeing the final output mismatch.

---

## How to Contribute

Contributions are welcome.

Typical contribution areas:

- new layer types,
- improved planners,
- quantization strategies,
- memory and communication optimization,
- benchmark enhancements,
- plotting and experiment automation,
- documentation.

### General contribution flow

1. fork the repository,
2. create a feature branch,
3. implement your change,
4. add tests,
5. run benchmark,
6. open a pull request.

---

## Contributor Guide for Adding a New Layer

Adding a new operator usually touches these areas:

### 1. Frontend / import

Make sure the ONNX importer maps the ONNX op into FPGAI IR.

### 2. Analysis

Update graph analysis so the new op gets:

- input/output shape info,
- parameter bytes,
- backend kernel hint,
- compute and memory characterization.

### 3. Planner

Update compile planning if the operator needs custom tiling, unrolling, or memory behavior.

### 4. HLS emitters

Add the HLS layer implementation under the HLS backend.

Typical additions include:

- operator emitter,
- header emission,
- C++ emission,
- integration hooks.

### 5. Top integration

Update top generation so the new layer is instantiated and connected correctly.

### 6. Benchmark compatibility

Make sure:

- final output still works,
- intermediate dumps can identify the layer,
- comparison layout is handled correctly if needed.

---

## Testing Strategy

FPGAI uses two levels of testing.

### 1. Compiler correctness tests

These validate full-model compilation against ONNX reference.

Goal:

- prove generated code is correct end to end.

### 2. Layer contributor tests

These validate individual operators in isolation.

Goal:

- help contributors test a new layer before full integration.

---

## Recommended Testing Workflow for Contributors

### A. Basic compile test

```bash
python3 main.py --config fpgai.yml
```

Check:

- compile result,
- HLS logs,
- generated artifacts.

### B. Final correctness benchmark

Enable benchmark in YAML and inspect:

```text
build/.../bench/summary.txt
build/.../bench/metrics.json
```

### C. Intermediate debugging

Inspect:

```text
build/.../bench/intermediate/intermediate_summary.txt
build/.../bench/intermediate/intermediate_metrics.json
```

This shows where errors first appear.

### D. New-layer debugging

For a newly added operator:

- first check compile path,
- then final output,
- then intermediate mismatch position,
- then operator-specific layout and precision assumptions.

---

## HLS Debugging Guide

### HLS run failed

Check:

```text
build/.../hls/logs/vitis_hls_stdout.log
build/.../hls/logs/vitis_hls_stderr.log
```

Typical causes:

- unsupported C++ in synthesizable region,
- missing include,
- unsupported standard library usage in csynth,
- type mismatch between generated headers and source,
- mismatch between top signature and testbench signature.

### CSIM passed but CSYNTH failed

This often means:

- code is valid C++ but not synthesizable,
- debug-only code leaked into synthesis,
- unsupported file I/O or system calls are present.

Use guards like:

```cpp
#if defined(FPGAI_DEBUG_DUMP) && !defined(__SYNTHESIS__)
...
#endif
```

---

## Final Benchmark Debugging Guide

### Benchmark failed but final errors are small

Possible reasons:

- thresholds are too strict,
- softmax is sensitive to small numeric changes,
- argmax flips because top classes are very close.

In that case:

- inspect dense output and pre-softmax behavior,
- do not rely only on final softmax probabilities.

### Benchmark failed badly in stream mode

Possible reasons:

- runtime weights are dummy or ordered incorrectly,
- streamed weight packing does not match ONNX parameter layout.

Start with `embedded` mode first.

---

## Intermediate Benchmark Debugging Guide

### First bad layer is a spatial op like `conv0`

This may indicate a layout mismatch in the intermediate comparator.

Check whether ONNX is NCHW and HLS internal storage is HWC-flat.

### First bad layer is `dense0`

This often means:

- flatten or reshape ordering mismatch,
- Dense is consuming feature maps in the wrong flattened order.

This is a very common CNN-to-Dense bug source.

### Reshape or Flatten skipped

This is acceptable in the current version if layout semantics differ between ONNX and internal HLS representation.

---

## Common Development Advice

- Start with `weights.mode: embedded` when validating correctness.
- Enable intermediate benchmark when debugging a new operator.
- Do not judge correctness only from final softmax if logits or dense outputs already match well.
- Be careful with layout conversions between ONNX tensors and internal HLS buffer order.
- Benchmark thresholds should reflect numeric precision mode.
- Keep debug dump code out of synthesis using `!defined(__SYNTHESIS__)`.
- Fix the first bad layer before trying to tune final output thresholds.

---

## Example Commands

### Compile only

```bash
python3 main.py --config fpgai.yml
```

### View final benchmark summary

```bash
cat build/fpgai_example_dense/bench/summary.txt
cat build/fpgai_example_dense/bench/metrics.json
```

### View intermediate benchmark summary

```bash
cat build/fpgai_example_dense/bench/intermediate/intermediate_summary.txt
```

### Print first bad layer and worst layers

```bash
python3 - <<'PY'
import json
p="build/fpgai_example_dense/bench/intermediate/intermediate_metrics.json"
data=json.load(open(p))
print("first_bad_layer =", data["first_bad_layer"])
for x in data["worst_layers"][:10]:
    print(x["layer_name"], x["op_type"], x["max_abs_error"], x["mean_abs_error"], x["cosine_similarity"], x["passed"])
print("skipped =", data["skipped_layers"])
PY
```

### Inspect HLS logs

```bash
sed -n '1,220p' build/fpgai_example_dense/hls/logs/vitis_hls_stdout.log
sed -n '1,120p' build/fpgai_example_dense/hls/logs/vitis_hls_stderr.log
```

---

## Current Status

FPGAI currently supports an automated flow for:

- ONNX import,
- HLS generation,
- Vitis HLS execution,
- final correctness benchmarking,
- intermediate layer-wise debugging,
- experiment scripting for policy exploration.

The system is actively evolving, especially around:

- layout handling,
- streamed weights,
- training flow,
- richer operator coverage,
- contributor testing infrastructure,
- deeper policy-driven planner support.

---

## Roadmap

Planned or ongoing directions include:

- better streamed-weight benchmarking,
- full layer contributor test suite,
- training benchmark support,
- richer custom operator plugin flow,
- pre-softmax/logit benchmark mode,
- broader board support,
- automatic real-weight export for runtime preload mode,
- improved reshape and flatten semantic handling,
- per-layer contributor reference harnesses,
- stronger policy-driven parallelization support,
- automated paper table generation.

---

## Contact

Repository:

- [https://github.com/umutcanaltin/fpgai_compiler](https://github.com/umutcanaltin/fpgai_compiler)

For issues and contributions, use GitHub Issues and Pull Requests.
