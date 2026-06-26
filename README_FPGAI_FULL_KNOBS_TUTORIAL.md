# FPGAI

FPGAI is a resource-aware compilation and execution framework for neural-network inference and training on FPGA-SoCs.

The main idea is simple:

```text
Model + YAML configuration
→ FPGAI compiler plan
→ generated HLS C++ / runtime artifacts
→ resource and timing predictions
→ Vitis HLS reports
→ Vivado implementation reports / bitstream
→ real FPGA inference or training validation
```

FPGAI is designed around one rule:

```text
Do not hide missing features.
```

If a YAML setting is accepted, it should either:

1. change the compiler plan,
2. change generated HLS/runtime artifacts,
3. change prediction reports,
4. change HLS/Vivado results,
5. affect real board execution,

or FPGAI should clearly report that the feature is not fully implemented yet.

A user should not need to guess whether a setting worked.

---

# Table of contents

- [What is an FPGA?](#what-is-an-fpga)
- [What is FPGA-SoC?](#what-is-fpga-soc)
- [What is HLS?](#what-is-hls)
- [What FPGAI does](#what-fpgai-does)
- [The FPGAI truth path](#the-fpgai-truth-path)
- [Quick start](#quick-start)
- [Minimal YAML configuration](#minimal-yaml-configuration)
- [Beginner FPGA concepts](#beginner-fpga-concepts)
- [YAML knob precedence](#yaml-knob-precedence)
- [Target and board knobs](#target-and-board-knobs)
- [Precision knobs](#precision-knobs)
- [Parallelism knobs](#parallelism-knobs)
- [Pipeline knobs](#pipeline-knobs)
- [Tiling knobs](#tiling-knobs)
- [Memory knobs](#memory-knobs)
- [Training knobs](#training-knobs)
- [HLS knobs](#hls-knobs)
- [Vivado knobs](#vivado-knobs)
- [Reports and artifacts](#reports-and-artifacts)
- [How to verify that a knob really worked](#how-to-verify-that-a-knob-really-worked)
- [Beginner recipes](#beginner-recipes)
- [Development audits](#development-audits)
- [Current truth boundary](#current-truth-boundary)
- [Glossary](#glossary)

---

# What is an FPGA?

An FPGA is a chip whose hardware can be reconfigured after manufacturing.

A CPU runs instructions on fixed hardware. A GPU runs many threads on fixed parallel hardware. An FPGA lets you build a custom hardware circuit for your workload.

For neural networks, this means FPGAI can generate a circuit for layers such as:

- Dense / fully connected layers,
- convolution layers,
- activation functions,
- pooling layers,
- training/backpropagation components where supported.

The advantage is that the hardware can be specialized. The cost is that every design choice matters.

A small change such as increasing parallelism from `2` to `8` can create much more hardware.

---

# What is FPGA-SoC?

An FPGA-SoC combines:

- **PS**, the processing system, usually ARM CPU cores,
- **PL**, the programmable logic, the FPGA fabric.

Example boards:

- PYNQ-Z2,
- KV260,
- KR260.

In an FPGA-SoC flow, the CPU can run Linux and software control code, while the FPGA fabric runs the accelerated neural-network kernel.

A typical data path is:

```text
CPU / Linux application
→ DMA transfer
→ FPGA accelerator in PL
→ DMA transfer back
→ CPU reads result
```

---

# What is HLS?

HLS means **High-Level Synthesis**.

Instead of writing hardware directly in Verilog or VHDL, HLS lets you write C++ code with special pragmas. The HLS tool converts that C++ into hardware.

Example HLS pragma:

```cpp
#pragma HLS PIPELINE II=1
```

This tells Vitis HLS to pipeline a loop or function.

Another example:

```cpp
#pragma HLS UNROLL factor=4
```

This tells Vitis HLS to copy loop hardware so several iterations can run in parallel.

FPGAI generates HLS C++ and pragmas from your YAML configuration.

---

# What FPGAI does

FPGAI takes:

- a neural-network model,
- a YAML configuration,
- board information,
- precision choices,
- hardware optimization choices,
- memory choices,
- inference or training mode,

and generates:

- compiler plans,
- HLS C++ source,
- HLS scripts,
- prediction reports,
- board-fit reports,
- hardware knob contracts,
- optional Vitis/Vivado artifacts,
- optional runtime validation artifacts.

The goal is not only to generate code. The goal is to make the design decisions traceable.

---

# The FPGAI truth path

Every user-facing hardware knob should follow this path:

```text
YAML setting
→ compiler plan
→ generated HLS/runtime artifact
→ prediction report
→ Vitis HLS report
→ Vivado report / bitstream
→ real FPGA inference/training validation
```

For example:

```yaml
memory:
  weight_storage: uram
```

should be visible in:

```text
reports/hardware_knob_contract.json
reports/resource_prediction.json
hls/src/deeplearn.cpp
hls/src/fpgai_params.cpp
Vitis HLS utilization report
Vivado utilization report
```

If it only appears in YAML but not in generated HLS or reports, then the knob is not fully wired.

---

# Quick start

A typical workflow is:

```bash
python -m fpgai.cli compile --config configs/examples/inference_compile.yml
```

or, depending on your local CLI entry point:

```bash
fpgai compile --config configs/examples/inference_compile.yml
```

Then inspect generated artifacts under the configured output directory.

Important files to check:

```text
reports/hardware_knob_contract.json
reports/hardware_knob_contract.md
reports/prediction_summary.md
reports/resource_prediction.json
reports/timing_prediction.json
reports/board_fit.json
reports/board_fit.md
hls/src/deeplearn.cpp
hls/include/fpgai_types.h
hls/include/fpgai_params.h
hls/src/fpgai_params.cpp
hls/run_hls.tcl
```

---

# Minimal YAML configuration

This is a small inference-style example.

```yaml
project:
  mode: inference
  outputs:
    top_kernel_name: deeplearn

model:
  format: onnx
  path: models/mlp.onnx

target:
  board: kv260
  clock_mhz: 100
  fit_policy: warn

numerics:
  defaults:
    activation: {type: ap_fixed, total_bits: 16, int_bits: 6}
    weight:     {type: ap_fixed, total_bits: 16, int_bits: 6}
    bias:       {type: ap_fixed, total_bits: 24, int_bits: 10}
    accum:      {type: ap_fixed, total_bits: 24, int_bits: 10}

optimization:
  parallel_policy: Balanced
  parallel:
    pe: 2
    simd: 2
    unroll_factor: 2
    partition_factor: 2
    array_partition_mode: cyclic
  pipeline:
    style: balanced
    ii: 2
  tiling:
    dense:
      tm: 8
      tn: 8
      tk: 8
    conv:
      tm: 4
      tn: 4
      tk: 4

memory:
  weight_storage: bram
  allow_double_buffer: false
```

---

# Beginner FPGA concepts

## Resources

An FPGA has limited hardware resources.

Common resource types:

| Resource | Meaning | Used for |
|---|---|---|
| LUT | Look-up table | Logic, small computations, control |
| FF | Flip-flop | Registers, pipeline stages |
| BRAM | Block RAM | On-chip memory |
| URAM | UltraRAM | Larger on-chip memory on some Xilinx devices |
| DSP | Digital signal processing block | Fast multiply/accumulate operations |

When FPGAI increases parallelism, it usually uses more LUTs, FFs, BRAM/URAM, and DSPs.

---

## Latency

Latency means how long one input takes to produce one output.

Example:

```text
input image enters accelerator
→ computation happens
→ output class comes back
```

The time between input and output is latency.

Lower latency is better for real-time applications.

---

## Throughput

Throughput means how many inputs can be processed per second.

A design can have high latency but good throughput if it is deeply pipelined.

Example:

```text
The first image takes 1000 cycles,
but after the pipeline is full,
one new image finishes every 10 cycles.
```

---

## Clock frequency

Clock frequency controls how fast the FPGA circuit runs.

Example:

```yaml
target:
  clock_mhz: 100
```

A 100 MHz clock has a 10 ns period.

Higher clock frequency can improve speed, but makes timing closure harder.

---

## Timing closure

Timing closure means the FPGA design can physically run at the requested clock speed.

A design may pass HLS but fail Vivado timing because the routed hardware path is too long.

This is why final claims need Vivado timing reports, not only estimates.

---

## Parallelism

Parallelism means building more hardware so multiple operations happen at the same time.

Example:

A dense layer computes:

```text
y = W x + b
```

For each output neuron, it performs many multiply-add operations.

With low parallelism, one small hardware unit does many operations sequentially.

With high parallelism, many multipliers/adders work at the same time.

Trade-off:

```text
more parallelism → faster but larger design
less parallelism → slower but smaller design
```

---

## Pipelining

Pipelining splits computation into stages.

A useful analogy is a factory line.

Without pipelining:

```text
item 1 fully finishes
then item 2 starts
then item 3 starts
```

With pipelining:

```text
item 1 is in stage 3
item 2 is in stage 2
item 3 is in stage 1
```

This improves throughput.

In HLS, this is often controlled by:

```cpp
#pragma HLS PIPELINE II=1
```

---

## Initiation interval

Initiation interval, or `II`, means how many cycles pass before a pipelined loop can start the next iteration.

```text
II=1 → start a new iteration every cycle
II=2 → start a new iteration every 2 cycles
II=4 → start a new iteration every 4 cycles
```

Lower II is usually faster, but harder to achieve and may use more resources.

---

## Loop unrolling

A loop normally reuses the same hardware repeatedly.

Example:

```cpp
for (int i = 0; i < 8; i++) {
    y += a[i] * b[i];
}
```

If unroll factor is 1, one multiplier may be reused.

If unroll factor is 4, HLS may build 4 multipliers and process 4 iterations at once.

Trade-off:

```text
higher unroll factor → faster but more hardware
lower unroll factor  → slower but smaller
```

---

## Array partitioning

FPGA memories have limited read and write ports.

If many parallel compute units need the same array, one memory may not provide enough values per cycle.

Array partitioning splits an array into multiple banks.

Example:

```cpp
#pragma HLS ARRAY_PARTITION variable=W cyclic factor=4
```

This can allow more parallel reads.

Trade-off:

```text
more partitioning → more memory bandwidth but more resources
less partitioning → smaller but may bottleneck parallel compute
```

---

## Tiling

Tiling means splitting a large computation into smaller blocks.

This is one of the most important FPGA concepts.

Imagine a matrix multiplication:

```text
Y = W × X
```

If `W` and `X` are large, the FPGA may not have enough on-chip memory to hold everything at once.

Instead, FPGAI can compute smaller tiles:

```text
load tile 1
compute tile 1
store/update result
load tile 2
compute tile 2
store/update result
...
```

Why tiling helps:

- It reduces on-chip memory pressure.
- It improves data reuse.
- It makes large layers fit on smaller boards.
- It exposes loops that can be pipelined and unrolled.

Trade-off:

```text
larger tiles → more reuse and possibly faster, but more BRAM/URAM
smaller tiles → easier to fit, but more loop/control overhead
```

In FPGAI, dense tiling uses knobs such as:

```yaml
optimization:
  tiling:
    dense:
      tm: 32
      tn: 32
      tk: 32
```

Convolution tiling uses similar knobs:

```yaml
optimization:
  tiling:
    conv:
      tm: 16
      tn: 16
      tk: 16
```

The exact mapping depends on the layer type and backend, but the idea is always the same: split large work into hardware-friendly chunks.

---

## Precision

Precision controls how numbers are represented.

Software often uses 32-bit floating point:

```text
float32
```

FPGA designs often use fixed-point numbers:

```text
ap_fixed<16,6>
```

This means:

```text
total bits = 16
integer bits = 6
fractional bits = 10
```

Smaller precision:

- uses less memory,
- can use fewer DSPs/LUTs,
- can be faster,
- may reduce accuracy.

Larger precision:

- improves numerical range/stability,
- uses more resources,
- can be slower.

---

## BRAM vs URAM vs DDR

FPGA memory hierarchy matters.

| Memory | Meaning | Typical use |
|---|---|---|
| BRAM | On-chip block RAM | small/medium buffers, weights, activations |
| URAM | Larger on-chip UltraRAM | larger weights or buffers on supported boards |
| DDR | External memory | large data, slower access, DMA transfers |
| LUTRAM | Memory made from LUTs | very small arrays |

Choosing memory affects performance and resource usage.

---

# YAML knob precedence

When multiple settings could affect the same design choice, FPGAI should use this precedence:

```text
manual_yaml_override
> board_aware_policy_scaling
> policy_preset
> compiler_default
```

Meaning:

1. explicit YAML values should win,
2. board-aware scaling may clamp unsafe values,
3. policy presets fill missing values,
4. compiler defaults are used only when nothing else is provided.

Example:

```yaml
optimization:
  parallel_policy: Balanced
  parallel:
    pe: 8
```

Even if `Balanced` would normally choose a smaller `pe`, the manual value `pe: 8` should be respected unless the board-fit layer rejects or clamps it.

---

# Target and board knobs

## `target.board`

Selects the FPGA board.

Example:

```yaml
target:
  board: kv260
```

Common examples:

```yaml
target:
  board: pynq_z2
```

```yaml
target:
  board: kv260
```

```yaml
target:
  board: kr260
```

What it means:

The board determines the FPGA part and available resources.

What it affects:

- target FPGA part,
- board resource budget,
- board-fit report,
- default clock,
- board-aware policy scaling,
- whether aggressive settings are realistic.

Beginner advice:

Start with safer settings on small boards. Use more aggressive settings on larger boards.

Expected artifacts:

```text
reports/board_fit.json
reports/board_fit.md
reports/hardware_knob_contract.json
reports/hardware_knob_contract.md
```

---

## `target.clock_mhz`

Sets the requested hardware clock frequency.

Example:

```yaml
target:
  clock_mhz: 100
```

What it means:

The accelerator should be synthesized for a 100 MHz clock target.

What it affects:

- HLS clock constraint,
- timing prediction,
- Vivado timing target,
- whether timing closure is easy or hard.

Hardware effect:

Higher clock can make the accelerator faster, but also makes routing/timing harder.

Expected artifacts:

```text
hls/run_hls.tcl
reports/timing_prediction.json
reports/prediction_summary.md
```

---

## `target.fit_policy`

Controls what FPGAI should do if the design may not fit on the selected board.

Examples:

```yaml
target:
  fit_policy: report_only
```

```yaml
target:
  fit_policy: warn
```

```yaml
target:
  fit_policy: enforce
```

Options:

| Value | Meaning |
|---|---|
| `report_only` | generate fit report but do not stop |
| `warn` | warn when near/over board limits |
| `enforce` | block deployable bitstream flow when over limit |

What it affects:

- board-fit status,
- whether over-limit designs can continue,
- whether bitstream/deployable overlay is allowed.

Expected artifacts:

```text
reports/board_fit.json
reports/board_fit.md
```

---

# Precision knobs

## `numerics.defaults`

Sets the default number format for the model.

Example:

```yaml
numerics:
  defaults:
    activation: {type: ap_fixed, total_bits: 16, int_bits: 6}
    weight:     {type: ap_fixed, total_bits: 16, int_bits: 6}
    bias:       {type: ap_fixed, total_bits: 24, int_bits: 10}
    accum:      {type: ap_fixed, total_bits: 24, int_bits: 10}
```

Terms:

| Field | Meaning |
|---|---|
| `activation` | values flowing between layers |
| `weight` | model weights |
| `bias` | bias values |
| `accum` | accumulation type used during multiply-add operations |

Hardware effect:

Smaller bit widths usually reduce resource use and memory bandwidth. Larger bit widths usually improve numerical stability but use more resources.

Expected HLS effect:

```cpp
typedef ap_fixed<16,6> act_t;
typedef ap_fixed<16,6> wgt_t;
typedef ap_fixed<24,10> bias_t;
typedef ap_fixed<24,10> acc_t;
```

Expected artifacts:

```text
hls/include/fpgai_types.h
hls/include/fpgai_params.h
hls/src/fpgai_params.cpp
reports/resource_prediction.json
reports/prediction_summary.md
```

---

## `numerics.layers`

Overrides precision for specific layers.

Example:

```yaml
numerics:
  layers:
    - match:
        name: dense0
      activation: {type: ap_fixed, total_bits: 12, int_bits: 4}
      weight:     {type: ap_fixed, total_bits: 9, int_bits: 3}
      bias:       {type: ap_fixed, total_bits: 15, int_bits: 6}
      accum:      {type: ap_fixed, total_bits: 20, int_bits: 8}
```

What it means:

Only `dense0` uses this precision. Other layers can use defaults or their own overrides.

Hardware effect:

Layerwise precision lets you spend more bits only where needed. This can reduce hardware cost while preserving accuracy.

Expected HLS effect:

```cpp
typedef ap_fixed<12,4> op0_act_t;
typedef ap_fixed<9,3> op0_wgt_t;
typedef ap_fixed<15,6> op0_bias_t;
typedef ap_fixed<20,8> op0_acc_t;
```

Expected artifacts:

```text
hls/include/fpgai_types.h
hls/include/fpgai_params.h
hls/src/deeplearn.cpp
```

---

## `analysis.precision_sweep`

Defines named precision candidates for experiments.

Example:

```yaml
analysis:
  precision_sweep:
    candidates:
      - name: fx8_3
        defaults:
          activation: {type: ap_fixed, total_bits: 8, int_bits: 3}
          weight:     {type: ap_fixed, total_bits: 8, int_bits: 3}
          bias:       {type: ap_fixed, total_bits: 16, int_bits: 6}
          accum:      {type: ap_fixed, total_bits: 16, int_bits: 6}
      - name: fx16_6
        defaults:
          activation: {type: ap_fixed, total_bits: 16, int_bits: 6}
          weight:     {type: ap_fixed, total_bits: 16, int_bits: 6}
          bias:       {type: ap_fixed, total_bits: 24, int_bits: 10}
          accum:      {type: ap_fixed, total_bits: 24, int_bits: 10}
```

A sweep can then select:

```yaml
precision_mode: fx8_3
```

What it affects:

- materializes selected candidate into `numerics.defaults`,
- changes HLS typedefs,
- changes resource estimates,
- should be compared against accuracy results.

Expected artifacts:

```text
reports/hardware_knob_contract.json
hls/include/fpgai_types.h
reports/resource_prediction.json
reports/prediction_summary.md
```

---

# Parallelism knobs

Parallelism controls how much hardware FPGAI builds to compute at the same time.

More parallelism usually improves speed but increases resource usage.

---

## `optimization.parallel_policy`

High-level preset.

Example:

```yaml
optimization:
  parallel_policy: Balanced
```

Common values:

| Policy | Meaning |
|---|---|
| `Resource-First` | prefer smaller design |
| `Balanced` | balance performance and resources |
| `Latency-First` | prefer lower latency |
| `Throughput-First` | prefer high streaming throughput |

Hardware effect:

A policy can select defaults for parallelism, pipelining, tiling, and memory behavior.

Important:

Manual YAML values should override policy defaults.

Expected artifacts:

```text
reports/hardware_knob_contract.json
reports/resource_prediction.json
reports/timing_prediction.json
```

---

## `optimization.parallel.pe`

Processing element count.

Example:

```yaml
optimization:
  parallel:
    pe: 8
```

What it means:

`pe` controls output-side parallel workers.

For a dense layer, it can mean computing multiple output neurons in parallel.

For a convolution layer, it can mean computing multiple output channels in parallel.

Hardware effect:

Higher `pe` can reduce latency but increases resource usage.

Expected artifacts:

```text
hls/include/fpgai_types.h
hls/src/deeplearn.cpp
reports/hardware_knob_contract.json
reports/resource_prediction.json
```

---

## `optimization.parallel.simd`

Input-side parallelism.

Example:

```yaml
optimization:
  parallel:
    simd: 8
```

What it means:

`simd` controls how many input multiplications can happen in parallel.

Hardware effect:

Higher `simd` usually increases DSP usage and memory bandwidth needs.

Expected artifacts:

```text
hls/include/fpgai_types.h
hls/src/deeplearn.cpp
reports/resource_prediction.json
```

---

## `optimization.parallel.unroll_factor`

General loop unroll factor.

Example:

```yaml
optimization:
  parallel:
    unroll_factor: 4
```

What it means:

Loop unrolling duplicates loop hardware.

Hardware effect:

Higher unroll factor can reduce cycle count but increases LUT/FF/DSP use.

Expected HLS effect:

```cpp
#pragma HLS UNROLL factor=4
```

Expected artifacts:

```text
hls/src/deeplearn.cpp
reports/hardware_knob_contract.json
```

---

## `optimization.parallel.partition_factor`

Array partitioning factor.

Example:

```yaml
optimization:
  parallel:
    partition_factor: 4
```

What it means:

Splits arrays into multiple memory banks to allow more parallel reads/writes.

Hardware effect:

Higher partition factor can remove memory bottlenecks but increases resource usage.

Expected HLS effect:

```cpp
#pragma HLS ARRAY_PARTITION variable=W cyclic factor=4
```

Expected artifacts:

```text
hls/src/deeplearn.cpp
reports/hardware_knob_contract.json
```

---

## `optimization.parallel.array_partition_mode`

Controls how arrays are split.

Example:

```yaml
optimization:
  parallel:
    array_partition_mode: cyclic
```

Options:

| Mode | Meaning |
|---|---|
| `cyclic` | round-robin distribution across banks |
| `block` | contiguous blocks |
| `complete` | each element becomes separate storage |

Hardware effect:

`complete` gives maximum parallel access but can be very expensive.

Expected artifacts:

```text
hls/src/deeplearn.cpp
reports/hardware_knob_contract.json
```

---

# Pipeline knobs

## `optimization.pipeline.style`

High-level pipeline style.

Example:

```yaml
optimization:
  pipeline:
    style: aggressive
```

Common values:

| Style | Meaning |
|---|---|
| `conservative` | easier to fit, less aggressive |
| `balanced` | middle ground |
| `aggressive` | lower II / higher throughput target |

Hardware effect:

Aggressive pipelining may improve throughput but can increase resource usage and timing pressure.

Expected artifacts:

```text
hls/src/deeplearn.cpp
hls/include/fpgai_types.h
reports/hardware_knob_contract.json
reports/timing_prediction.json
```

---

## `optimization.pipeline.ii`

Initiation interval target.

Example:

```yaml
optimization:
  pipeline:
    ii: 1
```

What it means:

`II=1` means the pipeline should start a new loop iteration every cycle.

Hardware effect:

Lower II improves throughput but is harder to satisfy.

Expected HLS effect:

```cpp
#pragma HLS PIPELINE II=1
```

Expected artifacts:

```text
hls/src/deeplearn.cpp
reports/hardware_knob_contract.json
reports/timing_prediction.json
```

---

# Tiling knobs

Tiling splits large layers into smaller pieces.

This helps fit large computations into limited FPGA memory.

---

## `optimization.tiling.dense`

Dense layer tiling.

Example:

```yaml
optimization:
  tiling:
    dense:
      tm: 32
      tn: 32
      tk: 32
```

Beginner explanation:

A dense layer performs a matrix-vector or matrix-matrix style computation.

Instead of processing the whole matrix at once, FPGAI can process smaller blocks.

Terms:

| Field | Beginner meaning |
|---|---|
| `tm` | output tile size |
| `tn` | input/output dimension tile size depending on backend mapping |
| `tk` | reduction/input tile size |

Hardware effect:

- Larger tiles can improve reuse.
- Larger tiles use more on-chip memory.
- Smaller tiles are easier to fit but may add overhead.

Expected artifacts:

```text
hls/src/deeplearn.cpp
reports/hardware_knob_contract.json
reports/resource_prediction.json
```

---

## `optimization.tiling.conv`

Convolution tiling.

Example:

```yaml
optimization:
  tiling:
    conv:
      tm: 16
      tn: 16
      tk: 16
```

Beginner explanation:

A convolution layer has loops over:

- output channels,
- input channels,
- image height,
- image width,
- kernel height,
- kernel width.

Tiling splits those loops into smaller chunks.

Hardware effect:

- improves locality,
- controls buffer sizes,
- can reduce memory bandwidth pressure,
- affects BRAM/URAM usage.

Expected artifacts:

```text
hls/src/deeplearn.cpp
reports/hardware_knob_contract.json
reports/resource_prediction.json
```

---

# Memory knobs

## `memory.weight_storage`

Controls where weights should be stored.

Examples:

```yaml
memory:
  weight_storage: bram
```

```yaml
memory:
  weight_storage: uram
```

Options:

| Value | Meaning |
|---|---|
| `bram` | use block RAM |
| `uram` | use UltraRAM where available |
| `lutram` | use LUT-based memory where supported |
| `ddr` | external memory / streamed mode where supported |

Hardware effect:

- BRAM is good for small and medium buffers.
- URAM is useful for larger weights on supported boards.
- DDR supports larger data but has higher latency and needs DMA/data movement.

Expected HLS effect:

```cpp
#pragma HLS BIND_STORAGE variable=W0 type=ram_1p impl=bram
```

or:

```cpp
#pragma HLS BIND_STORAGE variable=W0 type=ram_1p impl=uram
```

Expected artifacts:

```text
hls/src/deeplearn.cpp
hls/src/fpgai_params.cpp
reports/hardware_knob_contract.json
reports/resource_prediction.json
```

---

## `memory.allow_double_buffer`

Enables double buffering where supported.

Example:

```yaml
memory:
  allow_double_buffer: true
```

What it means:

Double buffering uses two buffers:

```text
buffer A is being processed
buffer B is being loaded
then they swap
```

Hardware effect:

Can improve throughput by overlapping compute and data movement, but uses more memory.

Expected artifacts:

```text
reports/hardware_knob_contract.json
reports/resource_prediction.json
hls/src/deeplearn.cpp
```

---

## `memory.storage`

Lower-level memory placement settings.

Example:

```yaml
memory:
  storage:
    weights: bram
    activations: bram
    gradients: bram
    optimizer_state: bram
```

What it affects:

- weight placement,
- activation placement,
- gradient placement,
- optimizer-state placement.

Recommendation:

Prefer the explicit user-facing knob:

```yaml
memory:
  weight_storage: bram
```

If both `memory.weight_storage` and `memory.storage.weights` exist, keep them consistent.

---

# Training knobs

Training on FPGA is more complex than inference because it may need:

- forward pass,
- loss computation,
- backward pass,
- gradients,
- optimizer update,
- updated weights.

---

## `project.mode`

Selects inference or training flow.

Inference:

```yaml
project:
  mode: inference
```

Training:

```yaml
project:
  mode: training_on_device
```

What it affects:

- generated HLS top,
- generated testbench,
- whether training logic is emitted,
- reports and validation mode.

Expected artifacts:

```text
hls/src/deeplearn.cpp
hls/src/tb.cpp
reports/prediction_summary.md
```

---

## `training.optimizer`

Configures optimizer.

Example:

```yaml
training:
  optimizer:
    type: sgd
    learning_rate: 0.005
```

What it means:

The optimizer updates weights during training.

Hardware effect:

Training update logic is added to the HLS design or testbench path depending on the selected mode.

Expected artifacts:

```text
hls/src/deeplearn.cpp
hls/src/tb.cpp
reports/training_*.json
```

---

## `training.loss`

Configures loss function.

Example:

```yaml
training:
  loss:
    type: mse
```

What it means:

The loss measures how wrong the model output is.

Hardware effect:

Loss evaluation may appear in the training testbench or training kernel path depending on mode.

Expected artifacts:

```text
hls/src/tb.cpp
reports/training_*.json
```

---

## `training.execution`

Controls training run settings.

Example:

```yaml
training:
  execution:
    batch_size: 1
    epochs: 1
```

What it affects:

- testbench loop count,
- runtime validation behavior,
- convergence experiments,
- timing measurements.

Expected artifacts:

```text
hls/src/tb.cpp
reports/training_*.json
```

---

## `training.storage`

Controls training-specific memory placement.

Example:

```yaml
training:
  storage:
    weights: bram
    activations: bram
    gradients: bram
    optimizer_state: bram
```

Hardware effect:

Training uses more memory than inference because it needs gradients and optimizer state.

If `memory.weight_storage` is also set, keep training storage consistent.

---

# HLS knobs

HLS knobs control Vitis HLS behavior.

Example:

```yaml
hls:
  run_csim: true
  run_csynth: true
  export_ip: false
```

## `hls.run_csim`

Runs C simulation.

```yaml
hls:
  run_csim: true
```

What it proves:

The generated C++ can simulate functionally.

It does not prove final FPGA timing or resources.

---

## `hls.run_csynth`

Runs HLS synthesis.

```yaml
hls:
  run_csynth: true
```

What it proves:

Vitis HLS can synthesize the generated C++ into RTL and produce estimated resource/timing reports.

Expected artifacts:

```text
hls/fpgai_hls_proj/solution1/syn/report/*csynth*
```

---

## `hls.export_ip`

Exports HLS kernel as an IP block.

```yaml
hls:
  export_ip: true
```

What it affects:

Vivado block design can use the HLS kernel IP.

Expected artifacts:

```text
hls/fpgai_hls_proj/solution1/impl/ip
```

---

# Vivado knobs

Vivado turns generated RTL/IP into a full FPGA design.

Example:

```yaml
vivado:
  run_synth: true
  run_impl: true
  bitstream: true
```

## `vivado.run_synth`

Runs Vivado synthesis.

What it proves:

The design can be synthesized for the target FPGA part.

---

## `vivado.run_impl`

Runs Vivado implementation.

What it proves:

Vivado can place and route the design.

This is closer to real hardware than HLS estimates.

---

## `vivado.bitstream`

Generates a bitstream.

What it proves:

The design can become a loadable FPGA configuration, assuming timing and board integration are valid.

Expected artifacts:

```text
vivado/reports/*
*.bit
```

---

# Reports and artifacts

## Hardware knob contract

```text
reports/hardware_knob_contract.json
reports/hardware_knob_contract.md
```

This should show:

- requested value,
- effective value,
- source,
- status,
- where it was applied.

Use this first when checking whether YAML was understood.

---

## Prediction reports

```text
reports/resource_prediction.json
reports/timing_prediction.json
reports/prediction_summary.md
```

These are pre-HLS estimates.

They are useful for fast iteration, but they are not final hardware proof.

---

## Board-fit reports

```text
reports/board_fit.json
reports/board_fit.md
```

These estimate whether the design fits on the selected board.

Possible statuses may include:

```text
fits
near_limit
over_limit
unknown
vivado_allowed
```

---

## Generated HLS source

Important files:

```text
hls/src/deeplearn.cpp
hls/src/tb.cpp
hls/include/fpgai_types.h
hls/include/fpgai_params.h
hls/src/fpgai_params.cpp
hls/run_hls.tcl
```

Look for real changes such as:

```cpp
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS ARRAY_PARTITION variable=... factor=8
#pragma HLS BIND_STORAGE variable=W0 type=ram_1p impl=uram
typedef ap_fixed<8,3> op0_act_t;
```

---

## Real HLS reports

When Vitis HLS is available:

```text
csynth report
schedule report
utilization report
```

These show what HLS actually synthesized.

---

## Real Vivado reports

When Vivado is available:

```text
utilization report
timing report
power report
bitstream
```

These are needed for final hardware claims.

---

# How to verify that a knob really worked

Do not trust YAML alone.

For every knob, check:

```text
1. hardware knob contract
2. compiler plan
3. generated HLS source
4. prediction reports
5. real HLS report if available
6. real Vivado report if available
7. board runtime validation if claiming real execution
```

Example for `memory.weight_storage: uram`:

```bash
grep -R -n "memory.weight_storage\|impl=uram\|requested for URAM" \
  reports hls/src hls/include
```

Example for pipeline II:

```bash
grep -R -n "PIPELINE\|II=1" hls/src hls/include
```

Example for precision:

```bash
grep -R -n "typedef ap_fixed" hls/include/fpgai_types.h
```

---

# Beginner recipes

## Recipe 1: small safe design

Use this on smaller boards or when starting a new model.

```yaml
target:
  board: pynq_z2
  clock_mhz: 100
  fit_policy: warn

optimization:
  parallel_policy: Balanced
  parallel:
    pe: 2
    simd: 2
    unroll_factor: 2
    partition_factor: 2
    array_partition_mode: cyclic
  pipeline:
    style: balanced
    ii: 2
  tiling:
    dense: {tm: 8, tn: 8, tk: 8}
    conv:  {tm: 4, tn: 4, tk: 4}

memory:
  weight_storage: bram
  allow_double_buffer: false
```

Expected behavior:

- smaller design,
- easier board fit,
- lower performance.

---

## Recipe 2: aggressive KV260 design

Use when you want more performance and have enough resources.

```yaml
target:
  board: kv260
  clock_mhz: 100
  fit_policy: warn

optimization:
  parallel_policy: Latency-First
  parallel:
    pe: 8
    simd: 8
    unroll_factor: 8
    partition_factor: 8
    array_partition_mode: complete
  pipeline:
    style: aggressive
    ii: 1
  tiling:
    dense: {tm: 32, tn: 32, tk: 32}
    conv:  {tm: 16, tn: 16, tk: 16}

memory:
  weight_storage: uram
  allow_double_buffer: true
```

Expected behavior:

- faster design,
- more resource usage,
- more timing pressure,
- may need URAM and careful board-fit checking.

---

## Recipe 3: compare precision modes

Define candidates:

```yaml
analysis:
  precision_sweep:
    candidates:
      - name: fx8_3
        defaults:
          activation: {type: ap_fixed, total_bits: 8, int_bits: 3}
          weight:     {type: ap_fixed, total_bits: 8, int_bits: 3}
          bias:       {type: ap_fixed, total_bits: 16, int_bits: 6}
          accum:      {type: ap_fixed, total_bits: 16, int_bits: 6}
      - name: fx16_6
        defaults:
          activation: {type: ap_fixed, total_bits: 16, int_bits: 6}
          weight:     {type: ap_fixed, total_bits: 16, int_bits: 6}
          bias:       {type: ap_fixed, total_bits: 24, int_bits: 10}
          accum:      {type: ap_fixed, total_bits: 24, int_bits: 10}
```

Then sweep over:

```yaml
precision_mode: fx8_3
```

and:

```yaml
precision_mode: fx16_6
```

Compare:

```text
accuracy
latency
LUT/FF/BRAM/URAM/DSP
energy if available
HLS/Vivado timing
```

---

## Recipe 4: verify a hardware knob contract

Run:

```bash
python -m fpgai.devtools.contract_source_audit
```

This development audit checks that important contract values appear in generated source or trace artifacts.

It should never pass with empty checks.

---

# Development audits

FPGAI includes audit tools used during development.

## Contract/source audit

```bash
python -m fpgai.devtools.contract_source_audit
```

Currently checked examples include:

```text
memory.weight_storage
optimization.pipeline.ii
optimization.parallel.partition_factor
optimization.parallel.unroll_factor
```

When a new user-facing knob is added, extend this audit.

---

## End-to-end audit

```bash
python -m fpgai.devtools.end_to_end_audit
```

This creates representative compile cases such as:

```text
inference_pynq_safe
inference_kv260_aggressive
training_kv260_safe
training_kv260_aggressive
```

and checks whether expected artifacts are present.

---

# Current truth boundary

FPGAI can generate plans, reports, and HLS source without Vitis/Vivado installed.

But final hardware claims require real tool reports.

If `vitis_hls` is not on PATH, then real Vitis HLS reports are not proven on that machine.

If `vivado` is not on PATH, then real Vivado utilization/timing/bitstream are not proven on that machine.

If the board runtime is not executed, then real FPGA inference/training execution is not proven.

Use honest wording:

```text
Generated HLS source and predictions are available.
Real HLS/Vivado/board results require running the hardware toolchain and board validation.
```

Do not claim real hardware numbers from predictions alone.

---

# Glossary

## Activation

The output values flowing between neural-network layers.

## Accumulator

The internal type used while summing multiply-add operations.

## AXI

A common bus protocol used in FPGA-SoC designs.

## AXI DMA

A hardware block that moves data between CPU memory and FPGA logic.

## Bitstream

The file loaded onto the FPGA to configure the hardware.

## BRAM

Block RAM. On-chip memory inside the FPGA.

## Clock

The timing signal that drives hardware. Higher frequency can mean faster design but harder timing closure.

## DSP

Dedicated multiply/add hardware block inside the FPGA.

## FF

Flip-flop. A one-bit register used for storing state and pipeline stages.

## FPGA

Field-programmable gate array. A chip that can be reconfigured into custom hardware.

## HLS

High-Level Synthesis. Converts C++ with pragmas into hardware.

## II

Initiation interval. Number of cycles before a pipelined loop can start the next iteration.

## LUT

Look-up table. Basic logic resource in an FPGA.

## PE

Processing element. A parallel compute worker generated in hardware.

## Pipeline

A hardware structure that overlaps different stages of computation.

## PL

Programmable logic. The FPGA fabric part of an FPGA-SoC.

## PS

Processing system. The CPU part of an FPGA-SoC.

## SIMD

Single instruction, multiple data. In FPGAI, it usually means input-side parallel compute lanes.

## Tiling

Splitting a large computation into smaller chunks to fit memory and improve reuse.

## Timing closure

The design meets the requested clock period after synthesis/place/route.

## Unrolling

Duplicating loop hardware so multiple loop iterations run in parallel.

## URAM

UltraRAM. Larger on-chip memory available on some Xilinx devices.

## Vivado

Xilinx/AMD FPGA implementation tool used for synthesis, place, route, timing, and bitstream generation.

## Vitis HLS

Xilinx/AMD HLS tool that converts C/C++ into RTL hardware.
