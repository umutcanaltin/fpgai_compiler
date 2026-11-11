# Supported features and limitations

This matrix describes the **research compiler's source capability**, not a guarantee of native implementation on any FPGA. Support is **conditional on shapes, attributes, numerical contracts and the selected source path**; unsupported inputs should raise an explicit error.

| Area | Current research support | Boundary |
|---|---|---|
| Functional graph | Static tensor shapes and typed dependencies; strict ONNX import | Arbitrary dynamic shapes, sparse initializers and every ONNX op are not supported |
| Dense / matrix products | 2D MatMul; restricted Gemm (including supported bias and transposed weights) | Not universal batched/broadcast GEMM, arbitrary transpositions or exact all-input fixed-point proof |
| Convolution | Static NCHW Conv2D with supported strides/padding and optional bias | Grouped/dilated convolution and arbitrary auto-padding not supported in the canonical strict path |
| Pooling and activations | Supported combinations of ReLU, MaxPool and other bounded elementwise operators | Feature recognition is not proof of independent native implementation for every composition |
| Neural chains | Bounded convolution → activation/pooling → flatten → dense/matmul/Gemm pipelines; supported multi-layer sequential graphs | General residual DAGs, arbitrary attention/transformers and unconstrained branching are not established |
| Architecture parameters | Precision, parallelism, PE, SIMD, serialization, tiles, pipeline requests | Target/implementation-dependent; not all parameters legal for every operator |
| Model IR | Typed Functional → Architectural → Lowered refinement with provenance | No arbitrary-system temporal/CDC/PS design guarantee |
| Source ownership | Selected static MatMul chains and 2–4-reader fanout source-event/last-use contracts | Not arbitrary graph proof; source address safety is not physical memory-port legality |
| Compiler decisions | Complete typed candidate states, action transitions, fail-closed certification/unknown/rejection | Not a physical feasibility oracle or trained-policy performance guarantee |
| C++ generation | HLS-oriented C++ and compiler manifests | Does not invoke, validate or run native vendor tools automatically |
| Validation | NumPy and selected GCC host-generated-C++ comparisons; optional real Vitis reports | No implied successful synthesis, routing, hardware inference or numerical all-input theorem |
| Toolchains | Explicit opt-in Vitis HLS and Vivado research workflows | Proprietary external tools not bundled; require license, target and independently qualified results |
| RL research | Separate experimental environment under `rl_environment/` | Frozen for this compiler-paper phase; no claimed trained-agent improvements |

## Automated local execution

The `fpgai-run` onboarding command orchestrates existing source generation, attested C simulation, HLS synthesis, and optional Vivado out-of-context routing for a single supported static ONNX model. The feature does not expand the operator matrix or create board bitstreams. Vendor binaries execute only under `--execute-native`. See [Neuroscientist quickstart](NEUROSCIENTIST_QUICKSTART.md).

## Runnable multi-layer examples

Three static ONNX networks are provided in the
[neural-network gallery](../examples/NETWORK_GALLERY.md):

- Conv2D → ReLU → MaxPool2D → Conv2D → Flatten → MatMul → ReLU → Gemm.
- Strided Conv2D → ReLU → Flatten → Gemm → ReLU → MatMul.
- Conv2D → Conv2D → ReLU → Flatten → Gemm → Softmax.

These are **source compilation and optional GCC-host** demonstrations
using synthetic weights. They do not independently establish
timing/area closure or all-graph contract soundness.

## Model/operator contributions

The resolver contains further named operator cases including identity, sigmoid, reshape/squeeze/unsqueeze, softmax, transpose, global average pooling, batch normalization, concatenate, multiply and leaky-ReLU. **Names recognized by a parser/resolver do not imply each operator has an independently validated end-to-end native HLS implementation.** Check the corresponding code emission path and regression evidence before claiming support for a particular topology.

Unsupported operator attributes, incompatible shape relationships, illegal numeric formats and architecture requests outside the compiler's supported implementation contracts must fail before being treated as qualified source.

## Hardware deployment status

The current core research compiler produces source-level HLS C++ and can prepare optional Vitis and Vivado **out-of-context** workflows. It does **not** provide automatic HLS IP export, Vivado block-design integration, board bitstreams, Linux/XRT host drivers, DMA transfers, board programming, or on-device inference. See [AMD toolchain setup](AMD_TOOLCHAIN_SETUP.md) and [FPGA deployment](FPGA_DEPLOYMENT.md).

## Evidence qualification

The following observations are **not interchangeable**:

1. Source compiler/contract acceptance;
2. Emitted C++ and selected numerical host checks;
3. Vitis C simulation;
4. Vitis HLS cycles and resources (estimates);
5. Vivado post-route timing/resources (specific implementation);
6. Actual board runtime/power (requires hardware execution).

Do not infer achieved frequency from a requested HLS clock. Some source configurations are numerically correct yet do not satisfy effective HLS timing. Earlier physical research includes out-of-context placed/routed cases, but those do not qualify all newer multi-operator experiments.

For evidence methods see [Scientific reproducibility](SCIENTIFIC_REPRODUCIBILITY.md). For simple valid examples, begin with [Using FPGAI](USING_FPGAI.md).


### Frozen ONNX weights and biases

- Supported static float32 ONNX initializers and standard float32 Constant tensors are retained as Model IR constant payloads and emitted as `static const data_t` HLS C++ arrays; separately passing a weights file is not required for these supported models.
- The generated RTL may implement these constants as block ROM/BRAM, distributed LUT structures or optimized logic, subject to synthesis. Fixed-point conversion (`ap_fixed`) means float32 values are not guaranteed to be represented exactly.
- `fpgai-audit-weights --onnx ORIGINAL.onnx --case EXISTING_CASE` independently checks the exact generated C++ constant literals against the ONNX input and source manifest, without exporting the values, rerunning tools or modifying original results.
- This audit **does not** establish final bitstream memory-cell values, FPGA programming, board execution, hot-loading new weights, or trained-model accuracy; a model-level Vitis/Vivado OOC result is not a board-deployed accelerator.


### Experimental board DMA/bitstream — native image generated, not qualified

- `fpgai-board` exports source-attested PYNQ-Z2 AXI4-Stream/AXI-Lite IP, creates a complete PS7 + AXI DMA Vivado design, and **has generated actual .bit/.hwh artifacts in native Vivado 2023.2** for the bounded toy model. The user's second build met routed timing (+2.311 ns), but remains **`BOARD_NOT_QUALIFIED`** due to four PS DDR critical warnings plus RTSTAT-10 and REQP-181 DRC observations. **No physical board programming or inference was demonstrated.** This independent board pilot does not turn the existing OOC `fpgai-run` command into a board deployment.
- `fpgai-board-reference` produces an independent float32 expected output. `fpgai-pynq-run` (or its copied standalone script) is intended to load a verified .bit/.hwh, run DMA and compare output on a real PYNQ-Z2 with explicit user consent.
- KV260 is deliberately rejected until a distinct Kria MPSoC platform and Linux runtime are engineered and validated. See [pilot](PYNQ_Z2_BOARD_DMA_PILOT.md).

### NCHW source contract checks (10 November 2026, author verified on `2689745`)

A prospective implementation extends the bounded general NCHW Conv/Relu/Add/Concat/MaxPool DAG source contract with independently derived Conv PE/replica/SIMD/fold ownership, rather than relying solely on the compiler's own `resolver.coordinate`. It separately requires modeled produced-input reads before their consumer's output commit. The independently calculated owner digest is exposed in E0 source and E1 actual GCC Conv READ receipts. The author verified these changes on `2689745`: **38 focused tests passed**, **2,395 full-suite tests passed (3 skipped)**, **176 research audit tests passed**. The NumPy integer digest regression found in the previous revision was fixed and retested. This does **not** establish actual RTL PE placement, initiation interval, memory-bank ports, full observed scalar/read/commit traces, grouped/dilated convolutions or arbitrary dynamic graphs. The source-level limitation and separate hardware qualification remain unchanged.

## Cross-network benchmark interface (11 November 2026, awaiting local verification)

An opt-in repository-only `scripts.network_benchmark` uses existing synthetic static ONNX generator and current canonical compiler to prepare MLP, pooling CNN, strided CNN and two-Conv Softmax study cases. Scalar versus operator-local independent architecture mapping, 16/6 versus 12/5 fixed-point CSim formats, and multiple requested clocks are explicit factors. Generated source files and their testbench/Tcl scripts are hash-attested and never overwritten. **Preparation is not E2/E3/E4**. Quantitative metrics require independent source/RTL-attributed vendor execution and the existing strict numerical/DRC/timing qualifiers. A successful OOC implementation is neither an achieved maximum Fmax measurement nor board inference latency. The source and tests for this interface have not yet been author-verified.
