# FPGAI Master Sprint Roadmap

This roadmap is cumulative. Community extensibility is a parallel program and
must not replace model support, scientific IR work, memory/dataflow research,
training, validation, cleanup, hardware analysis, or paper experiments.

## Execution rule

Implementation batches should combine two or three related sprints. Before any
change, inspect existing owners and refactor/wire existing paths. Every
architectural mechanism remains YAML-selectable and must materially affect
code generation, reports, or hardware. Inference and training share common
IR/tensor/memory/transport semantics.

## F5 — Parameter-gradient architecture and validation

- **F5C.1** Backend-neutral numeric capture schema for Python, HLS CSim,
  RTL/VHDL simulation, Vivado hardware, board runtime, and community adapters.
- **F5C.2** Populate Python-reference and HLS captures for losses, gradients,
  parameters, Adam m/v, and optimizer step.
- **F5C.3** Layer/tensor maps, precision-aware tolerances, mismatch localization,
  and mechanism-equivalence reports.
- **F5D** Matched HLS, Vivado, power/energy, and board-runtime comparison of
  full-buffer, tiled-accumulate, and fused-update designs.

## IR — Scientific FPGAI IR

- **IR1** Formal semantic layers: imported graph, canonical neural IR,
  training IR, tensor/memory IR, streaming/transport IR, schedule IR, backend IR.
- **IR2** Mutation, persistent state, loss/target/gradient/optimizer semantics.
- **IR3** Legality, compatibility, capability, and unsupported-combination analysis.
- **IR4** Stable pass manager, pass preconditions/preserved properties, IR snapshots.
- **IR5** Scientific comparison against ONNX-only and MLIR-based approaches.
- **IR6** Experiments measuring early invalid-configuration detection,
  lowering reuse, implementation portability, and YAML-to-hardware traceability.

## Models and reusable operators

- **M1** CNN prerequisites: Conv2D, depthwise/pointwise convolution, pooling,
  BatchNorm, residual add, concat, resize, multi-output graphs.
- **M2** ResNet and general CNN inference/training examples.
- **M3** YOLO importer/patterns, detection heads, preprocessing/postprocessing,
  memory/streaming plans, layerwise validation, HLS/Vivado/board experiments.
- **M4** SSD and U-Net families.
- **M5** Attention, LayerNorm, embeddings, transformer blocks, compact ViT/LLM.
- Every model sprint includes importer, IR, reusable lowering, inference,
  training where applicable, reference/testbench, YAML, reports, and support status.

## Memory, streaming, and transport

- **MEM1–MEM6** lifetime allocation, aliasing, BRAM/URAM/DDR residency,
  double buffering, checkpointing/recompute, paging/caching/compression/sparsity.
- **STR1–STR6** layer/tensor/gradient/weight/optimizer streaming, FIFO planning,
  fork/join, backpressure, deadlock analysis, compute/transfer overlap.
- **TR1–TR5** DMA/AXI scheduling, burst planning, precision-aware packing,
  multi-bank memory, external datasets and online execution.

## Precision and QAT

- **P1–P6** PTQ, QAT, mixed precision, accumulator/overflow/rescale analysis,
  rounding/saturation, master weights, optimizer/gradient precision, and
  accuracy-resource-latency experiments.

## Community extensibility

- **E1** Versioned extension manifest and stable ABI.
- **E2** Model-family, operator-definition, operator-implementation, and
  whole-model implementation registries.
- **E3** HLS C++ and VHDL/RTL adapters plus mixed-source project assembly.
- **E4** Memory-policy extensions.
- **E5** Streaming and transport-policy extensions.
- **E6** Training-mechanism, optimizer, and numerical-policy extensions.
- **E7** Backend, toolchain, board, runtime, validation-adapter, report-parser extensions.
- **E8** Validation status promotion: registered, schema validated, simulated,
  numerically validated, synthesized, implemented, board validated.
- **E9** Mixed built-in/community HLS/VHDL composition with clock/reset/tensor/
  memory/transport contracts.
- **E10** Catalog, licensing/authorship, version locks, examples, reproducible benchmarks.

## Validation and hardware behavior

- **V1–V6** Python and layerwise model behavior, gradient/optimizer/weight/loss
  analysis, CSim/CoSim, runtime captures, mismatch localization.
- **HW1–HW6** HLS schedule/II, Vivado timing/Fmax/slack, resource/data movement,
  activity, power/energy, board telemetry, and relation to YAML/IR decisions.

## Professionalization and paper

- **CLEAN1–CLEAN5** repository-wide dead-code and duplicate-path cleanup,
  compiler.py decomposition, schema/YAML deprecation audit, readable generated
  C++, all-knobs YAML and model examples.
- **PAPER1–PAPER8** reproducible benchmark matrix, baselines, IR ablations,
  memory/streaming/training/precision experiments, cross-board evaluation,
  figures/tables, artifact and claim-status traceability.
