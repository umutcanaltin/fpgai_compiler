# FPGAI implementation status

This document summarizes the current implementation status of FPGAI in product-facing terms. It is intentionally concise and does not track internal development iteration names.

## Core compiler

- ONNX import into FPGAI IR is implemented for the maintained operator set.
- Inference and training share the same compiler, tensor, memory, transport, and reporting foundations.
- Precision, memory, parallelism, tiling, build-stage, runtime, and board settings are represented through the canonical YAML contract.
- HLS C++ generation, Vitis HLS execution, Vivado handoff, implementation characterization, runtime packaging, and report generation are integrated compiler stages.

## Extensibility

- External packages use `fpgai.package/v1` manifests.
- Registry/discovery infrastructure supports deterministic package precedence and explicit loading.
- External logical operators can bind into ONNX import and FPGAI IR.
- External HLS implementations support flat-array and tensor-port ABIs.
- External VHDL implementations support scalar stream, scalar ready/valid, and grouped multi-port ready/valid contracts.

## Mixed-backend hardware

Validated architecture paths include:

- HLS-generated RTL connected to external VHDL RTL in one Vivado project.
- HLS → VHDL and VHDL → HLS physical boundaries.
- Ready/valid backpressure across mixed-language boundaries.
- Graph-driven physical composition.

Implemented and awaiting local Vivado validation: explicit multi-output split and multi-input merge VHDL contracts for physical DAG composition.

The maintained multi-port VHDL policy is `grouped_transaction`: all declared input ports participate in one logical input transaction and all declared output ports participate in one logical output transaction.

## Graph execution

- Branch-aware HLS generation is implemented.
- Tensor liveness and reusable activation-buffer allocation are implemented.
- Residual vector and residual CNN graphs are supported by the maintained HLS path.
- Mixed-backend DAG composition requires explicit split/merge nodes when one transaction branches or rejoins; implicit physical fanout is intentionally rejected so handshake semantics remain deterministic.

## Validation terminology

FPGAI reports implementation status using explicit levels such as:

- generated
- reference tested
- RTL simulated
- HLS synthesized
- Vivado synthesized
- Vivado implemented
- bitstream generated
- runtime validated

A compiler estimate is not reported as a measured hardware result, and timing closure at a requested clock is kept distinct from a clock-sweep Fmax characterization.

## Current boundaries

The following remain active development areas:

- arbitrary multi-port HLS RTL composition;
- independent per-port VHDL handshakes in addition to grouped transactions;
- wider mixed-backend DAG topologies and reusable buffering policies;
- broader CNN/residual model coverage;
- attention/transformer operator and IR coverage;
- training-capable external hardware implementation contracts and physical validation;
- broader board/backend coverage, including first-class VHDL and non-Xilinx targets.
