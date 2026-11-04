# FPGAI Research roadmap

This document identifies **areas open to contribution**, not promised delivery dates or already-qualified FPGA features. Active scientific priorities may change after independent review.

## Reliable user-facing foundations

- Maintain a reproducible source-only installation, [CNN/dense example gallery](../examples/NETWORK_GALLERY.md) and compiler CLI.
- Keep operator capability tables tied to real import, reference, source-generation and native qualification evidence.
- Improve errors, precise shape/attribute diagnostics and discoverable architecture contracts.
- Stabilize typed Model IR and feature-focused APIs without breaking existing source-pinned research results.

## Compiler semantics and bounded composition

- Extend verified tensor-element readiness, last-use and issue-order contracts beyond currently bounded static MatMul families where a precise source-level semantics can be defined.
- Investigate legal memory aliasing and source generation for more multi-operator graphs without treating source scratch as physical BRAM.
- Carefully expand Conv2D/Dense/operator support while preserving numerical references, tail behavior and explicit unsupported cases.

## Experimental board integration milestone (unverified)

- Prototype PYNQ-Z2-specific HLS AXI4-Stream wrapper, AXI-Lite control,
  PS7/AXI DMA Vivado project, full .bit/.hwh build and standalone PYNQ host
  numerical comparison. See [PYNQ-Z2 board DMA pilot](PYNQ_Z2_BOARD_DMA_PILOT.md).
- CPU qualification gate: the user confirmed 1,926 passed, 21 skipped on
  2026-11-04. The installed Vivado 2023.2 has no PYNQ-Z2 board definitions.
  The experimental HLS-only command now accepts the exact Zynq-7020 silicon
  without a board preset; bitstream generation remains blocked without it.
- Qualification sequence: run source-only tests locally, inspect the real
  Vitis export, fix the first actual Vivado BD/Tcl issue, enforce timing/DRC
  checks and finally test transfers and output comparison **on a PYNQ-Z2**.
- Implement a *separate* KV260 MPSoC PS/platform, clock/reset, DMA and Linux
  device-tree/runtime flow only after verifying exact carrier board files and
  OS image. Do not reuse PYNQ-Z2 bitstreams or software indiscriminately.
- Never mark board operation passed from an OOC E4 report or a generated
  source/Tcl file.

## Future FPGA deployment integration

- Design and independently validate optional board-specific HLS IP/AXI interfaces, DMA buffers, clocks and resets, without changing existing compiler semantics silently.
- Establish an explicit reproducible full-bitstream and host-inference reference workflow for each claimed board, separately from existing out-of-context routing evidence.
- Document vendor setup in [AMD toolchain setup](AMD_TOOLCHAIN_SETUP.md) and missing stages in [FPGA deployment](FPGA_DEPLOYMENT.md); neither guide claims automatic deployment exists.

## Native evidence and scientific evaluation

- Reproduce individual Vitis/Vivado results only with explicit tool authorization and new original study workspaces.
- Report matched designs, negative outcomes, timing/DRC failures, numerical criteria and exact source/target conditions.
- Compare capabilities of relevant prior-art compilers fairly before making novelty or hardware QoR claims.

## Separate future optimization research

The RL/Gym code is currently preserved but **not a new compiler requirement**. Any future policy-learning experiments must first establish real source and hardware qualifications, matched baselines and generalization methodology. Do not encode vendor hardware behavior inside policy code.

## Documentation and governance

- Improve source packaging, documentation consistency and onboarding.
- Resolve source authorship, institutional/commercial IP, third-party attributions and final license.
- Review the full repository history for confidentiality and secrets before authorizing separately licensed distribution.

**Interested contributors:** Begin with a documented valid use case or reproducible bug in [Using FPGAI](USING_FPGAI.md), the [support matrix](SUPPORTED_FEATURES.md), [Contributing](../CONTRIBUTING.md) and [the operator guide](COMPILER_OPERATOR_CONTRIBUTION.md). No roadmap item authorizes unaudited vendor execution or claims unmeasured performance gains.
