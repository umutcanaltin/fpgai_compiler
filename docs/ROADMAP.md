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

- Complete the clean-room packaging test and public onboarding.
- Resolve source authorship, institutional/commercial IP, third-party attributions and final license.
- Perform a full historical confidentiality/secret review before authorizing any public release.

**Interested contributors:** Begin with a documented valid use case or reproducible bug in [Using FPGAI](USING_FPGAI.md), the [support matrix](SUPPORTED_FEATURES.md), [Contributing](../CONTRIBUTING.md) and [the operator guide](COMPILER_OPERATOR_CONTRIBUTION.md). No roadmap item authorizes unaudited vendor execution or claims unmeasured performance gains.
