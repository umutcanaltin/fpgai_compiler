# Changelog

## Documentation maintenance — November 2026

- Aligned the repository overview and installation instructions with the canonical `fpgai_compiler` GitHub URL.
- Consolidated licensing and permission notices without presenting working terms as an adopted license.
- Updated user guides, contribution information and documentation links.


This is the **public-facing project change summary**, not a replacement for the source-pinned scientific experiment archive. It does not claim a released software version or an approved open-source license.

## Experimental feature branch — one-command neuroscientist workflow

- Added a local `fpgai-run` command for ONNX -> HLS source -> Vitis CSim/synthesis -> optional Vivado OOC.
- Added explicit vendor-installation directory resolution, source-preserving ROM staging, numerical and synthesis qualification gates, and output summaries.
- Kept native tool invocation opt-in and GitHub Actions disabled; board bitstream/runtime generation remains out of scope.
- Added source-only regression tests and a workstation validation guide. Results remain subject to local Vitis/Vivado testing before merge.

## Unreleased — repository onboarding and release preparation (November 2026)

- Added an installable, documented architecture-aware ONNX-to-HLS research compiler quickstart using synthetic data.
- Added a public source-only neural-network gallery with three complete Conv2D-to-dense topologies and a two-layer MLP, deterministic synthetic ONNX, optional independent operator mappings and generated-C++/NumPy intermediate verification.
- Clarified typed Functional/Architectural/Lowered Model IR, supported operator and mapping constraints, and bounded compositional source-contract semantics.
- Separated research compiler documentation from the optional, currently frozen RL research environment.
- Reworked contribution and operator-extension guidance, examples, FAQ, security/citation notices and scientific evidence boundaries.
- Curated source-distribution files to exclude internal research diaries and private hardware study materials.
- Added a **nonpublishable, create-only clean-room source preview** and local documentation link checker, with regression coverage.
- Preserved previous internal landing pages and long scientific review notes in maintainer archives.
- The documentation maintenance did not issue a source license, rewrite history or execute vendor tools.

- Added [academic research-only draft terms](docs/ACADEMIC_RESEARCH_LICENSE_PROPOSAL.md) and a [permitted-user guide](docs/WHO_CAN_USE_FPGAI.md) without issuing a license before IP clearance.
- Added [AMD environment setup](docs/AMD_TOOLCHAIN_SETUP.md), read-only tool detection, a sourceable Vitis/Vivado shell helper, and [FPGA deployment guidance](docs/FPGA_DEPLOYMENT.md) explaining the distinction between HLS C++/OOC reports and real bitstream/board execution.
- Added a deterministic [synthetic input generator](examples/make_sample_input.py) for the explicit Vitis HLS numerical-testbench workflow.

## Earlier research development

The research prototype evolved through static ONNX importing, scoped architecture decisions, HLS C++ generation, reference comparison, explicitly qualified native toolchain studies and finite source-event/memory-lifetime contracts.

Experiments have different source revisions, cohorts and evidence strengths; detailed histories are preserved under `paper/compiler_ir/` and the source repository's Git history. Those chronological notes are not a stable public API or a claim that a particular hardware benefit has been shown in all configurations.

**Licensing:** The public repository has no operative source-code license. A separately licensed distribution requires the relevant rights and third-party review.
