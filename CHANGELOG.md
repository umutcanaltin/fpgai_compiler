# Changelog

## Documentation maintenance — November 2026

- Aligned the repository overview and installation instructions with the canonical `fpgai_compiler` GitHub URL.
- Consolidated licensing and permission notices without presenting working terms as an adopted license.
- Updated user guides, contribution information and documentation links.


This is the **public-facing project change summary**, not a replacement for the source-pinned scientific experiment archive. It does not claim a released software version or an approved open-source license.

## Experimental PYNQ-Z2 board preset decoupling — feature branch

- The native HLS-only IP export and read-only HLS plan no longer require
  installed third-party Vivado PYNQ-Z2 board files; both still use the exact
  Zynq-7020 part. Full bitstream stage continues to require the verified
  PYNQ-Z2 board preset.
- Optional `FPGAI_BOARD_REPO` allows separate user-owned board file
  repositories in Vivado discovery and full board build, without writing
  to the AMD installation directory.
- Added fail-closed tests for HLS without board-part, missing board-part
  for bitstream stage, and explicit user repository setup.
- Previous workstation test suite: 1,926 passed and 21 skipped; these
  **new changes are not yet workstation-verified**.

## Experimental PYNQ-Z2 board DMA/bitstream pilot — feature branch

- Added bounded 32-bit AXI4-Stream / AXI4-Lite HLS wrapper around existing model C++ and frozen ONNX constants, with separate native CSim and IP-Catalog export.
- Added a PYNQ-Z2 PS7/AXI DMA/SmartConnect/reset full-board Vivado script, strict clock, board part, bitstream and HWH gates.
- Added read-only planning, explicit vendor opt-in, DRC-clean default, narrowly opted-in full-board warning reports and standalone PYNQ host DMA/reference comparison.
- Added source-only negative tests for wrong boards, unsafe Tcl values, DMA packing, hash/shape mismatches and missing board qualification; **no native or physical board validation yet**.
- KV260 integration is a distinct follow-up; no files in older OOC cases or GitHub Actions were changed.

## Experimental ONNX weight-provenance audit — feature branch

- Added a read-only `fpgai-audit-weights` command that checks the exact ONNX model hash, functional Model IR fingerprint, generated source hash, and every float32 initializer/Constant HLS C++ literal.
- Reports aggregate parameter counts and a payload digest only; distinguishes source-level frozen weight evidence from fixed-point numerical equivalence, RTL-cell memory value verification and board deployment.
- Added mutation and source-provenance regression tests; native tests remain workstation-only.

## Experimental disclosed OOC advisory observation — feature branch

- Added a narrow, opt-in `--accept-ooc-advisories` path for timing-met, source-attested E4 OOC designs with only DPIP-2/DPOP-3/DPOP-4/RTSTAT-10 warnings. Default strict DRC qualification is unchanged.
- The `OOC_ROUTED_WITH_ADVISORIES` state preserves warning counts and explicitly marks strict DRC qualification false, never claiming a clean board image. Unknown, critical, unconstrained-I/O, incomplete, stale or timing-failing cases are rejected.
- Added standalone and integrated fail-closed regression tests. The native behavior remains to be validated on the user's workstation before merge.

## Experimental OOC diagnostics refinement — feature branch

- Added `fpgai-diagnose` for read-only inspection of existing Vivado invocation receipts, ROM stability and unpromoted timing/DRC reports.
- Exposed the existing strict source-/log-attested post-Vivado ROM-absence protocol only via explicit opt-in for new OOC cases; old runs remain unchanged.
- Distinguished Vivado process exit zero from a rejected evidence receipt in user-facing diagnostics.

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
