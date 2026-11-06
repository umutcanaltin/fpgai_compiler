- R4y source manifest audit fix: JSON-normalize canonical tuple-valued source-slot metadata before comparison to saved JSON manifests; retain independent source SHA, exact package and all-file inventory checks. Added explicit round-trip regression; **new tests not yet author-workstation-verified**.
## 6 November 2026 — prospective R4y four-arm HLS source preparation, no native calls

- Author verified previous revision 42326c4 with **47/47** focused tests and read-only source-attested R4w/R4x, selecting 24ns post-hoc as conservative next experiment candidate; zero strict DRC-clean pairs and zero measured board latencies remain.
- Added source-only R4y four-arm preparation: full/row storage × canonical/producer MAC inner-loop PIPELINE II=1 requested, 24ns fixed target, 1.5ns uncertainty, same graph/32,6 precision, matched numerical testbench and canonical source-event identity. The original R4l baseline SHA must match before creating any new study.
- Added read-only source-package audit, create-only provenance, negative mutation/sidecar tests and Makefile plan/prepare/audit targets; detailed new protocol and file-level change map. Presence of a pragma is NOT evidence of achieved II, DSP pipelining or hardware speedup.
- All new R4y tests and source packages remain **unrun on the author's workstation**. No GCC, Vitis, Vivado, original evidence mutation, GitHub Actions or main merge.

## 6 November 2026 — 24ns retrospective margin-based planning, not board qualification

- The author pulled the previous R4x implementation at `066a57c` and verified **43/43** focused tests, full original R4w 16/20/24ns audit and R4x read-only comparison: exactly **two warning-only timing-met matched pairs**, strict clean-DRC E4 pairs **zero**. No new native calls were made by either audit.
- Added a deterministic **post-hoc** read-only clock-planning policy (minimum paired WNS >=1.0ns) on existing source-attested R4w observations: select **24ns** (full/row routed WNS **1.249/2.555ns**, min **1.249ns**), not 20ns (min **0.433ns**). Explicit JSON states **NOT physically safe or board qualified**, hold timing/all-path constraint coverage unknown, no achieved Fmax/real latency claims.
- Strengthened prospective experimental protocol: matched source/precision/part, new four-arm storage×DSP pipeline HLS source and numerical checks at 24ns, preserved source and routed reports, independent hold/constraint/DRC inspections, staged ≥3 graphs/physical configurations if a hardware-general claim is pursued, and board measured cycles/II/DMA latency only after qualification. The main manuscript still prioritizes original C1/C2 source semantics and independent novelty gates.
- No new Vitis/Vivado attempts, no original evidence mutation, no GitHub Actions, no merge. New source/test edits are **pending local author tests**.

## 6 November 2026 — verified six-clock audit and non-promoting advisory sensitivity

- The author pulled revision `5bbea74`, passed **32/32** focused R4w/R4x/R4v tests and independently audited the six already executed source/RTL/ROM-attested Vivado routes. Full/row WNS pairs: **+0.241/−0.975ns** at 16ns, **+0.433/+0.538ns** at 20ns, **+1.249/+2.555ns** at 24ns. The earlier generated-sidecar audit blockage is resolved. Nonzero recognized DRC findings **74/71** prevent every strict clean-DRC matched pair; core and board latency remain unknown.
- The existing R4x *strict* eligibility and measured-latency NULLs are unchanged. Extended its source-only JSON report with a separate, explicitly non-promoting **advisory-only HLS cycles × requested timing-met period** sensitivity and descriptive mapped resources, guarded by original attempt lock, timing/DRC/status and resource invariants. Added 20/24ns positive fixtures and tampered-source/DRC negative tests, not yet independently tested on the author's workstation.
- Reconciled the active manuscript, public research status, R4w method and change-impact ledger with actual original author results. **No new vendor runs, no workflow, no original output mutations, no draft PR merge.**

## 6 November 2026 — original six Vivado runs, auditable source sidecar repair

- Author original 54-profile E1 read-only analysis completed after **40/40 targeted tests passed**: **median 15.5 saved source scratch slots**, **50% median per-profile reduction**, per-profile savings 4–45 slots and 38.461538–55.555556%, maximum final/intermediate absolute numerical errors 2.3781921365539915e-7 / 2.408715820134688e-7, **271 original artifacts**; zero new compiler/GCC/vendor calls during the analysis. Aggregate 1999 vs 1009 source slots are across **distinct** generated programs, never a BRAM observation.
- Author completed all **six** one-time new R4w Vivado 2023.2 processes for full/row at 16/20/24ns, process exits all 0; earlier 12ns original output and locks unchanged. The subsequent R4w and R4x read-only audits both stopped at `audit_prepared`: the original audit wrongly rejected **tool-generated sidecars** in the `vivado/rtl` working directory. WNS/DRC and valid physical latency remain UNKNOWN.
- Fixed R4w `audit_prepared` to allow non-design/native-generated files in the native `vivado/rtl` work directory **only when the original, unaltered, valid one-time native attempt lock exists**. Still requires exact original SHA for staged RTL/ROM/Tcl, forbids postrun extra `.v/.sv/.dat/.mem/.xdc/.tcl/.xci` design inputs, rejects symlinks and any extra canonical HLS source files, and preserves existing native receipt and strict routed WNS/zero-DRC requirements. The native tool **must not be rerun**, and no `build/` evidence was changed.
- Added four targeted synthetic postrun sidecar/lock/source-tamper regression tests in `tests/test_cst_paper_r4w_feasible_clock_sweep.py`, updated paper original-vendor execution status and C1 source effect-size table, and updated all file/dependency documentation. Fix and tests **not yet author-workstation-run**; next commands are tests and **read-only** R4w/R4x audits.

## 6 November 2026 — valid-design-only Vitis/Vivado timing and latency research setup

- Corrected a local `--summary-only` fixture failure in `cst_paper_c1_original_e1_readonly_descriptives`: the CLI now passes its exact original SHA manifest constant explicitly to `analyze()` so that test fixture monkeypatching works; original production attestation remains fixed. **The revised test is not yet author-rerun.**
- Added original R4l Vitis E2/E3 vs R4p/R4q/R4r Vivado OOC **read-only** cross-stage Pareto report and synthetic source/negative-timing tests. These are separate from the completed 54-profile E1 source cohort; no new hardware result is claimed.
- Added `cst_paper_r4w_feasible_clock_sweep`: new **create-only**, original RTL/ROM-SHA-staged paired **16/20/24ns** Vivado clock feasibility pilot (6 native cases), double-confirm native opt-in, permanent attempt lock, original source integrity re-audit, full routed WNS/resource/DRC metrics. It reuses the existing narrow conservative DSP warning-only OOC observation classifier but never waives warnings or promotes them to clean DRC or board deployment.
- Added `cst_paper_r4x_feasible_only_core_latency`: **no latency winner unless both arms meet timing and clean DRC at the exact same clock**; `cycles × feasible_period` is explicitly an **estimate**, measured core/DMA/board latency remains null. Added six-source-stage tests, pipeline/latency method documentation, strict and failure-preserving make targets.
- Corrected prior interpretation of original DPIP/DPOP messages: they are **DSP48 input/output pipelining recommendations**, not evidence of memory-port collisions. A new storage×DSP-pipeline 2×2 experiment requires independent Vitis numerical/schedule qualification followed by Vivado timing, and separate core/DMA/board measurement before physical performance can be published. **No new Vitis or Vivado run was launched by these commits; all new code/test checks require the author's workstation.**

## 6 November 2026 — author verified full C1 E1 source bridge and source-slot analysis prepared

- The author completed a separate **full, original source-revision-pinned C1 E0→generated-C++ E1** study at `3bd1e36ff49864d89fa4c676e61da8ba0b936fce`: **17 focused tests passed**, unchanged original E0 54-profile/486-trace report SHA verified; **54 canonical source profiles × 2 scratch layouts = 108 real GCC compilations/executions**, all original canonical E0 hashes matched, **54/54 positive source-slot reductions**, no vendor calls. Its independent read-only SHA audit verified **271 source artifacts** and original manifest SHA `a79e700208e5d7cada93dd1f488809b6622f52cf6b14fa7b7d7b05a8d624d83e`. This is host source E1 evidence only, not BRAM/Fmax/board latency.
- Added full original result ledger `paper/compiler_ir/C1_FULL_54_PROFILE_E1_GCC_ORIGINAL_RESULTS_20261106.md`, updated the active C1/C2 manuscript, evidence/claim lock, source E1 preregistration, public research guide and archive index, without modifying any original experiment, original native R4p/R4u locks or production compiler semantics.
- Added **new, unexecuted** read-only SHA-pinned per-profile C1 E1 analyzer `src/fpgai_research/experiments/cst_paper_c1_original_e1_readonly_descriptives.py` and synthetic/negative original-manifest tests; `make research-c1-e1-review` is the safe no-compiler/no-vendor invocation and `make research-audit-tests` includes its tests. This will quantify saved source slots, percentages, address witnesses and numeric-error distributions without recompilation. Exact source numerical effects are not claimed until the user runs it.
- The critical external matched Allo/CODO semantic novelty test, timing-qualified FPGA latency and clean-room licensing/IP/history release review remain open, so PR #2 stays draft. Eager C2 source catalogs still lose to lazy caching in 108/108 original CPU trials.

## 6 November 2026 — completed original C2 cost analysis and genuine C1 E1 GCC smoke

- The author pulled source revision `5bc8e9c1ec749c1c3dcf09bf85bace91503ee0c7`, passed 30 targeted source/public tests (1.60 s), and verified all 168 public relative links (31 Markdown files, zero unresolved). The public-source preview selected 472/895 tracked files and still reported **no legal publication clearance**.
- The existing immutable C2 source benchmark (`637f274...`, original report SHA `be2adaa4acf30196a2ccb17114f2caa0af3e6679dc2685872e27cb086ca86ec1`) was analyzed read-only across 108 paired workloads / 324 measured method rows: **lazy caching faster than eager full catalogs in 108/108** paired CPU trials, with total original measured source calls **lazy 198 / eager 864 / direct 1,476**, plus 48 excluded known-certified eligibility preflight calls. No new source compiler or vendor calls were made during the analysis. The paper now reports full six-condition median CPU ms data and does **not** claim eager source screening is cheaper than lazy.
- The author executed a separate original two-profile matched C1 E0→generated-C++ E1 smoke at `5bc8e9c1…`: **4 actual g++ compilations and executions**, exact original E0 canonical events matched, and positive source slot savings in **2/2** original profiles. The read-only audit checked **11 files** with manifest SHA `51e022a07e35ab4ee96aed25d4a7bb990d97d49b54e98a07d977a6ed0bce4435`; 54-profile/108-GCC full original study remains **not yet executed**. No FPGA latency/BRAM claims.
- Added [source-pinned verified C2 cost and C1 E1 smoke interpretation](paper/compiler_ir/C1_E1_SMOKE_C2_SOURCE_COST_ORIGINAL_RESULTS_20261106.md) and updated manuscript, evidence-to-claim ledger, E1 protocol, scientific strengthening plan and public research status. This is scientific evidence accounting and publication hygiene, not a new product compiler implementation.

## 6 November 2026 — direct public documentation fix and stricter scientific novelty gate

- Fixed the locally observed `make release-links` failure (`ModuleNotFoundError: No module named 'scripts'`) by making `scripts/check_public_docs.py` support both direct and module invocation, changing Makefile to `python -m scripts.check_public_docs`, and adding a direct-vs-module subprocess regression test. **New fix not yet workstation-tested.** The preceding user run verified 55 targeted tests and a clean 894-file/718-module static source inventory before the command stopped.
- Found and explicitly incorporated Allo PLDI 2024 Section 5.2/Figure 6: its verifier already detects a producer/consumer FIFO access-order mismatch in two MatMul kernels. The paper no longer treats elementary source order verification as novel. Added falsifiable A–G semantic comparison protocol with exact external `SUPPORTED/UNKNOWN/SETUP_FAILED` distinctions and C2 lazy-cache/latency stop rules. **No new external Allo or native HLS experiment executed.**

## 6 November 2026 — scientific results and non-destructive public artifact cleanup

- Original source-revision `637f2745721b32d6f5a7e02c4659b0623bc9eee7`: 72 targeted source tests passed, C1 seeded held-out 54 profiles/486 typed event variants/0 observed oracle disagreements, and C2 324 source-compiler method rows with matching source identities and 48 explicitly excluded setup compiler calls; immutable original SHA reports retained in ignored `build/`.
- Introduced SHA-pinned **read-only** C2 postrun CPU cost analysis; per-method CPU timing conclusions remain unverified until the author's original analysis run. Physical FPGA latency remains UNKNOWN for the earlier failed-clock routed pair.
- Added `.gitignore`, public-safe GitHub issue/PR templates (NO GitHub Actions), a research publication/reproducibility guide, and publication-hygiene regression tests. Repaired public preview/source-distribution file inclusion for the separately linked PYNQ-Z2 pilot guide, which remains BOARD_NOT_QUALIFIED.
- All 880 tracked-file baseline names on the original branch were inspected at its fixed 6-Nov source SHA. Existing source modules, full historical R3/R4 evidence, research manuscripts, RL work and original results were **not deleted, moved or retroactively rewritten**. Publicly viewable source remains **unlicensed for redistribution** pending institutional/third-party rights, full Git-history safety review and explicit release approval.
- Research manuscript and original evidence/claims ledger updated without claiming external compiler novelty or finished paper submission. This cleanup is committed on the research branch, not merged into `main`. New hygiene and analysis tests still require workstation execution.
- Added a static tracked-file source ownership/AST import map and six synthetic tests, plus `make research-inventory`, `make research-audit-tests`, `make research-c2-review` contributor entry points. The map does not infer dynamic callers or safe deletion, and the new code remains workstation-unverified.
- Added a source-pinned, two-layout **actual GCC** C1 E0→E1 controlled refinement study for the original 54 new-shape source cohort, using existing R2h source emission and independent NumPy/scratch witnesses: **108 new GCC executions planned, not executed**. An isolated two-profile scalar/mixed-Φ smoke and immutable source-audit protocol are provided. Frozen original C1 results remain unchanged.
- Corrected the source-pinned external Allo runbook to the active `fpgai_compiler` checkout; external semantic parity is still unexecuted and a P0 publication blocker.
## PYNQ-Z2 source-backed PS/DDR read-only audit — feature branch

- User-run second Vivado full-board implementation generated .bit/.hwh,
  passed +2.311 ns routed slack, and resolved the read-only reset
  warning. State remains `BOARD_NOT_QUALIFIED` with four independent
  PSU-DDR critical warnings and RTSTAT-10/REQP-181 DRC observations.
- Added `fpgai-board-ddr-audit` to verify untouched .bit/.hwh/report
  hashes, native DDR warning messages and board-preset source values,
  and compare those values to Xilinx/PYNQ's published PYNQ-Z2 hardware
  reference (without changing DDR parameters or clearing warnings).
- Added isolated negative tests for tampered artifacts, missing/changed
  source warning values, incorrect board identity and symlinked
  board presets. **The new audit must be tested on the workstation**;
  no on-device testing or production qualification is claimed.

## First full PYNQ-Z2 bitstream evidence and qualification hardening — feature branch

- User-run Vivado 2023.2 successfully generated `.bit` and `.hwh`,
  with +2.197 ns routed slack and completed routing, but correctly
  returned `BOARD_NOT_QUALIFIED` for RTSTAT-10 (warning) and
  REQP-181 (advisory). No physical hardware inference was run.
- Identified four `PSU-` critical warnings for negative PS DDR skew
  parameters and one read-only `BD 41-737` reset polarity warning
  *outside* the separate full-board DRC summary.
- New qualified-board gate inspects native Vivado stdout and stderr
  for any `CRITICAL WARNING` and reports their separate code/count
  without hiding them under `drc_critical_warnings: 0`.
- Replaced the read-only `proc_sys_reset.C_EXT_RESET_HIGH` setter
  with runtime Tcl polarity inspection and a conditional 1-bit reset
  inverter. No PS DDR parameter guesses or DRC severity changes.
- Added source-only tests for non-DRC critical warnings, unavailable
  native logs, and the read-only reset polarity regression; the revised
  full board system must be re-run with licensed Vivado before qualification.

# Changelog

## Documentation maintenance — November 2026

- Aligned the repository overview and installation instructions with the canonical `fpgai_compiler` GitHub URL.
- Consolidated licensing and permission notices without presenting working terms as an adopted license.
- Updated user guides, contribution information and documentation links.


This is the **public-facing project change summary**, not a replacement for the source-pinned scientific experiment archive. It does not claim a released software version or an approved open-source license.

## Experimental PYNQ-Z2 board HLS CSim compatibility — feature branch

- Fixed a Vitis HLS 2023.2 compilation error in the generated *board
  adapter testbench*: direct `ap_range_ref != af_range_ref` was an
  ambiguous overload. Raw 32-bit AXI data and fixed-point golden output
  bits are first materialized as matching explicit `ap_uint<W>` values,
  then compared bit-for-bit.
- Added source-only negative regression checks for W=8/16/24/32 and
  preserved the canonical ONNX model, weights, AXI adapter and prior
  failed original native case unchanged.
- User independently confirmed 35 focused tests passed and PYNQ-Z2 board
  definition was successfully discovered; **the corrected HLS testbench
  still requires a new real native run**, with no bitstream claims.

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
