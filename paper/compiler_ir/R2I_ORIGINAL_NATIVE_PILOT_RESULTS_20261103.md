# R2i — Original native Vitis fan-out pilot, pair 0

**Source-pinned original workstation observation, November 3, 2026.**

## Source/test/experiment provenance

- Full source revision: `b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`.
- Full Python regression: **1,394 / 1,394 passed**.
- New focused R2i source-preparation tests: **6 / 6 passed**.
- Original study: `build/r2i_fanout_native_36_pairs_v1`.
- Native packages prepared: **72 cases, 36 prespecified matched pairs**,
  all distinct source files but each pair shares original functional,
  architecture, input, numerical testbench and Tcl SHA.
- One matched pair, `case_000` and `case_001`, has been **actually
  executed** with Vitis; the other **35 pairs have no original Vitis
  invocation** in the supplied observation.
- Device: `xck26-sfvc784-2LV-c`, requested 32 ns clock, explicit
  8.64 ns clock uncertainty, fixed-point
  `ap_fixed<16,6,AP_RND,AP_SAT>`, numeric tolerance 0.1.
- The two original vendor invocations returned exit 0 and both
  were independently qualified as **E2 CSim and E3 CSynth** with no
  auditor warnings. Source+tool provenance and effective HLS
  scheduling budget checks passed.
- The read-only auditor made **zero new vendor calls**.

## Measured first pair — graph fanout2_2x2x3x4, scalar Phi

| Metric | case_000 disjoint | case_001 per-element optimal | Disjoint minus optimized |
|---|---:|---:|---:|
| Source scalar scratch unit slots | 30 | 17 | +13 |
| Vitis HLS estimated BRAM_18K | 0 | 0 | 0 (TIE) |
| Vitis HLS estimated LUT | 2566 | 2564 | +2 |
| Vitis HLS estimated FF | 478 | 476 | +2 |
| Vitis HLS estimated DSP | 4 | 4 | 0 |
| Vitis HLS estimated URAM | 0 | 0 | 0 |
| Vitis HLS estimated worst latency cycles | 355 | 342 | +13 |
| HLS estimated clock period (ns) | 7.683 | 7.683 | 0 |
| Uncertainty-adjusted effective budget (ns) | 23.36 | 23.36 | 0 |
| HLS scheduling margin (ns) | 15.677 | 15.677 | 0 |
| Qualified source-attributed CSim | yes | yes | both pass |
| Qualified source-attributed CSynth | yes | yes | both pass |

**Interpretation:** This one exact paired native observation reports
13 fewer C++ scratch elements and 13 fewer **HLS-estimated** worst
latency cycles (355 to 342, a 3.66% decrease), with 2 fewer
LUTs and 2 fewer FFs. **Neither arm uses estimated BRAM**,
so this pilot demonstrates **no BRAM reduction**.

One small graph and one mapping policy provide no basis for a
general FPGA performance or hardware-memory claim.
HLS estimated clock period is **not achieved routed Fmax**,
and HLS latency-cycle difference is **not measured board throughput**.
Neither case has been submitted to source-attributed
Vivado placement/routing in this study.

## Full prespecified denominator — not yet completed

- Original Vitis invocations: **2 / 72**.
- Original E2-qualified CSim cases: **2 / 72**.
- Original E3-qualified CSynth cases: **2 / 72**.
- Original pairs with *both* native arms qualified: **1 / 36**.
- Among qualified original pairs: estimated BRAM savings
  **0**, ties **1**, regressions **0**.
- **35 / 36** pairs remain **UNKNOWN_NOT_BOTH_QUALIFIED**,
  not zero-BRAM ties or failed HLS cases.
- Strict clean-DRC E4 source-paired physical results: **0**.
- External compiler parity and independent firstness: **not established**.

The R2h source-memory result is independently verified but
must not be pooled with R2i vendor estimates as if they
were the same evaluation tier.

## Critical original-study handling

Continue only from the already prepared original
`build/r2i_fanout_native_36_pairs_v1` and **the locally pinned
checkout at the original SHA above**. Do not `git pull`
during this cohort: the source-pinning preflight requires
the prepared study's source revision to match HEAD.
Do not rerun pair 0 or delete its original vendor
invocation and report artifacts.

All remaining original pairs are explicitly enrolled.
The exact next native run selects `--start-pair 1 --max-pairs 35`
with both explicit native confirmation switches.
Any native failure, timeout, missing metric, BRAM tie or
regression must be retained as-is and included in the
completed 36-pair audit.
