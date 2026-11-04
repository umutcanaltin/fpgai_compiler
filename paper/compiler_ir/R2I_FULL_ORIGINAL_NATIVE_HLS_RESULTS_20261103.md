# CST R2i — Completed 36-pair native Vitis HLS original study

**Original execution date:** 2026-11-03.
**Pinned compiler revision:** `b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`.
**Original root:** `build/r2i_fanout_native_36_pairs_v1`.
**Original read-only audit:** `build/r2i_fanout_native_final_audit.json`.

This document transcribes **user-provided original workstation
reports** and does not itself create or rerun vendor measurements.
The already completed original study must remain immutable.

## Original source integrity and qualification denominator

- Prespecified population: 36 matched disjoint-vs-optimal
  fan-out source pairs, **72 original Vitis HLS invocations**.
- Executed: **72/72**. Original native invocation receipts: **72/72**.
- Actual Vitis process return success in the supplied logs:
  all 70 newly executed cases exit 0; prior first pair also
  completed successfully.
- Independent numerical CSim qualified: **72/72 (E2)**.
- Independent strict CSim+CSynth timing/resource qualified:
  **48/72** cases, or **24/36** both-qualified matched pairs (E3).
- **24/72** cases, exactly **12/36** matched pairs,
  are **CSim only (E2)**. The vendor reports
  `HLS latency_best = undef` for these configurations.
  Exit code zero does not override missing HLS latency.
- Vivado placed/routed strict clean-DRC evidence **0**.
  Physical BRAM and achieved post-route Fmax remain unknown.
- No original Vitis reruns or source changes are permitted
  within this completed study.

## HLS-estimated BRAM contrast — all 36 pairs

| Outcome | Matched pairs | Denominator | What it means |
|---|---:|---|---|
| Estimated BRAM_18K reduction | **6** | All 36 | Both source-attested E3 reports valid; control uses one more BRAM_18K than optimal source |
| Estimated BRAM_18K tie | **18** | All 36 | Both E3 reports valid with equal BRAM_18K |
| Estimated BRAM_18K regression | **0** | All 36 | No E3-qualified pair showed an increase |
| **UNKNOWN/not both E3-qualified** | **12** | All 36 | At least one original report has undefined latency; cannot count as tie or reduction |

**Conditional on E3 qualification:** 6/24 (25%) reductions,
18/24 (75%) ties, 0/24 regressions. These are **HLS estimates**,
not actual routed FPGA BRAM improvements.

## Preselected HLS BRAM-reduction pairs

These are the six **reported E3 HLS-estimate reductions**:
all six show **disjoint 3, optimal 2 BRAM_18K**.

| Pair ID | Original cases | Fan-out | Mapping | HLS worst cycles disjoint → optimal |
|---:|---|---:|---|---|
| 20 | 040 / 041 | 3 | scalar (corrected after physical source audit) | 1243 → 1208 |
| 23 | 046 / 047 | 3 | all_pes2 | 1243 → 1207 |
| 28 | 056 / 057 | 4 | scalar | 1384 → 1351 |
| 31 | 062 / 063 | 4 | all_pes2 | 1384 → 1350 |
| 32 | 064 / 065 | 4 | scalar | 1706 → 1658 |
| 35 | 070 / 071 | 4 | all_pes2 | 1706 → 1658 |

All six pair IDs and exact HLS cycles are supported by the
supplied original vendor result log.

## HLS-estimated latency and negative/unknown results

Every **24/24** E3-qualified pair has fewer estimated
worst-case HLS cycles in the optimized arm in the original
records, including the earlier pair 0 (355→342).
This is evidence of a consistent **estimated** latency
change conditional on E3 qualification. It is NOT measured
board throughput or achieved routed Fmax. The 12 unknown
pairs have no qualified worst-cycle metric and are never
included as zeros in a mean.

The 12 unqualified matched pairs are:

`13,14,17,18,21,22,25,26,29,30,33,34`.

Both arms in each of those pairs passed CSim but have
HLS XML `latency_best = undef`. We do NOT substitute
the estimate from another shape, map, sibling policy
or earlier source generation and do NOT suppress
the negative evidence with a parser tolerance.

Possible explanations (e.g., nonstatic loop bound or
synthesis scheduling limit) are **hypotheses**, not proven
causes. Investigate them separately by inspecting the
original HLS synthesis log/XML without changing original
artifacts. A distinct new experimental source revision
is necessary before retrying modified source policies.


### Post-original-study correction and physical counterexample (2026-11-03)

The first source-pinned R2j physical diagnostic independently confirmed
that pair20's actual policy was **`scalar`**, not the mistakenly
transcribed `last_reader_pes2` label above; the table has been corrected.
Its exact graph is `fanout3_3x3x4x4`.
The original R2i source files, HLS reports, numerical receipts and
raw measurements are unchanged.

The original HLS **3 vs 2 BRAM_18K estimate** for pair20
was followed by a valid *warning-only routed* Vivado comparison
that measured **1.5 vs 1.5 actual BRAM tiles** (no savings),
**649 vs 684 LUT** (+35 optimized), **396 vs 407 FF**
(+11 optimized), **7 vs 7 DSP**, and
**+26.356 vs +27.304 ns WNS** (+0.948 ns descriptive
optimized slack). Both showed 10 DSP pipelining/no-routable-load
DRC warnings, so neither reached strict zero-DRC E4.
The full physical study denominator stays 36 pairs,
35 of which remain unobserved.

See [R2j pair20 post-route diagnostic](R2J_PAIR20_POSTROUTE_NUMERIC_DIAGNOSTIC_20261103.md).

## Valid paper claim, and the decisive next experiment

Supported: In this bounded 2–4 MatMul fanout,
left-Add-chain, source-serialized C++ workload, exact
source scratch array reduction was validated by
original 72-GCC execution (R2h) and then **72/72
numerical CSim** in R2i. Among the **24/36
E3-qualified native paired cases**, six reported
one lower **HLS-estimated BRAM_18K**,
18 tied and no regression, and all had lower
estimated worst-case cycles.

Not yet supported: an actual **routed** BRAM or
timing improvement, physical RAM port/bank
correctness, achieved Fmax, deployable KV260
bitstream, general DAG/operator correctness,
novel interval coloring, or superiority to
a pinned external FPGA compiler.

The next physical gate is to independently
package the immutable original HLS-generated
RTL (including all required ROMs), first
pilot a selected **E3 both-qualified pair
with estimated BRAM change**, and run strict
source-pinned Vivado OOC complete-route,
WNS, resource and **zero-DRC** qualification.
Any warning-only observation must remain
separate from clean-DRC E4, and any physical
regression must be reported.

The original R2i source revision is
`b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`.
New publication-document commits on `main`
**do not change** that original experimental
revision. Never rewrite old Vitis manifests
with new source hashes.
