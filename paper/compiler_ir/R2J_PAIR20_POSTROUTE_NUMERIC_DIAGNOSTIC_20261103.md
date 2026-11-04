# R2j pair 20: post-route BRAM tie, area regression, DSP warning inventory

**Evidence date:** 2026-11-03. **Evidence tier: source-attributed,
routed OOC observation with DRC warnings, NOT strict E4.**

This document is based on the user's original **read-only**
`cst_r2j_physical_diagnostics --pair 20` output, generated after
the corrected diagnostic passed **16/16 focused tests and
1,418/1,418 full tests**. Neither the diagnostic nor this paper
note performed new original GCC/Vitis/Vivado runs.

## Immutable evidence provenance

- Original R2i HLS source SHA:
  `b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`.
- Original R2j source/RTL packaging revision:
  `adaa373145ed2e3ecf508b78091db4620338e777`.
- Corrected read-only diagnostic code revision:
  `142ca73d569ec6944ee30db3382eac333e887905`.
- Original native HLS study:
  `build/r2i_fanout_native_36_pairs_v1` (frozen).
- Original Vivado OOC study:
  `build/r2j_fanout_physical_36_pairs_v1` (frozen).
- Original physical first export:
  `build/r2j_fanout_physical_first_audit_v1` (frozen).
- NEW read-only numeric diagnostic artifact:
  `build/r2j_physical_pair20_readonly_diagnostic_v1.json`.
- Pair 20, original cases `case_040` (disjoint) and
  `case_041` (per-element optimal).
- **Correct graph/mapping:** `fanout3_3x3x4x4`,
  **`scalar`**, verified by the source-attested
  physical diagnostic. An older paper draft incorrectly
  said `last_reader_pes2`; this is an explicit correction.
- The original observer lists valid full
  source CPP SHA-256s for each case:
  - Disjoint: `07a35b4fcd7255784669d32854f8573303bb55daabcd541857b8d32b5aedbe6d`.
  - Optimized: `edfdaa7e4071036dd815fcf2913bdc853283dfa00eae2821d5329ac6d5bb11ea`.

## Actual Vivado routed resource measurements

| Metric | disjoint / case_040 | optimal / case_041 | optimized minus disjoint |
|---|---:|---:|---:|
| Source CPP scratch | distinct original per-arm source | certified scalar-unit alias layout | source verified in earlier R2h |
| HLS-estimated BRAM_18K (original R2i) | 3 | 2 | **−1 estimate** |
| **Routed BRAM tiles (Vivado)** | **1.5** | **1.5** | **0.0 (TIE)** |
| **Routed LUT** | **649** | **684** | **+35 (5.39% regression)** |
| **Routed FF** | **396** | **407** | **+11 (2.78% regression)** |
| Routed DSP | 7 | 7 | 0 |
| Routed URAM | 0 | 0 | 0 |
| Routed WNS (ns) | +26.356 | +27.304 | **+0.948 ns descriptive slack** |
| Complete route and timing met | yes | yes | both observations valid |
| DRC rule violations | 10 | 10 | no clean pass |
| DRC critical warnings | 0 | 0 | both zero |
| DRC unconstrained IO | no | no | neither |
| Strict clean-DRC E4 | **false** | **false** | **0 strict qualified pairs** |

The independent original read-only diagnostic reports
`descriptive_physical_resource_delta_disjoint_minus_optimized`
of BRAM **0.0**, LUT **−35**, FF **−11**, DSP **0**, URAM **0**,
and descriptive `optimized_minus_disjoint_WNS_ns`
**+0.9479999999999968**. These signs are
consistent with the table above.

**This is a meaningful negative result:** the earlier
HLS-estimated 3→2 BRAM_18K advantage **did not materialize
as a routed BRAM tile reduction in this pair**.
The optimized arm also increased physical LUT/FF.
These results refute any claim that fewer source-level
scratch slots invariably improve physical area.

No physical LUT/FF/BRAM advantage is claimed.
The +0.948 ns difference is a descriptive OOC WNS
comparison at a requested 32 ns clock, NOT achieved
maximum board clock, throughput or performance per watt.

## Exact DRC rule population — both cases

| Vivado DRC | disjoint | optimized | General meaning / investigation |
|---|---:|---:|---|
| `DPIP-2` | 3 Warning | 3 Warning | DSP input pipelining absent/insufficient |
| `DPOP-3` | 3 Warning | 3 Warning | DSP PREG output stage not pipelined |
| `DPOP-4` | 3 Warning | 3 Warning | DSP MREG multiplier stage not pipelined |
| `RTSTAT-10` | 1 Warning | 1 Warning | No routable loads, inspect net identity |
| **Total** | **10** | **10** | **not strict clean DRC** |

These are diagnostic descriptions of *rule types*, not a
proven claim that the exact same DSP cells or same route
are implicated on both sides. The user-provided summary
contains rule IDs/counts, not the affected DSP hierarchy
and net names. Inspect the saved original
`vivado/reports/drc.rpt` in each case if instance-level
root-cause evidence is needed. The original receipt uses
ROM evidence mode
`PRE_RUN_HASH_AND_VENDOR_REPORTED_READ_POST_ABSENCE`
for both arms; this is an independently accepted
**limited ROM provenance mode**, not proof of an
unchanged staged post-run ROM blob.

No DRC waivers, severity changes or assertion overrides
should be applied to the original routed studies.
A new pipeline-aware or registered-wrapper comparison
requires a **new source-pinned experiment and source
commit**, with new original source packages, and
both matched arms rerun under the same declared design
and timing constraints. Pipeline changes can also alter
latency and numeric/temporal behavior, so independent
equivalence checks would be necessary.

## Interpretation for the compiler paper

The complete original R2i HLS cohort remains:
**72/72 E2 CSim**,
**48/72 qualified E3 CSynth**,
**24/36 paired E3**,
**6/36 HLS BRAM estimated reductions**,
**18/36 qualified HLS BRAM ties**,
**12/36 E3 UNKNOWN**.
Those are native *estimates*.

The later original R2j physical pilot has:
**2/72 routed warning-only observations**,
**1/36 observed matched pair**,
**0/72 strict clean-DRC E4**,
**0/36 strict E4 pairs**.
The other **35/36 physical pairs remain unobserved/unknown**.
Routed BRAM for this observed pair is a **tie**, not
one of six verified routed BRAM reductions.

A viable bounded paper may state the IR's
independently certified source semantics,
successful 72-GCC source allocation proofs,
72 numerical native CSim checks, conditional
HLS estimated effects and this explicit physical
counterexample. It must NOT claim universal memory
or FPGA physical advantage, novelty of interval
coloring, arbitrary graph hardware deployment,
clean E4 qualification, or external compiler
superiority.

**Publication still requires wider physical
qualification and feature-equivalent external peer
comparisons.** Prioritize scientific honesty and
targeted experiments over repeating source-level
success cases.
