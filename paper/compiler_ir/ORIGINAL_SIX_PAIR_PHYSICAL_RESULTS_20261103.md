# CST NCHW Model IR — original six-pair routed resource observations

**Original user workstation report, November 3, 2026.** All six pairs
come from **one original source-attested** fixed-point HLS cohort
`build/nchw_native_v1` pinned at SHA
`8814bd0005e8a852348b49f19131b2410117216b`,
and Vivado physical follow-up
`build/nchw_bram_vivado_rom_v6`.
All **12/12** original native Vivado executions exited normally,
passed independent source/ROM/report provenance, were fully
routed and timing-met, and had verified **noncritical DRC warnings**.
Therefore **0/12 STRICT CLEAN-DRC E4 cases and 0/6 strict E4 pairs**,
despite **12/12 valid exploratory routed observations and
6/6 valid exploratory routed pairs**. Original warning policy
was pilot-informed; it is NOT an independently preregistered
DRC gate. No constrained board, bitstream, power, achieved
Fmax or cross-compiler speedup is established.

The independent physical export
`build/nchw_v6_full_physical_export_v1` passed, returning
`routed_observation_valid_cases=12`,
`routed_observation_valid_pairs=6`, original receipts 12,
`routed_qualified_cases=0`, `routed_qualified_pairs=0`.
The independent cross-cohort source reviewer reports original
CPU source revision `1cefdd8566658cb6b69e12444c9b5ce9ba6e4a31`,
which differs from native HLS source revision; NEVER mix
cross-revision results as a single comparison.

## Full six original disjoint-vs-shared pairs

**All deltas = DISJOINT MINUS SHARED**. Positive resource
differences mean shared uses fewer of that resource; positive
WNS difference means shared has LESS timing slack. Units
are vendor-reported `bram` units, LUTs, FFs, DSPs,
URAMs, and ns; no conversions to a different physical
BRAM primitive are justified by this record.

| Pair | Original cases | Disjoint BRAM | Shared BRAM | Delta BRAM | Delta LUT | Delta FF | Delta DSP | Delta WNS ns |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 000/001 | 2.5 | 2.5 | 0 | -155 | -10 | +3 | +0.195 |
| 1 | 002/003 | 32 | 24 | +8 | +57 | -234 | 0 | -1.470 |
| 2 | 004/005 | 4.5 | 4.5 | 0 | -4 | -120 | 0 | +0.537 |
| 3 | 006/007 | 0.5 | 0.5 | 0 | 0 | 0 | 0 | -1.182 |
| 4 | 008/009 | 16 | 16 | 0 | +9 | -507 | 0 | +0.707 |
| 5 | 010/011 | 3.5 | 6 | -2.5 | -350 | -21 | 0 | -1.046 |

The original numeric result is **NOT uniformly positive**:
- BRAM: shared saves in **1/6**, ties **4/6**,
  regresses **1/6**. Pair 1 = 32 to 24 (25% fewer);
  pair 5 = 3.5 to 6 (regression of 2.5 vendor BRAM units).
- LUT: shared saves in **2/6**, ties **1/6**,
  uses more in **3/6**.
- FF: shared saves in **0/6**, ties **1/6**,
  uses more in **5/6**.
- DSP: shared saves **3 DSP in pair 0**,
  ties the other **5/6**.
- WNS slack: shared is better in **3/6**, worse in
  **3/6**. No WNS delta is a valid achieved Fmax metric.
- URAM: tied in all 6.

Physical resource differences are descriptive outputs
from genuine independently audited original routed
case reports, not controlled timing/resource superiority
over another published compiler. Since every case is
warning-only, strict E4 primary numerical observations
remain absent. Observing BRAM reductions in a single
pair does not rescue a universal resource claim.

## Independent scientific R2 / RL controls

The separate complete original 40-stage source/GCC
paper campaign passed (latest workstation suite 1240/1240).
Counterfactual reader-Phi study: 24 original GCC
cases, 18 valid effective mapping interventions;
**18/18 changed C++ and event SHA**, but
**0/18 changed last-use SHA** and
**0/18 changed source scratch capacity** under
the fixed sequential original lowerer. This
**fails to demonstrate** stronger Phi-dependent
source-element lifetime discrimination in that
specific study, though per-iteration Phi ownership
and rejection controls are verified. Do not turn
an unchanged lifetime into a claim of a novel
schedule merely by changing a source hash.

RL-compatible C3 interface: 3 graph depths; 9
certificate proposals, 3 certified legal,
3 SOURCE-permitted but conservative UNKNOWN;
3 graph-rename invariance cases and
3 identical-candidate certificate checks.
**0 learned policies** and no RL performance or
held-out graph generalization claim.

## Scientific interpretation and remaining gates

The supported C1/C2/C3 **source evidence protocols**
are complete, not demonstrably novel relative to
ST-Flow, Allo, HIDA, StreamTensor or Wavelet.
The publication gate explicitly reports
`paper_submission_ready=false` and
`strict_physical_primary_gate=false`.
Full exact published-artifact parity, original
trained-network generality, prospective genuinely
`Phi`-causal source scheduling and independently
qualified physical comparisons remain outstanding.

Original audited files should NEVER be regenerated or
overwritten. A future R2b experimental study must be
prospectively distinct and use real emitted C++
source READ/COMMIT/RELEASE traces and independent
reference semantics under an explicitly fixed scheduler.
See `CST_R2_NEGATIVE_FINDING_AND_NEXT_FALSIFICATION.md`.

**This file records a user-supplied original terminal
observation, not a new native vendor execution or a
recomputed statistical analysis performed by GitHub.**
