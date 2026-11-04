# R2k — Pair 20 DRC forensics and next falsification decision

**Status (2026-11-03):** New source-only forensic code and tests committed; actual workstation report verification is still pending. Original R2i/R2j studies and prior exports are frozen. No new vendor invocation is implied by this document.

## What the original workstation paste establishes

Original routed OOC case_040 (disjoint) and case_041 (optimized) display the same reported ten violation classes and named DSP paths:

| Rule | Count per case | Explanation |
| --- | ---: | --- |
| DPIP-2 | 3 | DSP input pipelining suggestion |
| DPOP-3 | 3 | DSP PREG output not pipelined |
| DPOP-4 | 3 | DSP multiplier MREG not pipelined |
| RTSTAT-10 | 1 | 26 nets without routable loads |

The pair has routed BRAM 1.5 versus 1.5 (no saving), LUT 649 versus 684 (optimized +35), FF 396 versus 407 (+11), DSP 7 versus 7, and WNS +26.356 versus +27.304 ns. Its original E3 HLS BRAM estimate 3 versus 2 **did not survive implementation**. These remain **warning-only OOC observations, never strict clean-DRC E4**, and do not establish achieved Fmax, power, board qualification or an external compiler advantage.

**Do not infer that these warnings caused the physical resource regression.** Adding pipeline stages or changing DSP inference changes the architecture, cycles and resources; it requires a fresh prospective matched experiment with numerical/temporal qualification and new source attestations. RTSTAT-10 may relate to unconsumed or optimized-away signals in the isolated module, but that mechanism is not proven from the pasted excerpt.

## New hash-checked create-only original report audit

The `cst_r2j_drc_forensics` module reads the *actual* original `drc.rpt` for both cases, requires SHA-256 agreement with **both** the immutable R2j export and original Vivado invocation receipts, cross-checks all numbered DRC identities and counts against the existing source-attributed diagnostic, records nine named DSP witnesses and reported RTSTAT net counts, and compares the two arms. It has **no** GCC/Vitis/Vivado invocations and changes no DRC classification.

Run on the same workstation that retains the original artifacts:

~~~bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m pip install --no-deps -e .
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_r2j_drc_forensics.py tests/test_cst_r2j_physical_diagnostics.py
python -m pytest -q
python -m fpgai_research.experiments.cst_r2j_drc_forensics \
  --physical-study build/r2j_fanout_physical_36_pairs_v1 \
  --export-study build/r2j_fanout_physical_first_audit_v1 \
  --pair 20 \
  --output build/r2j_pair20_raw_drc_forensics_v1.json
~~~

The output is **create-only** and the originals stay unchanged. Any missing/symlinked/edited report, vendor-receipt mismatch, incomplete source cohort, or DRC summary contradiction fails closed. The exact RTSTAT net-name equivalence remains **UNKNOWN** rather than inferred from equal reported counts.

## Decision gate for the remaining experiments

The five other originally prespecified HLS-estimated BRAM-saving pairs are **23, 28, 31, 32 and 35** (original cases **046/047, 056/057, 062/063, 064/065, 070/071**). None has yet demonstrated routed memory savings. Before new original native calls, review unchanged part, clock, numerical semantics and original E3-qualified source/RTL/ROM manifests. Keep the full 36-pair denominator and all blocked E2-only results; retain every failure and regression.

Even running these five pairs would **not** establish strict E4 if the same warnings recur. Consider a separate, source-pinned matched better-constrained OOC/board-appropriate experiment only after this diagnostic; do not waive DRC or retrofit the original study. Independently reproduce against feature-equivalent public compilers before claiming external superiority or broad firstness.

**Publication is not ready** for universal speedup, routed BRAM advantage or clean-DRC physical validation. A bounded paper may report independently checked IR semantics, source-level reductions, qualified HLS-estimated effects, and the routed counterexample with explicit limitations.
