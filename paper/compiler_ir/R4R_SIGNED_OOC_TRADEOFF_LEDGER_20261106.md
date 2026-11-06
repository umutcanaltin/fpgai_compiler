# R4r — original source-attested routed OOC trade-off ledger (not strict E4)

**6 November 2026 — original R4q native evidence VERIFIED; R4r implementation and synthetic tests COMMITTED but NOT YET WORKSTATION-VERIFIED.** The author ran **44/44 targeted tests** and then a read-only R4q original Vivado report diagnostic, reporting `SIGNED_ORIGINAL_R4P_POSTRUN_DIAGNOSTICS_NOT_E4`. No new Vivado/Vitis processes were launched. Both original R4p Vivado OOC runs completed once with exit 0, and all RTL, ROM, invocation and routed report hashes are source-attested.

## Prespecified result — one 32/6 graph, two readers, same source schedule

| Original source-attributed routed observation | `full_storage` | `row_storage` | Row − full |
|---|---:|---:|---:|
| Vivado mapped BRAM tiles | **7** | **4** | **−3 (−42.9%)** |
| Vivado DSPs | **24** | **21** | **−3 (−12.5%)** |
| Vivado LUTs | **1,282** | **1,710** | **+428 (+33.4%)** |
| Vivado FFs | **883** | **1,022** | **+139 (+15.7%)** |
| Vivado URAM | 0 | 0 | 0 |
| Routed WNS, 12ns requested | **−2.170ns** | **−3.718ns** | **−1.548ns (worse)** |
| Source-attested route complete | YES | YES | both |
| Timing_met | NO | NO | neither |
| DRC warnings | **74** | **71** | **−3** |
| DRC critical warnings | 0 | 0 | none |
| Physical strict clean-DRC E4 | **NULL** | **NULL** | **NULL** |

Earlier HLS BRAM18K estimates were **13 vs 13** and source scratch memory declarations were three sites of **214 vs 134** 32-bit entries; neither number should be conflated with this independently routed Vivado BRAM mapping. The source-level matched loop/reader order was verified by R4k E1, **not** identical scheduled RTL cycles. The meaningful new observation is BRAM/DSP saving at a cost in LUTs/FFs and worse routed slack, not a universally superior design.

## Why E4 is still blocked

All four original routed reports per case have SHA-verified vendor invocation receipts; the R4q parser recognized every mandatory resource field, and routing is complete. Both signed original R4q `blocked_conditions` arrays are exactly `NEGATIVE_ROUTED_WNS` and `DRC_NONCLEAN`. Each failed the 12ns OOC timing constraint and had an unclean DRC summary. **An exit-zero Vivado invocation and valid resource counts do not imply timing-qualified E4 or deployed-board Fmax.**

| Exact DRC rule label | Full | Row | Delta |
|---|---:|---:|---:|
| `DPIP-2` Warning | 29 | 29 | 0 |
| `DPOP-3` Warning | 20 | 20 | 0 |
| `DPOP-4` Warning | 24 | 21 | −3 |
| `RTSTAT-10` Warning | 1 | 1 | 0 |
| **Total** | **74** | **71** | **−3** |

The warning names are preserved without assuming causes or safety implications. `unconstrained_io=false`, `critical_warnings=0`, and `board_deployment_qualified=false` for both source-attested observations. Even though warnings rather than critical errors were counted, **the prespecified strict clean-DRC E4 requirement is not satisfied**.

## New R4r code, connections, and engineering rules

- `src/fpgai_research/experiments/cst_r4r_original_routed_tradeoff_ledger.py` — read-only, no-vendor source-attested R4r comparison. Calls the ORIGINAL `cst_r4q_original_r4p_physical_diagnostics.inspect`, which independently rechecks R4p native attempt lock, exact source/ROM/vendor/log and original report SHA chain. Checks the original source revision `776e3a7a4b599fd114718f6a03ab73e06be17f5c`, exactly two original arms, each source-attested complete route, each finite nonnegative signed utilization value, exact original timing-met semantics, verified DRC rule totals and warning severities. Reports deltas and percentages with complete original denominator. Missing/tampered/unknown values yield NULL differences; NEVER impute zero or promote to E4.
- `tests/test_cst_r4r_original_routed_tradeoff_ledger.py` — synthetic fixtures grounded in the author's R4q stdout, with checks for exact observed descriptive values, invalid or missing resources, absent report/lock attestations, DRC total mismatch, incorrect timing state, source drift and create-only results. **These tests are not fresh Vivado experiments.**
- `paper/compiler_ir/R4R_SIGNED_OOC_TRADEOFF_LEDGER_20261106.md` — this documentation, evidence interpretation and source dependency index.
- `paper/compiler_ir/R4Q_ORIGINAL_R4P_PHYSICAL_DIAGNOSTICS_20261106.md` — original user-verified 44-pass diagnostic and raw source report results, immutable.
- `paper/compiler_ir/FIRST_PAPER_SCIENTIFIC_COMPLETION_20261106.md` — scientific gate and limitations updated; `main` and original vendor studies unchanged.

R4r optional output is create-only, in an entirely new directory OUTSIDE all original R4p/R4o/R4k/R4d/R4a study roots. It only writes `report.json` and its `artifact_sha256.json` in that new directory. No Vitis/Vivado, no GitHub Actions, no original source/routed report rewrite, no lock clearing or retry.

## Next local gate — safe and vendor-free

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q \
  tests/test_cst_r4r_original_routed_tradeoff_ledger.py \
  tests/test_cst_r4q_original_r4p_physical_diagnostics.py \
  tests/test_cst_r4p_matched_vivado_ooc_native.py

python -m fpgai_research.experiments.cst_r4r_original_routed_tradeoff_ledger \
  --study build/r4p_matched_vivado_ooc_native_20261106_01 \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --output build/r4r_original_routed_tradeoff_20261106_01
~~~

If this output path already exists, OMIT `--output` to produce a read-only stdout report. Preserve all original native attempts forever. After R4r, the scientifically useful NEXT question is to inspect **existing** critical timing paths and exact DRC warning contexts on the original routed checkpoint or textual timing reports **without running Vivado**, before preregistering any distinct clock sweep or architecture follow-up. A new design or implementation study requires a new separately named output and new source-hash lineage.

**First paper: MAJOR REVISION / NOT SUBMISSION READY.** We may report qualified descriptive source-attested routed resource values with explicit timing/DRC caveats, but not physical strict E4 superiority, real Fmax, throughput, power, or generality.
