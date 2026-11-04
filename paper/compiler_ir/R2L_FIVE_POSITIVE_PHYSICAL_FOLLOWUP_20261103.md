# R2l — Follow-up physical validation after original R2j counterexample

**Research status, 2026-11-03.** R2k real workstation forensics verified **24/24 focused and 1426/1426 full tests**, with zero vendor calls. Original HLS/Vivado artifacts remain unchanged. The exact source-attributed original pair20 report has matching DSP instance witness identities, identical DRC rule populations and equal 26-net RTSTAT no-load counts, but not yet exact RTSTAT net-name equality. Both original cases are warning-only OOC routed observations. They **do not qualify as clean-DRC E4**.

## Scientific objective and fixed denominator

Falsify the hypothesis that the original **six** independently qualified HLS-estimated BRAM-positive pairs produce actual routed BRAM savings. Predeclared positive indices from the original R2i full 72-case / 36-pair cohort: **20, 23, 28, 31, 32, 35**. The original pair20 physical pilot is preserved and already showed a 1.5→1.5 BRAM tie, optimized +35 LUT, +11 FF, +0.948 ns OOC WNS. The other **five** have no source-attributed routed pair observations.

This is *not* a replacement for source, numerical, timing or external-comparator validation. Outcome tables must always distinguish: (a) HLS **estimates**, (b) routed **warning-only descriptive** effects, (c) **strict clean-DRC E4**, (d) UNKNOWN/missing, and (e) still-unqualified external novelty.

The new R2l ledger **does not run Vivado**. Its ten isolated source-only tests passed during development; the actual user workstation regression and original-data run remain pending. It reconstructs original HLS E3 effects, re-audits original R2j results, checks all source/clock identities across newly prepared physical study, preserves all 36 pairs, and counts observed/strict/unknown results without changing either original cohort. It refuses to include an extra/nonpredeclared follow-up pair or repeat the original pair20 pilot.

## Source-only new checkout verification

Use the existing local repository. Do **not** delete, overwrite or rerun the original \`build/r2i_fanout_native_36_pairs_v1\` or \`build/r2j_fanout_physical_36_pairs_v1\` directories.

~~~bash
cd ~/Desktop/github_projects/fpgai_research
git status --short
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_r2l_physical_completion.py
python -m pytest -q
~~~

New R2l source-only planning: no vendor execution and no output file needed.

~~~bash
python -m fpgai_research.experiments.cst_r2l_physical_completion \
  --hls-study build/r2i_fanout_native_36_pairs_v1 \
  --original-physical-study build/r2j_fanout_physical_36_pairs_v1
~~~

Expected: six HLS-positive indices; original pair20 remains routed warning-only, **zero strict E4**, five still UNKNOWN. If the actual CLI detects a mismatch or fails a source guard, stop before new hardware execution.

## Prepare an independent, prospective follow-up physical study

Preparing the new study copies only original E3-qualified, source-attributed RTL/ROM into a **new** create-only directory. This is not a rerun of the original pair20 and invokes no vendor tools.

~~~bash
python -m fpgai_research.experiments.cst_r2l_physical_completion \
  --hls-study build/r2i_fanout_native_36_pairs_v1 \
  --original-physical-study build/r2j_fanout_physical_36_pairs_v1 \
  --prepare-followup-output build/r2l_five_positive_physical_followup_v1 \
  --output build/r2l_five_positive_plan_v1.json
~~~

The new directory contains a 72-case *manifest*, of which only the original E3 source cases are prepared for physical runs. It does not imply all 72 cases should be run; this follow-up targets **exactly five original matched indices**. The original pair20 is not rerun or overwritten.

## First new original pair: index 23, cases 046/047

Read-only audited dry run:

~~~bash
python -m fpgai_research.experiments.cst_r2j_fanout_physical \
  --study build/r2l_five_positive_physical_followup_v1 \
  --start-case 46 --max-cases 2
~~~

If both are \`PREPARED_NOT_RUN\` and all audit/report guards passed, the existing canonical R2j runner can execute **only** these two original Vivado OOC cases in the *new* follow-up directory:

~~~bash
python -m fpgai_research.experiments.cst_r2j_fanout_physical \
  --study build/r2l_five_positive_physical_followup_v1 \
  --start-case 46 --max-cases 2 \
  --execute-native --confirm-isolated-native \
  --timeout-seconds 14400
~~~

**This is an irreversible pair of prospective Vivado invocations.** Every failed/warned/unqualified result must remain in the study. Do not reexecute failed cases, change original source or waive DRC. After each pair, refresh the **read-only** ledger, using a new optional output filename if saving snapshots:

~~~bash
python -m fpgai_research.experiments.cst_r2l_physical_completion \
  --hls-study build/r2i_fanout_native_36_pairs_v1 \
  --original-physical-study build/r2j_fanout_physical_36_pairs_v1 \
  --followup-study build/r2l_five_positive_physical_followup_v1
~~~

If index23 completes with acceptable provenance, the next predeclared bounded case pairs are **056/057 (28), 062/063 (31), 064/065 (32), 070/071 (35)**. The ledger explicitly lists ready indices; do **not** run indices that are partially completed/failed/blocked, already observed, or not source E3-qualified.

### Scientific finish criteria

- Ledger all 36 matched pairs, all missing/failed physical observations retained UNKNOWN, and original plus follow-up **clearly labeled separate**.
- Exhaustively report paired BRAM/LUT/FF/DSP and OOC WNS without treating HLS cycle count or requested target clock as achieved Fmax.
- For strict E4 claims, independently require complete routing, provenance, nonnegative WNS and **zero DRC violations**. A warning-only result is descriptive, never clean-DRC qualified.
- If the remaining five pairs also fail to show routed BRAM savings, explicitly report the falsification. If any have warning-only savings, report those as *descriptive* and do not silently promote to E4.
- Broader publication still needs a separately source-pinned clean physical control and feature-equivalent publicly pinned peer comparison, including preexisting schedule/lifetime-allocation prior art.

**No claim of full scientific paper completion, clean-DRC physical win, generality or firstness is permitted at this stage.**
