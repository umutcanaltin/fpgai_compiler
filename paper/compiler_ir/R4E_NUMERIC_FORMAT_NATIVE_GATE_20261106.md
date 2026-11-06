# R4e — guarded native CSim/CSynth format-by-storage study

**Workstation update (6 November 2026): 6/6 R4e targeted tests passed, one matched 32/6 native Vitis pair executed and BOTH arms passed E2/E3.** See [actual 32/6 E2/E3 resource tradeoff and R4f read-only review](R4E_32_6_WORKSTATION_HLS_RESULTS_AND_R4F_REVIEW_20261106.md). The other two R4d source-format pairs (new 16/6 and 32/8) remain NOT RUN natively. Existing Vitis case folders must not be reexecuted or overwritten.

## Recorded original workstation prerequisite

- 23 focused R4a/R4b/R4c/R4d tests passed.
- Original R4b case 003, 16/6: 28/128 comparisons failed with max absolute error 0.055395125 versus tolerance 0.02 in BOTH storage arms. Identical aggregates do not demonstrate pointwise equality.
- New R4d host GCC E1 source study, revision abe6aa37e5c799f27312b7f3a17309c6122ed7cb, prepared three format-matched storage pairs (16/6, 32/6, 32/8), six source packages and zero Vitis/Vivado calls.

R4e uses the preserved original R4a paper root and NEW R4d source root. Its read-only preflight checks both immutable source SHA inventories, original baseline 16/6 package identity, matched sampled-input/model fingerprints, all source/host receipts, testbench/Tcl hashes, supported widths, target device, 12 ns requested clock, 1.5 ns explicit uncertainty and unmodified 0.02 CSim threshold.

The native runner reuses existing Vitis, summary parser and fail-closed timing/CSim qualifier. Only source-attested E2 numerical passes and E3-qualified synthesis estimates may yield a hardware resource comparison within each numeric format. E3 HLS estimates are not physical E4 results. Missing reports or E2 failures always give UNKNOWN resource deltas.

## First: run safe workstation tests

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git switch research/first-paper-source-memory-bridge-20261106
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pip install --no-deps -e .

python -m pytest -q \
  tests/test_cst_r4a_row_retention_study.py \
  tests/test_cst_r4b_row_retention_native.py \
  tests/test_cst_r4c_csim_numerical_triage.py \
  tests/test_cst_r4d_matched_numeric_formats.py \
  tests/test_cst_r4e_numeric_format_native.py
~~~

Define both original roots:

~~~bash
R4A=build/r4a_row_retained_paper_20261106_01
R4D=build/r4d_numeric_format_source_20261106_01
~~~

Read-only audit (ZERO Vitis calls):

~~~bash
python -m fpgai_research.experiments.cst_r4e_numeric_format_native \
  --study "$R4D" --original-study "$R4A" --audit-native
~~~

Dry run of just format index 1 (32/6), planning two original calls but making ZERO:

~~~bash
python -m fpgai_research.experiments.cst_r4e_numeric_format_native \
  --study "$R4D" --original-study "$R4A" \
  --start-format 1 --max-formats 1
~~~

## Second: optional native pilot, AFTER tests/audit qualify

Exactly one two-arm matched 32/6 pilot. Both --execute-native and --confirm-isolated-native are required:

~~~bash
python -m fpgai_research.experiments.cst_r4e_numeric_format_native \
  --study "$R4D" --original-study "$R4A" \
  --start-format 1 --max-formats 1 \
  --execute-native --confirm-isolated-native \
  --vitis-executable /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls
~~~

Re-audit the same original paths after the vendor attempts:

~~~bash
python -m fpgai_research.experiments.cst_r4e_numeric_format_native \
  --study "$R4D" --original-study "$R4A" --audit-native
~~~

Format index 0 is 16/6; index 1 is 32/6; index 2 is 32/8. All six arms remain in the denominator. A native arm is create-once: any prior invocation receipt, logs folder or HLS project makes rerun forbidden. The runner preflights EVERY selected arm before making the first vendor call. Interrupted original results must be preserved and read-only-audited, never deleted or silently retried.

**32/6 pilot observed:** both arms passed E2, both qualified E3; estimated BRAM18K 13→13 (tie), DSP 23→19, LUT 4990→4755, FF 1798→1647, worst cycles 22192→22602, estimated clock 10.302→9.500 ns. The numerical success supports a precision-related hypothesis but does not isolate rounding, saturation, or source-level numerical mechanism. Run the new R4f read-only audit before choosing further native runs; preserve existing 32/6 vendor receipts as original evidence.

## Next missing independent evidence

- Independent per-output and per-intermediate fixed-point error measurements and a separate quantized reference, using NEW source-pinned outputs, not edits to the old Vitis testbench.
- Finalize matched native precision cases only after the first result is interpreted and preserve all failures.
- Compare storage deltas within each fixed precision, not across widths.
- Physical claims require new clean-DRC placed/routed matched Vivado evidence, not just source scratch savings or HLS estimates.

**Paper remains major revision. One matched 32/6 R4e pair has source-attested E2/E3 evidence reported by the workstation, but no physical E4 result or BRAM saving is established. R4f code is new and not yet workstation-tested.**
