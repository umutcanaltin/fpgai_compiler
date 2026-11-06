# R4q — original R4p signed Vivado report diagnostics without rerunning vendor tools

**6 November 2026.** R4p native synthesis/place/route was run ONCE by the author and **both Vivado 2023.2 OOC processes exited 0** (original full-storage execution ~290 seconds, row-storage execution ~279 seconds). The author ran the **read-only R4p audit**, which verified both original source/ROM chains, both original native invocation receipts, the source-revision-pinned permanent attempt lock, and the SHA-256 of all **four fresh Vivado reports per arm** (`route_status.rpt`, `timing_summary.rpt`, `utilization.rpt`, `drc.rpt`). These files are ORIGINAL completed vendor evidence and must not be overwritten or regenerated.

### Original observed R4p audit result

| Property | Full-storage row-source schedule | Row-storage same row-source schedule |
|---|---|---|
| Native Vivado process completed | YES, exit=0 (~290s) | YES, exit=0 (~279s) |
| Original RTL+ROM SHA integrity | PASS: 24 RTL + 3 ROM | PASS: 24 RTL + 3 ROM |
| Original attempt lock | ONE verified, original study | Same permanent lock |
| Fresh SHA-verified route report | YES | YES |
| Fresh SHA-verified timing summary | YES | YES |
| Fresh SHA-verified utilization report | YES | YES |
| Fresh SHA-verified DRC report | YES | YES |
| Original E4 observation status | `UNKNOWN_INCOMPLETE_PHYSICAL_RESOURCE_OR_TIMING` | `UNKNOWN_INCOMPLETE_PHYSICAL_RESOURCE_OR_TIMING` |
| Actual Vivado routed utilization published | UNKNOWN | UNKNOWN |
| Actual routed WNS published | UNKNOWN | UNKNOWN |
| Clean DRC qualified | NO | NO |
| Board qualified | NO | NO |

The original R4p pair audit returned `UNKNOWN_INCOMPLETE_OR_FAILED_PHYSICAL_PAIR`, `both_source_attributed_routed_observations_valid=false`, `both_strict_clean_DRC_OOC_E4=false`, `physical_E4=null`, and null WNS/resource deltas. **Do not call the run a failed route, a timing violation or a RAMB benefit without inspecting the signed reports.** The report parser succeeded before the R4p gate returned its generic unknown; the remaining diagnostic conditions are positive/nonnegative WNS and complete finite nonnegative LUT/FF/DSP/BRAM utilization. Either or both may be false; the original audit did not disclose which.

## New R4q source-only forensic analysis

- `src/fpgai_research/experiments/cst_r4q_original_r4p_physical_diagnostics.py` — reuses the original `cst_r4p_matched_vivado_ooc_native.audit_native` **as-is**, requiring both source-revision-locked, vendor-completed original arms and all source plus report SHA receipts. It independently rehashes every original report, parses the reports with unchanged `toolchains/vivado/report.py`, and discloses report-derived route complete, routed WNS and timing_met, identified physical utilization values, missing/invalid required resource fields, DRC and precise blocking conditions. If any mandatory resource field is unsupported, it also preserves bounded **verbatim labeled original utilization rows** rather than guessing column format or treating missing as zero. Re-verifies report SHA after parsing to reject time-of-check/time-of-use changes.
- `tests/test_cst_r4q_original_r4p_physical_diagnostics.py` — seven source-only synthetic tests: signed native reports with parseable values, negative WNS, unrecognized BRAM label, report SHA tampering, failed source provenance, absent native lock/population, and separate create-only report output. Synthetic values **do not establish original Vivado physical performance**.
- `paper/compiler_ir/R4Q_ORIGINAL_R4P_PHYSICAL_DIAGNOSTICS_20261106.md` — this paper protocol and file connectivity map.
- `paper/compiler_ir/FIRST_PAPER_SCIENTIFIC_COMPLETION_20261106.md` — status updated to acknowledge both original native Vivado invocations and the blocked R4p physical audit, without inferring a resource/clock benefit.

### Strict scientific interpretation

R4q is **READ-ONLY DIAGNOSTICS, NOT A NEW E4 EXPERIMENT**. It never launches Vitis or Vivado, never rewrites original reports, never clears the permanent attempt lock, never attempts reruns, and never converts a failed/unknown original R4p E4 verdict into an accepted result. A negative WNS is a genuine **timing failure at the requested OOC clock**, even when route is complete. Missing or unrecognized utilization columns stay UNKNOWN and can later motivate a separate parser fix, validated against original report examples, without changing source data. DRC completeness/cleanliness and board qualification remain independent gates.

## Required workstation verification — NO Vivado or Vitis calls

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pytest -q \
  tests/test_cst_r4q_original_r4p_physical_diagnostics.py \
  tests/test_cst_r4p_matched_vivado_ooc_native.py \
  tests/test_cst_r4o_matched_vivado_ooc_prepare.py

python -m fpgai_research.experiments.cst_r4q_original_r4p_physical_diagnostics \
  --study build/r4p_matched_vivado_ooc_native_20261106_01 \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --output build/r4q_original_R4p_physical_diagnostics_20261106_01
~~~

R4q creates only the new report folder and a SHA inventory. If the output already exists, OMIT `--output` and inspect stdout read-only; never delete/recreate or overwrite an original study.

**R4q implementation and new tests are COMMITTED but NOT YET WORKSTATION VERIFIED.** Once the original R4q output identifies the blocking conditions, amend the paper with verified WNS/utilization/DRC only. Any parser change must be reviewed against source excerpts and tested with actual formats; never infer physical advantage merely because both Vivado executables returned exit code zero. First paper remains **MAJOR REVISION / NOT SUBMISSION READY**, PR remains draft, `main` unchanged, and no GitHub Actions are used.
