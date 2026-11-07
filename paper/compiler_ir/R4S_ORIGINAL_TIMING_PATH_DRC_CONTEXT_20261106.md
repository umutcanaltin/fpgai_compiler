# R4s — original source-attested timing-path and DRC context (read-only lexical study)

**6 November 2026. R4r ORIGINAL WORKSTATION VERIFIED**: 40/40 focused R4r/R4q/R4p tests passed (0.38s), and the original read-only source-attested ledger was created at `build/r4r_original_routed_tradeoff_20261106_01`. It independently verified the mapped physical descriptive full→row changes: BRAM **7→4** (−3, −42.857%), DSP **24→21** (−3, −12.5%), LUT **1282→1710** (+428, +33.385%), FF **883→1022** (+139, +15.742%), URAM **0→0**; routed WNS **−2.170→−3.718 ns**, so row is **1.548ns worse** at the same 12ns requested clock; DRC Warning totals **74→71** with no critical warnings. Both original arms failed timing and strict clean DRC. `physical_E4` remains NULL. NO new vendor calls, source or report modifications.

**R4s NEW: IMPLEMENTED and committed, NOT YET WORKSTATION-TESTED.** It inspects only original report text already generated and SHA-attested by R4p/R4q/R4r; it NEVER reruns or alters Vivado. It does not claim a new routed E4 outcome, design optimization, board-level performance or root cause.

## Scientific question

Can the existing *signed* `timing_summary.rpt` and `drc.rpt` files expose explicit timing-path source and endpoint identifiers, path slack, path-group classification, and textual DRC rule contexts that narrow the possible causes of the row-retained design's BRAM/DSP reduction but LUT/FF growth and WNS degradation? Can the study distinguish absent path-level details from actual evidence?

Only source-attested **explicit** path detail may be reported. The design's source issue schedule is matched, but no hardware schedule equivalence is proven. A negative WNS and a DRC warning category do not identify an RTL net, RAM port or physical timing cause. No physical period/Fmax extrapolation is permitted.

## File, dependency and connectivity audit

| Path | Role and relationship |
|---|---|
| `src/fpgai_research/experiments/cst_r4s_original_timing_drc_context.py` | NEW read-only main CLI. Invokes original `cst_r4q_original_r4p_physical_diagnostics.inspect` and unchanged `cst_r4r_original_routed_tradeoff_ledger.summarize` as strict original SHA/provenance gates. Rehashes signed timing/DRC text and extracts bounded original path and DRC context, or records UNKNOWN. Never executes Vivado/Vitis. |
| `tests/test_cst_r4s_original_timing_drc_context.py` | NEW eight synthetic no-vendor tests for complete/incomplete path witnesses, bounded text, DRC rule context, source hash tampering and create-only output. These synthetic witness values are NOT actual physical evidence. |
| `src/fpgai_research/experiments/cst_r4q_original_r4p_physical_diagnostics.py` | EXISTING signed native report gate, original source/ROM, 4 report SHA receipts per arm. UNCHANGED. |
| `src/fpgai_research/experiments/cst_r4r_original_routed_tradeoff_ledger.py` | EXISTING paired source-attested tradeoff gate, both arms original revision, timing and DRC checks. UNCHANGED. |
| `src/fpgai_research/toolchains/vivado/report.py` | EXISTING original physical parser. UNCHANGED; R4s never substitutes its own timing or resource measurements. |
| `paper/compiler_ir/R4S_ORIGINAL_TIMING_PATH_DRC_CONTEXT_20261106.md` | NEW protocol, engineering rules and safe workstation command. |
| `paper/compiler_ir/R4R_SIGNED_OOC_TRADEOFF_LEDGER_20261106.md` | Original 40-pass workstation evidence, interpreted as descriptive-only physical OOC. |

## Safety and scientific interpretation

1. Original R4p native vendor attempts at `build/r4p_matched_vivado_ooc_native_20261106_01` are **ONE-TIME COMPLETED**. Preserve all 24 Verilog + 3 ROM original source files per arm, eight signed physical reports, vendor invocation receipts and permanent attempt lock. DO NOT issue `--execute-native` again.
2. Timing-path parser recognizes only bounded explicit `Startpoint:`/`Endpoint:` or `Source:`/`Destination:` text and a separately explicit numeric `Slack (VIOLATED|MET)` label in the same path block. Caps at five displayed paths per arm, does NOT treat them as all paths and does NOT assume the first is worst unless explicitly established by signed source. If the original report contains only a Design Timing Summary without individual paths, status is **`UNKNOWN_NO_COMPLETE_PATH_DETAILS_IN_ORIGINAL_TIMING_SUMMARY`**. Never fabricate startpoints/endpoints or suggest a root cause.
3. DRC extractor preserves at most three original source lines per observed rule, exact source line numbers and previously SHA-attested DRC counts. A summary row or warning category is only a source textual label, not a causal analysis or a fix. Missing context is UNKNOWN; report samples never replace the verified DRC population.
4. The original source text SHA-256 must match the 4-report R4p vendor receipt inherited through R4q. Rehash both timing/DRC after parsing to reject changes during inspection; any drift is a hard failure. No source path can be read from unsourced or unsigned original files.
5. Original WNS and physical mapping stay governed by R4p/R4q/R4r. R4s never changes their qualification or produces non-null strict `physical_E4`; board deployment and achieved Fmax are always unqualified.
6. Optional report output is create-only in a new path OUTSIDE all original R4p/R4o/R4k/R4d/R4a roots, writing only `report.json` plus SHA manifest. No original files or folders created/deleted/changed; no GitHub Actions.

## Workstation verification and original read-only inspection — NO vendor processes

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q \
  tests/test_cst_r4s_original_timing_drc_context.py \
  tests/test_cst_r4r_original_routed_tradeoff_ledger.py \
  tests/test_cst_r4q_original_r4p_physical_diagnostics.py

python -m fpgai_research.experiments.cst_r4s_original_timing_drc_context \
  --study build/r4p_matched_vivado_ooc_native_20261106_01 \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --output build/r4s_original_timing_drc_context_20261106_01
~~~

Never overwrite an existing destination; omit `--output` for source-only stdout if the folder already exists. R4s output may have no path witness if the existing summary lacks path detail. That is a VALID negative completeness result, not a reason to retroactively rewrite or rerun the original R4p native experiments.

**First paper remains MAJOR REVISION / NOT SUBMISSION READY**. A next physical experiment or separate checkpoint-based report generation must be preregistered with separate source receipts and a new output root. Do not use any new vendor run to overwrite original source/physical evidence.

## Original R4s workstation verification — 6 November 2026

The author pulled research revision `ee5caea95022886ff9bdac234531fa44efd24385`, ran **28/28 focused R4s/R4r/R4q tests** (0.29s), and completed the original create-only read-only audit under `build/r4s_original_timing_drc_context_20261106_01`. The report returned `SIGNED_ORIGINAL_R4P_TIMING_DRC_TEXT_FORENSICS_NOT_E4`, `source_attested_arms=2`, `no_new_vivado_or_vitis_calls=true`, `original_vendor_artifacts_modified=false` and `physical_E4=null`. Original source revision `776e3a7a4b599fd114718f6a03ab73e06be17f5c`.

**Timing summary coverage:** each original `timing_summary.rpt` contained **one** source/Startpoint header but **zero complete startpoint–endpoint–slack witnesses** under the predeclared bounded lexical grammar: `UNKNOWN_NO_COMPLETE_PATH_DETAILS_IN_ORIGINAL_TIMING_SUMMARY` for BOTH arms. Critical-path start/end, specific cell/net causes, mapped memory-port schedule and routed WNS degradation root cause remain UNKNOWN. The actual routed WNS values are unchanged: **−2.170ns** full and **−3.718ns** row (row − full **−1.548ns**). This is a source-detail completeness limit, **not evidence that no timing-critical path exists**.

**Signed DRC report context:** both arms gave original source text line examples and exact warning categories, but no causal classification. The original summary labels are `DPIP-2 | Warning | Input pipelining | 29`, `DPOP-3 | Warning | PREG Output pipelining | 20`, `DPOP-4 | Warning | MREG Output pipelining | 24` for full and `21` for row, `RTSTAT-10 | Warning | No routable loads | 1`. DRC total full **74**, row **71**, zero critical warnings. These rule description strings are from the original SHA-attested reports and are not necessarily explanations for the WNS difference.

**Next stage:** prepare an **independent new study** to generate an explicit `report_timing` against a COPIED original routed DCP checkpoint, rather than running Vivado implementation again. The original R4p vendor invocation receipt attests the original source/scripts and reports but did **not** include a routed DCP SHA at run completion; therefore a checkpoint found afterward must be described as a **post-hoc source-location and content SHA-attested artifact**, not a contemporaneously signed original output. Separate vendor report-only execution, if later performed, requires new script and result receipts and MUST NOT overwrite anything in the original native study.
