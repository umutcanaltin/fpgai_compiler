# R4t — isolated routed-checkpoint detailed path report preparation

**6 November 2026. Original R4s WORKSTATION VERIFIED:** 28/28 focused R4s/R4r/R4q tests passed (0.29s) and the author completed the original read-only study `build/r4s_original_timing_drc_context_20261106_01`. Each original source-attested timing summary has **one** source/Startpoint-type header but **zero complete startpoint–endpoint–slack witnesses**, so the exact original timing-path root cause is UNKNOWN. Both signed DRC reports provide original warning category names: `DPIP-2` Input pipelining 29/29; `DPOP-3` PREG Output pipelining 20/20; `DPOP-4` MREG Output pipelining 24/21; `RTSTAT-10` No routable loads 1/1. These labels alone cannot establish resource or WNS causes.

**R4t NEW, IMPLEMENTED but NOT YET WORKSTATION-TESTED. R4t does not call Vivado.** It validates original R4s/R4r/R4q/R4p report/source SHA chains then creates a NEW, separate, byte-identical two-arm snapshot of the original post-route Vivado `.dcp` design checkpoints, plus **inert report-only Tcl scripts**. The eventual checkpoint-opening Vivado run is deliberately NOT implemented or executed in this stage.

## Research question

Can we recover exact path startpoint, endpoint, slack, logic and net segments from the PREVIOUSLY ROUTED original designs, without repeating synthesis/place/route or altering the original vendor result, so we can test hypotheses about the row-retained architecture's BRAM/DSP versus LUT/FF/WNS tradeoff?

Original verified OOC mapping in the same 32/6 graph: BRAM tiles full/row **7/4**, DSP **24/21**, LUT **1282/1710**, FF **883/1022**, WNS at requested 12.0ns **−2.170/−3.718ns**, and 74/71 DRC warnings. Both fail requested timing and clean-DRC: strict physical E4 **NULL**, board qualification false.

## Provenance distinction: original signed reports vs POST-HOC DCP

The original R4p Vivado invocation receipt and subsequent R4q/R4r/R4s audits SHA-attest the exact **original generated RTL/ROM, invocation script/logs, and route/timing/utilization/DRC reports**. The original vendor invocation receipt does **NOT** contain a contemporaneous SHA-256 digest of the generated `outputs/routed.dcp`. R4t therefore finds the checkpoint at its expected **original R4p native run output location**, verifies that the original source and reports remain qualified, and separately computes a **POST-HOC checkpoint content SHA-256 and size**. This is source-location lineage and content pinning, *not* proof the current DCP bytes are unchanged since the vendor invocation. R4t copies those exact post-hoc SHA-pinned bytes into a new independent output root and rehashes the copy. It never modifies the original checkpoint.

If an original checkpoint is missing, symlinked, empty, unexpectedly large, or changes while copying, R4t rejects its staging. The experiment must not reconstruct or overwrite a missing original checkpoint or claim a new E4 result. A later report-only vendor execution needs its own separate invocation/hash/lineage receipt, original-DCP SHA pin, output SHA inventory and independent audit.

## File, dependency and engineering change inventory

| File | Role and dependencies |
|---|---|
| `src/fpgai_research/experiments/cst_r4t_routed_checkpoint_report_only_prepare.py` | NEW. Calls existing `cst_r4s_original_timing_drc_context.inspect`, which reuses original `cst_r4q_original_r4p_physical_diagnostics.inspect`, `cst_r4r_original_routed_tradeoff_ledger.summarize` and the source/RTL/ROM/vendor/report SHA lineage. Streams SHA256 of each original `routed.dcp`; creates new copy + Tcl for each arm only with explicit output; never calls Vivado. |
| `tests/test_cst_r4t_routed_checkpoint_report_only_prepare.py` | NEW. Seven synthetic/no-vendor tests for original provenance, DCP SHA and size, separate identical byte copies, local-only Tcl, create-only output, absent/symlinked/empty checkpoint and population drift. |
| `paper/compiler_ir/R4T_ROUTED_CHECKPOINT_REPORT_ONLY_PREPARE_20261106.md` | NEW. This protocol, source mapping, expected data, limits and safe commands. |
| `paper/compiler_ir/R4S_ORIGINAL_TIMING_PATH_DRC_CONTEXT_20261106.md` | VERIFIED original 28-pass source-only path-detail coverage and DRC labels. Unchanged except for original observation update. |
| `paper/compiler_ir/FIRST_PAPER_SCIENTIFIC_COMPLETION_20261106.md` | Updated scientific gate. Original R4p OOC timing failure and nonclean DRC preserved. |

## Proposed isolated report-only Tcl (not executed by R4t)

Every new R4t arm includes `input/routed.dcp` copied from the original R4p `vivado/outputs/routed.dcp` and a separate `report_only.tcl`. The script only performs `open_checkpoint` of the COPIED DCP, explicit `report_timing -delay_type max -path_type full -input_pins -max_paths 20 -nworst 10`, `report_timing_summary`, `report_drc`, and `close_design`, writing results into its NEW sibling `reports/` directory. Tcl is self-relative and includes no synth, optimize, place, route, checkpoint rewrite or bitstream generation. The report_timing option spelling is documented by the AMD Vivado Tcl guide: https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_timing .

**Reporting later is NOT automatically E4:** the later report-only outputs must prove the copied DCP SHA is unchanged and that the reopened design retains the original part/clock semantics. Detailed path counts/identities, DRC rule text, report hashes, native tool invocation identity and any WNS mismatch must be preserved and reported. Critical-path root-cause analysis remains an inference to be tested, not a fact from original R4s. No new implementation pass is allowed on the original study.

## Next workstation gate — SOURCE-ONLY; ZERO VENDOR CALLS

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pytest -q \
  tests/test_cst_r4t_routed_checkpoint_report_only_prepare.py \
  tests/test_cst_r4s_original_timing_drc_context.py \
  tests/test_cst_r4r_original_routed_tradeoff_ledger.py

python -m fpgai_research.experiments.cst_r4t_routed_checkpoint_report_only_prepare \
  --study build/r4p_matched_vivado_ooc_native_20261106_01 \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --output build/r4t_routed_checkpoint_report_only_prepare_20261106_01
~~~

The output is create-only. If it already exists, omit `--output` to run **read-only source preflight**. NEVER delete or overwrite any original R4p/R4o/R4k/R4d/R4a artifact, routed report or vendor attempt lock. **Do not run `report_only.tcl` yet**: the new report-only native execution/attestation gate must first be implemented and reviewed separately. No GitHub Actions or unrelated compiler feature edits.

**First paper: MAJOR REVISION / NOT SUBMISSION READY.** The documented descriptive BRAM savings do not establish timing-qualified or deployable FPGA superiority.
