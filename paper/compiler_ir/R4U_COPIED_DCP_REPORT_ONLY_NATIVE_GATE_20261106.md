# R4u — source-pinned routed DCP, isolated one-time Vivado report-only native and read-only audit

**6 November 2026 — R4t original AUTHOR VERIFIED, 28/28 tests passed**, and source-only checkpoint copies prepared at `build/r4t_routed_checkpoint_report_only_prepare_20261106_01` with `R4T_ISOLATED_CHECKPOINT_REPORT_ONLY_SOURCE_PREPARED`; zero new vendor tool calls and `strict_physical_E4=null`. Both original R4p Vivado implementation attempts MUST NEVER be rerun.

**R4u code/tests committed, NOT YET TESTED on the author's workstation. No R4u native reports exist.** The script is a separate, one-time, REPORT-ONLY Vivado invocation on NEW copies of routed checkpoints, never implementation. Its source-only staging/dry-run should pass before opt-in native execution.

## Exact original R4t source-pinned post-hoc checkpoint identity

| Original R4p arm | SHA-256 of original `routed.dcp`, pinned after original run | Bytes | Original routed WNS |
|---|---|---:|---:|
| `full_storage` | `dec6616bd7d01d582ed59445122b26f80eaafe23fd00c2e8082c3dd441936137` | 3,379,985 | −2.170ns |
| `row_storage` | `1cf5feea9bd4903d51ded333a640cdeb4bd0997c257dc3ceea2514b3c7ddb486` | 3,430,541 | −3.718ns |

R4u **hard pins these exact observed SHA/size pairs**. Original R4p execution SHA-signed RTL, ROM, scripts and four routed report files **but not DCP outputs**. Therefore the original DCP hashes are only **POST-HOC source-location and content pins**, not contemporaneous execution-time vendor attestation. R4u MUST preserve this distinction and cannot retroactively upgrade original physical E4.

## Research question

Can Vivado 2023.2 `open_checkpoint` on **copied original routed checkpoints**, followed only by `report_timing -path_type full -input_pins -max_paths 20 -nworst 10`, `report_timing_summary` and `report_drc`, return original startpoint→endpoint→slack paths? Do new WNS values and DRC counts match the SHA-attested original R4p results? If available, those paths can inform later falsifiable hypotheses about why BRAM drops 7→4 while LUT/FF increase and WNS deteriorates −2.170→−3.718ns. Nothing in this study itself proves timing qualified E4, board performance, actual Fmax or a causal explanation.

Vivado report-only command options: AMD Vivado UG835 [report_timing](https://docs.amd.com/r/2024.1-English/ug835-vivado-tcl-commands/report_timing), documented `-path_type full`, `-input_pins`, `-max_paths` and `-nworst`. Syntax verified for related UG835 versions; actual 2023.2 checkpoint opening and output semantics still require the author's native read-only pilot.

## New/unchanged files and dependency edges

| Path | Purpose and connectivity |
|---|---|
| `src/fpgai_research/experiments/cst_r4u_checkpoint_report_only_native.py` | NEW. Reuses original `cst_r4t_routed_checkpoint_report_only_prepare.inspect`, which calls the original signed R4s→R4q→R4p and R4r report/RTL/ROM provenance chain. Rechecks original R4t files and the exact author-observed post-hoc DCP hash/size pair. Creates a NEW independent R4u source-only copy, with `summary.json` and SHA manifest. Default dry-run verifies everything without invoking vendor tools. Optional explicitly confirmed native report-only execution is one-time, atomic-lock guarded, preserves stdout/stderr and native receipt. Separate read-only audit verifies all new input/report/log SHA hashes, WNS consistency and DRC counts and extracts explicit path witnesses via R4s parser. |
| `tests/test_cst_r4u_checkpoint_report_only_native.py` | NEW. Synthetic tests for source pin, immutable copies, tampered original or new DCP/script, create-only output, dry-run, opt-in and timeout, partial-case retry guard, durable native lock, signed mock Vivado reports, mismatched WNS/DRC, incomplete detailed paths and forged report SHA. Tests contain synthetic DCPs/paths and make **zero native Vivado calls**. |
| `paper/compiler_ir/R4U_COPIED_DCP_REPORT_ONLY_NATIVE_GATE_20261106.md` | NEW. This protocol and experiment change inventory. |
| `src/fpgai_research/experiments/cst_r4t_routed_checkpoint_report_only_prepare.py` | EXISTING, VERIFIED workstation source-only original, including byte-for-byte DCP copies. UNCHANGED. |
| `src/fpgai_research/experiments/cst_r4s_original_timing_drc_context.py` | EXISTING original timing/DRC lexical parser and original signed report gate. UNCHANGED. |
| `src/fpgai_research/toolchains/vivado/report.py` | EXISTING signed R4p routed report parser. The R4u new report-only audit reuses its numeric WNS and DRC summary parsers. UNCHANGED. |
| `paper/compiler_ir/FIRST_PAPER_SCIENTIFIC_COMPLETION_20261106.md` | Updated with R4t author's test and source-only preparation, R4u untested status. |

## Three distinct stages; do not confuse them

**Stage A — source-only NEW R4u preparation:** R4u refuses any output path inside/or overlapping original R4p/R4t/R4o/R4k/R4d/R4a studies. It independently rehashes original R4t report-only Tcl and both author-pinned post-hoc DCPs, copies them to `build/r4u_copied_dcp_report_only_native_20261106_01/{full_storage,row_storage}/` and writes an immutable source-only SHA manifest. **Zero vendor calls.**

**Stage B — default dry-run:** revalidates the complete original source/ROM/receipt/report chain and exact R4u staged input hashes, scripts and directories. Expected `R4U_DRY_RUN_NO_VENDOR_CALLS`, `R4P_R4S_R4T_R4U_POSTHOC_COPIED_DCP_SHA_ATTESTED`, two planned report-only calls and zero actual. Does not consume the one-time attempt.

**Stage C — separately opt-in one-time report-only native:** requires both `--execute-native` and `--confirm-report-only-native`, validates the actual Vivado executable and timeout, and atomically creates a permanent source-revision-pinned `r4u_native_attempt.json` BEFORE starting any Vivado subprocess. It refuses if the lock, original tool journal, report folder, log folder or invocation receipt already exists for either arm, and stops after any failed/incomplete first case. NEVER rerun either attempt in this study. Running Vivado is strictly limited to copies of existing checkpoints and the unchanged `report_only.tcl` already frozen by R4t. No synthesis, opt, place, route, bitstream or checkpoint rewrite commands. Any native failure must be kept as original negative evidence, not retried.

**Stage D — postrun read-only audit:** requires permanent attempt lock, original R4t SHA input pins, original source/ROM/reports, original script copies and SHA manifest plus a new invocation record proving exact argv/cwd, stable input hashes before/after, zero exit/timeout false, original stdout/stderr SHA logs, and fresh `max_delay_paths.rpt`, `timing_summary.rpt`, `drc.rpt` report SHA values. Reports WNS agreement within 0.005ns of original and exact original DRC rule counts, and extracts bounded explicit startpoint/endpoint/slack paths. Missing, mutated or mismatching report values are FAILURE/UNKNOWN, never converted to zero or qualified E4. Even valid new path witnesses are post-hoc checkpoint report-only forensics, not timing- or clean-DRC-qualified E4, and not a measured real board clock.

## Safe workstation commands — STAGES A AND B ONLY, NO VIVADO

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q \
  tests/test_cst_r4u_checkpoint_report_only_native.py \
  tests/test_cst_r4t_routed_checkpoint_report_only_prepare.py \
  tests/test_cst_r4s_original_timing_drc_context.py

python -m fpgai_research.experiments.cst_r4u_checkpoint_report_only_native \
  --r4t-study build/r4t_routed_checkpoint_report_only_prepare_20261106_01 \
  --study build/r4p_matched_vivado_ooc_native_20261106_01 \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --prepare-output build/r4u_copied_dcp_report_only_native_20261106_01

python -m fpgai_research.experiments.cst_r4u_checkpoint_report_only_native \
  --r4t-study build/r4t_routed_checkpoint_report_only_prepare_20261106_01 \
  --study build/r4p_matched_vivado_ooc_native_20261106_01 \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --native-study build/r4u_copied_dcp_report_only_native_20261106_01
~~~

**Do NOT invoke the `--execute-native` flag until the author's new R4u tests and both safe source gates pass.** R4u original implementation is committed but unverified; no native physical path reports are claimed. A future one-time native invocation and read-only audit will be documented only after those gates pass.

## Scientific status

Original R4p/r4q/r4r descriptive mapped BRAM tiles **7→4** and DSP **24→21**; LUTs **1282→1710**, FF **883→1022**, and WNS **−2.170→−3.718ns**. Both original designs fail the requested 12ns clock and have unclean DRC; R4p strict `physical_E4=null` remains final. Detailed critical path root cause is still UNKNOWN. First paper remains **MAJOR REVISION / NOT SUBMISSION READY**. Draft PR only; `main` unchanged, no GitHub Actions.
