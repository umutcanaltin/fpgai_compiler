# R4p — original R4l RTL/ROM, isolated Vivado OOC execution and strict routed-result audit

**6 November 2026. Status:** R4o source-only preparation **VERIFIED ON WORKSTATION**: 26/26 focused R4o/R4n/R4m tests passed and the author successfully created `build/r4o_matched_vivado_ooc_prepare_20261106_01`, 24 SHA-verified Vitis Verilog + 3 SHA-verified ROM files per arm, with original R4l source revision `776e3a7a4b599fd114718f6a03ab73e06be17f5c`, E2/E3 already qualified. The R4o result explicitly reports `SOURCE_ONLY_VIVADO_OOC_PREPARED_NOT_EXECUTED`, `new_vivado_calls = 0`, `physical_E4 = null`, and board qualification false.

**R4p is newly IMPLEMENTED but not yet workstation-tested or run through Vivado.** All initial R4p tests use MOCK vendor receipts and SYNTHETIC routed report text: their numbers are **not** new physical FPGA measurements.

## Question and matched physical design

With the 32/6 two-reader fanout graph, row-local original source schedule, the same XCK26 `xck26-sfvc784-2LV-c` part, requested 12.0ns clock on `ap_clk` and 1.5ns uncertainty, do two independently routed OOC Vivado implementations of the two *different but E2/E3-qualified* R4l-generated Verilog designs show differences in actual mapped BRAM, LUT, FF, DSP, or routed WNS? Is each case clean DRC or only warning/error-bearing OOC evidence?

| Source-fixed study arm | HLS scratch RAM source depth/site | HLS estimated BRAM18K | Physical routed BRAM |
|---|---:|---:|---|
| Full storage, row-local source schedule | 214×32, three declaration sites | 13 | UNKNOWN |
| Row storage, same row-local source schedule | 134×32, three declaration sites | 13 | UNKNOWN |

These three SOURCE sites are not proven physical RAMB instances. R4n's source memory totals must never be substituted for a Vivado utilization row. Neither HLS-estimated 13 BRAM18K nor a negative/positive routed WNS establishes board-level achieved Fmax or measured power.

## New files and cross-repository connectivity

- `src/fpgai_research/experiments/cst_r4p_matched_vivado_ooc_native.py`: new three-stage CLI: original SHA-gated R4o audit + create-only R4p run-package staging; default vendor-free dry run; explicitly confirmed one-time native Vivado invocation; and separate read-only physical DRC/timing/utilization audit. **Never executes original R4o or original R4l folders**.
- `tests/test_cst_r4p_matched_vivado_ooc_native.py`: synthetic original evidence and mocked vendor receipts test source lineage/tamper, copied RTL/ROM, failed runs and partial retry refusal, two confirmations, route completion, WNS, resource availability, DRC error and report freshness.
- `paper/compiler_ir/R4P_MATCHED_VIVADO_OOC_NATIVE_GATE_20261106.md`: this protocol, experimental limits and source dependency record.
- Existing `src/fpgai_research/toolchains/vivado/runner.py`, `src/fpgai_research/toolchains/runner.py`, `src/fpgai_research/toolchains/vivado/rtl_inputs.py`, and `src/fpgai_research/toolchains/vivado/report.py` are **REUSED WITHOUT EDITS**. The generic runner is not compatible with R4o's single `rtl/` layout: it expects the original `case/fpgai_hls_project/solution1/syn/verilog` source copy and ROMs in its `case/vivado` run cwd. R4p stages NEW exact byte-identical copies into that shape rather than silently weakening the generic runner.
- `src/fpgai_research/experiments/cst_r4o_matched_vivado_ooc_prepare.py`: original signed, immutable R4o source stage; R4p rejects any different summary, source inventory or script.
- `src/fpgai_research/experiments/cst_r4n_schedule_matched_rtl_memory.py`: real original R4l E2/E3, RTL source and ROM SHA provenance checked through the R4o preflight.

## R4p create-only staging and ROM integrity

For each arm R4p creates a NEW `<R4p-study>/<arm>/` case containing:

- `fpgai_hls_project/solution1/syn/verilog/`: exact original 24 RTL and 3 ROM files to be SHA-attested by the shared native tool runner.
- `vivado/rtl/`: exact original 24 RTL and 3 ROM files, used as Vivado Tcl's `cd` and actual `read_verilog` / `$readmem*` location.
- `vivado/`: three additional byte-identical ROM payloads (runner's checked local ROM assets) and the ORIGINAL unchanged R4o `run_vivado.tcl`.
- A NEW `summary.json` and full `artifact_sha256.json` covering ALL source-only copies. Vendor may write logs, DRC reports, checkpoints, journal and invocation receipt only inside the NEW run case. All original source inventory hashes must still match after execution.

Before ANY native attempt, the runner audits original R4a/R4d/R4k/R4l source lineage, R4n and R4o original SHA receipts, both complete staged R4p inputs, top/clock/part, and the original Tcl SHA. The native CLI requires **BOTH** `--execute-native` and `--confirm-ooc-native`. It checks BOTH arms for an existing invocation, log folder, report, checkpoint, journal, or failed partial attempt before starting the first. On failure of an arm, it stops rather than automatically attempting the next arm; existing attempts are NEVER overwritten or restarted.

## R4p qualification tiers

1. **No native attempt:** source-only R4p staging is NOT E4 and all physical metrics null.
2. **Native process complete:** Vivado exit 0 and source/ROM SHA receipts are necessary, not sufficient for E4. The generic runner additionally audits input hash stability and staged ROM integrity. R4p must also reverify the *Tcl-consumed* ROM+RTL copies and all source-input locations.
3. **Routed physical observation:** all four original reports (`route_status.rpt`, `timing_summary.rpt`, `utilization.rpt`, `drc.rpt`) must have independently fresh SHA hashes matching the vendor receipt. The existing report parser must verify completed routing (no routing errors/unrouted nets), finite nonnegative LUT/FF/DSP/BRAM values, finite WNS and met timing, and an explicitly summarized DRC report. Otherwise the source-attributed physical result is UNKNOWN/FAILED.
4. **Non-clean DRC routed observation:** a real routed observation with violations may be reported descriptively but is **not strict clean-DRC E4**. Errors/critical warnings must not be mislabeled 'warnings only'.
5. **Strict clean-DRC routed OOC:** all above plus verified `drc.clean=true` and no violations; a strict pair difference is numeric only if BOTH original cases independently pass. WNS is worst negative slack under the requested OOC clock, not physical board Fmax.

Both-arm physical deltas are `row_storage − full_storage`; the HLS BRAM 13/13 tie is kept separate. A DRC nonclean case is included in the descriptive population, not silently dropped. A missing or failed native case yields NULL differences rather than a fictional zero. No new Vitis calls, no unverified board pin mapping, no bitstream, no deployed KV260 throughput or power results.

## Workstation verification: initial steps have **NO Vivado calls**

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pytest -q \
  tests/test_cst_r4p_matched_vivado_ooc_native.py \
  tests/test_cst_r4o_matched_vivado_ooc_prepare.py \
  tests/test_cst_r4n_schedule_matched_rtl_memory.py

python -m fpgai_research.experiments.cst_r4p_matched_vivado_ooc_native \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --prepare-output build/r4p_matched_vivado_ooc_native_20261106_01

python -m fpgai_research.experiments.cst_r4p_matched_vivado_ooc_native \
  --study build/r4p_matched_vivado_ooc_native_20261106_01 \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01
~~~

Do not rerun an existing output directory. The above step must report dry-run with exactly two planned NEW Vivado calls and zero actual calls. Keep the R4p and R4o original SHA inventories and R4l source folders unchanged.

## Later one-time native execution, ONLY after the CPU tests and dry-run pass

~~~bash
python -m fpgai_research.experiments.cst_r4p_matched_vivado_ooc_native \
  --study build/r4p_matched_vivado_ooc_native_20261106_01 \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --vivado /tools/Xilinx/Vivado/2023.2/bin/vivado \
  --execute-native --confirm-ooc-native
~~~

**This command executes native Vivado twice only for the NEW isolated R4p cases** and may take significant workstation compute resources. Do NOT call it on R4o/Vitis originals, and never invoke it twice for the same case. If an attempt fails, preserve the partial logs and run the read-only audit instead.

## After any native attempt: read-only qualification

~~~bash
python -m fpgai_research.experiments.cst_r4p_matched_vivado_ooc_native \
  --study build/r4p_matched_vivado_ooc_native_20261106_01 \
  --r4o-study build/r4o_matched_vivado_ooc_prepare_20261106_01 \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --audit-native
~~~

The output might be strict-clean E4, routed with nonclean DRC, or UNKNOWN depending on real Vivado evidence. No result should be claimed until that workstation audit has completed. **Paper remains MAJOR REVISION / NOT SUBMISSION READY; PR remains draft and no GitHub Actions are used.**

## Additional R4p race/replay and vendor receipt controls

**Original native attempt marker:** before the first Vivado call, R4p now atomically creates a permanent `r4p_native_attempt.json` file using create-exclusive semantics, recording the source revision, the two original arm names and a no-retry marker. It refuses native execution whenever this marker, any original invocation receipt, log directory, output/report directory, or journal already exists. It preserves this marker on tool failure, timeout or interrupted process. Re-run is NEVER automatic; a new, separately named study is required if a fresh experiment is scientifically justified. The read-only physical audit refuses to classify apparent native receipts without the original attempt marker.

**Exact tool source/linkage:** read-only R4p audit verifies each original invocation's `argv`, resolved executable path, working directory, original Tcl SHA, immutable pre/post Tcl input hashes, start/duration witness, and exact original stdout/stderr SHA files, in addition to the generic Vivado receipt's source RTL/ROM and staged ROM integrity flags. It checks all fresh route/timing/utilization/DRC report hashes against the native receipt and refuses unsigned or stale files. The synthetic tests explicitly mutate cwd, argv, Tcl input hash, logs and the actual Tcl-consumed ROM while preserving old-looking receipts; such cases must not become E4.

**Updated DRC terminology:** `ROUTED_NONCLEAN_DRC_PAIR` includes warning-, critical-warning- and error-bearing routed observations and is never strict clean-DRC E4. `drc_warning_only_routed_observation` is true only for a non-clean DRC with zero critical-warning/error counts. Strict clean DRC requires zero verified violations, separately from route completion, positive or nonnegative timing slack, and resource completeness.

## First workstation regression triage (6 November 2026)

The author's first R4p/R4o/R4n targeted workstation regression completed with **36 passing, 1 failing** test, against branch revision `6efc7859285b1226ca59051023d7bf25d5cb4eb2`. The only failed test was `test_native_executes_only_separate_cases_and_stops_on_first_failure`. It deliberately mocked a failed first-arm native invocation, verified that only one arm was attempted and the permanent `r4p_native_attempt.json` lock was created, then attempted a prohibited retry. **The production runner correctly raised `FileExistsError`**, with message `R4p original native attempt lock exists; do not rerun`. The test incorrectly expected the later per-arm guard message `already attempted`. This is a **test assertion mismatch**, not failed isolation or an actual Vivado invocation.

The exact failing assertion was corrected to expect `original native attempt lock exists` (production native safety code unchanged). The independent test of a preexisting partial per-arm attempt retains its `already attempted` assertion. **The corrected tests have NOT yet been rerun on the author's workstation; do not report 37/37 until verified.** The R4p source-only create-only stage and any real native run are still pending. Original R4o and R4l results remain immutable, physical E4 remains UNKNOWN.

## Verified R4p workstation source-only preparation (6 November 2026)

The author pulled the corrected R4p branch revision `e94150df3e237aaa04966b0deb7bb449ad3a2ca4` and reran the complete targeted R4p/R4o/R4n suite: **37/37 tests passed in 0.38 s**. This supersedes the preceding 36-pass/1-fail initial regression, which had an incorrect expectation for the permanent attempt-lock exception. The production native safeguards remained unchanged in that correction.

The author then executed R4p `--prepare-output` **without any vendor execution**. The new immutable source-only execution study is `build/r4p_matched_vivado_ooc_native_20261106_01`, with `status=R4P_NATIVE_CASES_PREPARED_NOT_EXECUTED`, `native_vendor_calls=0`, `paired_native_arms=2`, original R4k source revision `776e3a7a4b599fd114718f6a03ab73e06be17f5c`, unchanged `xck26-sfvc784-2LV-c` part, 12.0ns requested clock and 1.5ns uncertainty. `board_qualified=false`, `physical_E4=null`.

**Next gate remains vendor-FREE:** run R4p without `--execute-native` or `--audit-native`, passing this new `--study` and the same frozen R4o/R4k/R4d/R4a source locations. Require `status=DRY_RUN_NO_VENDOR_CALLS`, `planned_vivado_calls=2`, `actual_vivado_calls=0` and exact SHA source gate `ALL_R4L_R4O_R4P_SOURCE_BYTES_SHA_ATTESTED`. The user has not yet supplied a result from this dry-run or any native Vivado invocation. Do NOT label the native experiment as successful before actual source-pinned routing/timing/DRC reports exist.

Explicit one-time Vivado execution may be considered only after this dry-run gate and a final input/script inspection; no original Vitis or original R4o sources may be modified. The permanent attempt lock is created by the first native execution attempt, **not** by source-only preparation or dry-run. In case of vendor failure, preserve the attempt lock and original partial outputs and conduct a read-only audit.

## Original R4p read-only preflight VERIFIED — 6 November 2026

The author pulled the research branch at `9ea35ca7232ace3a5009527ddd65de8f24e86201` and ran the **default vendor-free R4p dry-run** on the already prepared source-only study `build/r4p_matched_vivado_ooc_native_20261106_01`. The original stdout reported:

- `status=DRY_RUN_NO_VENDOR_CALLS`, `source_gate=ALL_R4L_R4O_R4P_SOURCE_BYTES_SHA_ATTESTED`, `source_Vivado_inputs_sha_attested=true`.
- `both_E2_E3_qualified_in_frozen_R4l=true`; `source_revision=776e3a7a4b599fd114718f6a03ab73e06be17f5c`.
- `planned_vivado_calls=2`, `actual_vivado_calls=0`, `new_vendor_calls_during_audit=0`.
- `physical_E4=null`, `board_qualified=false`.

This confirms all source/RAM/Tcl artifact chains requalified at read-only preflight. It does **NOT** show any Vivado synthesis, route, real mapped utilization, routed slack, clean DRC, or physical E4 outcome. Both original R4l Vitis cases and all R4o/R4p source-only assets remain immutable.

**Next valid action:** a separately explicit **single native execution attempt** using only `--study build/r4p_matched_vivado_ooc_native_20261106_01`, original read-only evidence roots, `--vivado /tools/Xilinx/Vivado/2023.2/bin/vivado`, and BOTH `--execute-native --confirm-ooc-native`. This permanently creates the source-revision-pinned native attempt lock before launching the first arm. The runner stops on a failed first arm and must never be restarted on this study; preserve failures as evidence. After any native attempt, run only the default `--audit-native` mode (no execution flags) to qualify real routed/DRC evidence. The dry-run itself consumed no one-time native attempt.
