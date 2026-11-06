# R4o source-pinned, schedule-matched Vivado OOC preparation — 6 November 2026

**Status:** R4n original workstation evidence **VERIFIED**; R4o code/tests/physical package preparation **COMMITTED, NOT YET WORKSTATION-TESTED**. R4o does not run Vivado. Do not reexecute the already completed original R4l Vitis 2023.2 arms. Keep all R4a/R4d/R4k/R4l source+native results immutable.

## Verified R4n source result motivating physical study

The user's 27/27 R4n/R4m/R4l focused tests passed and the author ran a create-only source-RTL audit at `build/r4n_schedule_matched_RTL_memory_20261106_01`. R4n's original R4l `full_storage` / `row_storage` pair has 24/24 original SHA-attested RTL modules, 3/3 unchanged ROM payloads, 5/5 named reachable memory module types, 8/8 source memory declaration sites, 3/3 matching scratch source sites, and literal 32-bit/8-bit-address scratch depths **214→134** in each site. HLS BRAM18K **13 vs 13**; HLS DSP/LUT/FF/cycles improved at matched SOURCE schedule. Physical mapped BRAM and routed Fmax **UNKNOWN**.

## R4o question

When original R4l Vitis RTL+ROM inputs are instantiated as **separately routed** Vivado out-of-context (OOC) implementations at the **same target part and constraints**, does row-retained source storage change actually mapped RAMB utilization, LUT/FF/DSP, or worst negative slack (WNS)? What DRC and routing caveats remain?

R4o currently PREPARES, not executes, the two physical cases. The prepared Tcl is intentionally qualified as an engineering draft until a fresh CPU-only regression and explicit native readiness review. It does not assert clock reachability, board IO constraints, clean-DRC, or routable design.

## Source lineage and engineering dependencies

- `src/fpgai_research/experiments/cst_r4o_matched_vivado_ooc_prepare.py`: source-only CLI. Calls `r4n.inspect`, which uses the original `r4l.audit_native`, `r4g.inspect_arm`, SHA-attested Verilog/ROM inventories, and the existing safe static source parsers. It re-verifies `inventory_vitis_rtl` and byte SHA-256 for every RTL and ROM file immediately before copying each source file into a NEW OOC case.
- `src/fpgai_research/toolchains/vivado/rtl_inputs.py`: shared conservative Verilog `.v/.sv`, ROM `.dat/.mem`, literal readmem references, original ROM integrity, and bad IP format rejection. R4o does not modify this canonical library.
- `src/fpgai_research/toolchains/vivado/report.py`: existing parser for actual complete routing, timing summary, utilization, DRC. Use only in a later source/receipt-attested physical gate; report files on their own are insufficient.
- `src/fpgai_research/toolchains/vivado/runner.py`: existing opt-in Vivado vendor runner. R4o **does not call** it. A later native execution gate must separately verify staged SHA manifests and reject partial/previously-run cases before launching vendor tools.
- `tests/test_cst_r4o_matched_vivado_ooc_prepare.py`: new source-only synthetic tests for both-arm staging, exact RTL/ROM SHA and testbench lineage, Tcl part/clock/uncertainty/report requirements, local ROM initialization working directory, create-only outputs, mutation/symlink/missing-E3 rejection, and provenance preservation.
- `paper/compiler_ir/R4O_MATCHED_VIVADO_OOC_PREPARE_20261106.md`: this document, limits and proposed measurement protocol.

## Proposed study parameters (same for both arms)

| Parameter | Original signed R4l source | R4o physical preparation |
|---|---|---|
| Part | `xck26-sfvc784-2LV-c` | identical |
| Top | `fpgai_model` | identical |
| Input C++ precision | 32 total / 6 integer bits | unchanged original generated RTL |
| Requested clock | 12.00 ns | 12.00 ns on `ap_clk` |
| Uncertainty | 1.50 ns | 1.50 ns |
| RTL and ROM | 24 signed source files + 3 signed ROM files per arm | staged copies with matching SHA |
| HLS E2/E3 | both already qualified | native OOC preparation only |
| Board-level IO | none in original OOC | not constrained, NOT deployable |

Each case receives its own `rtl/` subdirectory with the preserved generated Verilog and original ROM payloads, and its own `run_vivado.tcl`. The script deliberately uses the original ROM directory as its working directory during synthesis, so `$readmemh` local filenames are not silently redirected. Tcl requests OOC synthesis, `opt_design`, `place_design`, `route_design`, `report_route_status`, `report_timing_summary`, `report_utilization`, `report_drc`, and a routed checkpoint. The parent experiment has an immutable source-only `summary.json` and `artifact_sha256.json` covering staged inputs and Tcl scripts. Actual native output files, if run later, must be recorded separately and preserve the original staging inventory.

## Strict qualification rules for any subsequent Vivado run

1. A prepared Tcl is NOT physical evidence; validate source scope/ROM paths/clock and part against all signed original R4l receipts first.
2. New OOC physical runs must be opt-in and create-only, with separate source-inventory and tool-invocation receipts, no Vitis reruns, no overwrites or silent retries. A failed routing or timing case must remain recorded as FAIL/UNKNOWN.
3. The later physical audit must verify original RTL and ROM checksums, Vivado executable identity, completed routing (routable/fully routed counts and zero errors), real source-attributed timing and utilization files, and full DRC classification. HLS BRAM estimates MUST NOT be substituted for routed RAMB/BRAM tile counts.
4. Distinguish strict clean-DRC results, warning-only routed OOC results, and incomplete/failed routing; report all, not just favorable cases. Do not infer a deployable KV260 bitstream, verified board GPIO, real achieved frequency, or measured power from an OOC experiment.
5. One 32/6, two-reader mixed fanout graph is not enough for universal compiler claims. Additional graph/fanout/precision studies should be independently preregistered after the first physical pilot.

## Safe workstation commands: SOURCE-ONLY, no Vivado calls

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pytest -q \
  tests/test_cst_r4o_matched_vivado_ooc_prepare.py \
  tests/test_cst_r4n_schedule_matched_rtl_memory.py \
  tests/test_cst_r4m_three_point_source_contrast.py

python -m fpgai_research.experiments.cst_r4o_matched_vivado_ooc_prepare \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --output build/r4o_matched_vivado_ooc_prepare_20261106_01
~~~

The last CLI creates new source-only OOC case files with zero new native tool calls. If the destination exists, omit `--output` for a read-only preflight. **DO NOT run the prepared Tcl through Vivado until a separately reviewed guarded execution gate exists**, and never run either original R4l Vitis command again.

**First paper stays MAJOR REVISION / NOT SUBMISSION READY.**
