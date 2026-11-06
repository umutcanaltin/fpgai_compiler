# R4g — read-only 32/6 HLS RTL memory structure bridge

**Status (6 November 2026): code and source-only tests IMPLEMENTED on the existing research branch, NOT yet executed on the author's workstation.** No new native Vitis, Vivado or FPGA measurements are claimed.

## Starting point: original observed R4e/R4f data

- Original R4f workstation **6/6 tests passed**; R4e source-attested read-only audit confirmed **one 32/6 matched E2/E3 qualified pair** among 3 predeclared formats. R4f produced a separate create-only JSON report under build/r4f_paired_hls_report_20261106_01.
- At 32/6, two original Vitis HLS 2023.2 arms: BRAM18K = **13 full H, 13 row H** (no change); DSP 23 versus 19; LUT 4990 versus 4755; FF 1798 versus 1647; cycles 22192 versus 22602; estimated HLS period 10.302 ns versus 9.500 ns, both within 10.5 ns effective criterion.
- The earlier R4a source-level H buffer shrank by 240 scalar entries, but no HLS-estimated BRAM saving was measured. Source loop ordering changes as part of the intervention, so fewer logic resources cannot be attributed solely to fewer H array cells.
- The NEW 16/6 and 32/8 native arms are unexecuted; earlier 16/6 R4b E2 failure is a distinct prior experiment. Both remain in their correct denominators without invented measurements.

## R4g scientific question

What *static, original-source Verilog evidence* accompanies the zero BRAM estimate difference, and does the row-local C++ source change the identifiable structure of memory-related generated RTL modules?

New module: src/fpgai_research/experiments/cst_r4g_rtl_memory_bridge.py
New test suite: tests/test_cst_r4g_rtl_memory_bridge.py

R4g consumes R4e's full read-only source/CSim/HLS audit; uses the canonical Vitis RTL/ROM inventory; and checks original generated Verilog and ROM SHA maps against each original vendor invocation receipt. It then reuses the pre-existing **R2x static RTL type-graph scanner**, rather than duplicating a new Verilog parser.

It records per-arm original HDL hashes, source file counts, the scanner status, provisionally reachable scratch-/memory-named module types, candidate primitive instantiations and reg-array candidate declarations; if both matched arms are structurally interpretable, it reports their paired differences. It separately reports the qualified HLS-estimated BRAM difference. The original SHA-attested file set is never edited and no vendor tool is launched.

### Evidential limits that must appear in the paper

1. R2x is a bounded *textual source parser*: it is **not** a Verilog preprocessor, elaborator, synthesized netlist or physical primitive/cell inventory.
2. A module type called scratch/RAM/ROM, or even a RAMB identifier in source, is **not** proof of an actual placed FPGA RAMB. Static module-type reachability does not account for full parameterization, generate conditions, hierarchy optimization or resource mapping.
3. If HLS was not executed, RTL is absent, the native receipt lacks matching RTL SHA, or the parser cannot resolve the source, counts are **UNKNOWN (null)**, never inferred as zero.
4. Any changed original RTL/ROM hashes relative to its native invocation receipt are **rejected**; no comparison based on tampered artifacts.
5. This analysis can contextualize *how generated source structure differs*, not establish *why BRAM18K tied* or how FPGA BRAM/logic will appear after Vivado place-and-route.
6. A true memory-mapping explanation needs a separate more targeted architectural experiment and, for physical claims, independently qualified Vivado routed evidence.

## Next safe workstation checks

Run from the EXISTING research branch:

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q \
  tests/test_cst_r4g_rtl_memory_bridge.py \
  tests/test_cst_r4f_paired_hls_review.py \
  tests/test_cst_r2x_rtl_hierarchy_analysis.py

python -m fpgai_research.experiments.cst_r4g_rtl_memory_bridge \
  --study build/r4d_numeric_format_source_20261106_01 \
  --original-study build/r4a_row_retained_paper_20261106_01
~~~

Optional **separate create-only** export (not under original R4a/R4d roots):

~~~bash
python -m fpgai_research.experiments.cst_r4g_rtl_memory_bridge \
  --study build/r4d_numeric_format_source_20261106_01 \
  --original-study build/r4a_row_retained_paper_20261106_01 \
  --output build/r4g_static_rtl_memory_bridge_20261106_01
~~~

The CLI audit command makes zero Vitis/Vivado calls. An existing output path is deliberately rejected; no existing R4e cases should be reexecuted.

The current manuscript is **MAJOR REVISION / NOT SUBMISSION READY**. Its central scientific proof is architectural semantics and mapping contracts; resource benefits are heterogeneous observations with explicit negative results, not demonstrated general memory superiority.
