# R4y — 24ns four-arm source-only experiment preparation

**Status: source code and synthetic tests committed on the active research branch, NOT YET run on the author's workstation. No new native experiment has been executed.**

## Exact scientific comparison

The author's previous read-only R4w/R4x evidence confirms routed WNS >= 0 for both original full and row designs at 20ns and 24ns, with 74/71 warning-only DRC findings. Strict clean-DRC matched pairs remain ZERO, and physical or measured latency remains UNKNOWN. R4x selected 24ns under a deliberately post-hoc 1ns minimum pair WNS planning margin, NOT as a proven safe FPGA clock.

R4y proposes a source-level 2x2 factorial at a COMMON requested Vitis HLS period of 24ns, 1.5ns explicit uncertainty, target xck26-sfvc784-2LV-c, precision ap_fixed<32,6>:

| Source storage | MAC-loop policy | Prepared case |
|---|---|---|
| full storage | canonical baseline | full_storage__baseline |
| full storage | producer inner-loop HLS PIPELINE II=1 requested | full_storage__producer_mac_ii1_requested |
| row storage | canonical baseline | row_storage__baseline |
| row storage | producer inner-loop HLS PIPELINE II=1 requested | row_storage__producer_mac_ii1_requested |

The source intervention inserts exactly ONE pragma inside the producer's canonical innermost multiply–accumulate loop; deleting it must recover the exact canonical original C++ bytes. The graph, weights, source event-order fingerprint, full or row scratch addressing, fixed-point format and shared numerical oracle remain unchanged. Four newly packaged Vitis source cases use the SAME tb.cpp hash and run_hls.tcl hash, generated with the compiler's canonical package API. Frozen original R4k source SHA hashes are independently verified against the old R4l audited native input before allocating a NEW output root.

The pragma does NOT prove actual achieved II=1, DSP48 registered stages, reduced DSP warnings, E2 numerical equivalence, HLS timing or FPGA execution performance. Vitis can select a larger II in the presence of loop-carried dependencies, and the baseline may already be automatically pipelined. These are experimental hypotheses, not results. AMD UG1399 2023.2: https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/set_directive_pipeline and https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/Managing-Pipeline-Dependencies

## Implementation and safety

New experiment module: src/fpgai_research/experiments/cst_paper_r4y_24ns_pipeline_source_prepare.py. Its plan() is pure. prepare() creates a fresh, create-only 4-case source root after a read-only original R4l SHA/native audit; it never runs GCC/Vitis/Vivado. audit() rechecks all 25 originally staged files, signatures, canonical source identities, identical CSim testbench/Tcl and source source-revision receipt, without native executions. Failed preparation keeps FAILED.json; nothing deletes an older experiment or one-time lock.

New tests: tests/test_cst_paper_r4y_24ns_pipeline_source_prepare.py. Positive/negative synthetic checks cover exactly one pragma, four arms, 24ns/1.5ns identity, no vendor calls, source/testbench/Tcl drift, fake manifest, source-original mismatch, old-root overlap, symlinked package and create-only refusal. These synthetic tests are NOT actual E1/E2/E3 results.

Makefile adds explicit research-r4y-plan, research-r4y-prepare, research-r4y-audit; none execute vendor tools. Research paper main C1/C2 work is unaffected.

## Execute source-only checks locally

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q \
  tests/test_cst_paper_r4y_24ns_pipeline_source_prepare.py \
  tests/test_cst_paper_r4x_feasible_only_core_latency.py \
  tests/test_cst_paper_r4w_feasible_clock_sweep.py

make research-r4y-plan

# First-run source preparation only. Do not delete or overwrite any evidence.
test ! -e build/r4y_24ns_four_arm_sources_20261106_01
make research-r4y-prepare

# SHA/report audit only, no new compiler/vendor work.
make research-r4y-audit
~~~

**Expected:** Four source packages, shared testbench and Tcl, precise source hash manifest, E2/E3/E4 and real physical/board latency all NULL/UNKNOWN. The earlier verified 47 tests belong to previous source revision 42326c4, not this new source implementation. The newly added code and synthetic tests must be verified on the author's workstation.

## Independent native work still needed

Next, implement a separate prospective create-only one-time Vitis E2/E3 runner consuming ONLY these new SHA-attested packages, with source revisions, CSim oracle, CSynth loop II and warning logs, output latency cycles, estimated clock/resources, and clear failed-case denominators. Then add a fresh routed Vivado stage (setup AND hold, all-path constraint checks, exact DRC rule severities) and optional board-instrumented core and DMA timings. Never rerun original R4w, patch original Tcl or treat cycles×clock as physical measured latency.

This is a supporting hardware mechanism/falsification study. The primary first-paper scientific gates remain bounded C1 source-event conformance, C2 useful policy-independent decision interface, independent Allo/CODO novelty control and transparent negative trade-offs.
