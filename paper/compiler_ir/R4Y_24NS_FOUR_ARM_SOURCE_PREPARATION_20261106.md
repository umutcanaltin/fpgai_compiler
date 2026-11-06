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

**Source metadata round-trip guard:** canonical source allocation maps include tuple-based slot witnesses. The read-only audit now compares against the JSON-normalized canonical manifest (tuples become lists after serialization), while separately SHA-verifying the original emitted C++ bytes and every staged file. A new focused regression verifies that untouched four-arm source packages survive prepare→audit without weakening any source, testbench or Tcl hash gate.


## 6 November 2026 — first author workstation R4y attempt and source-only repair

**Author's actual uploaded terminal log, revision `9f04971`:** initial targeted pytest finished **44 passed, 6 failed**. All six failures were a common read-only `audit()` blocker: `ValueError: R4y prepared source plan changed: status`. R4y `plan()` had a valid not-executed status, and `prepare()` legitimately changed the receipt to the new prepared-status value; the original `audit()` incorrectly required those statuses to be equal. The initial `make research-r4y-plan` succeeded and printed **four source cases, 24ns period, 1.5ns HLS uncertainty**. The author's real `make research-r4y-prepare` then failed **before allocating the new destination** with `ValueError: R4y original frozen source/E2/E3 preflight rejected`; the subsequent audit failed because the new study directory did not exist. **No new Vitis/Vivado vendor run and no actual new four-arm source package evidence** are established by that log.

**Independent code inspection identified the second cause:** `cst_r4l_schedule_matched_storage_native.audit_prepared()` returns a flattened R4l schema including `E1_matched_pair_accepted`, `paired_HLS_source_arms=2`, `package_hashes` and read-only source receipts; `audit_native()` adds `both_E2_qualified`, `both_E3_qualified`, `E2_qualified_arms`, `E3_qualified_arms`, `original_vendor_calls_during_audit=0`, `new_vendor_calls_during_read_only_audit=0` and `original_sources_modified=false`. The `source_only_E1_verified` boolean is in the older R4k summary, not the returned R4l audit result. Requiring this nonexistent field caused a **false rejection of a valid original audit**.

**Implemented repairs on the same draft research branch:** R4y now checks the **actual R4l schema**, two accepted matched source arms, both E2/E3 booleans and counts, original SHA packages, zero vendor calls during either audit stage, immutable source revision and original-source unchanged. The new `audit()` explicitly compares ALL plan fields **except `status`**, with an immediate independent check that the only permitted prepared status is `PREPARED_FOUR_SOURCE_ONLY_NEW_VITIS_PACKAGES`. This preserves fail-closed verification for all graph/part/clock/format/source-SHA/numerical-null fields, while making the correct planning-to-prepared transition legal. File symlink checks for new source/package paths now run before model bytes are read.

**Additional synthetic regression controls:** the `_original_good()` test fixture now models the *real* returned R4l audit schema (no R4k-only E1 key). Positive tests cover prepare→audit and the real-schema acceptance. Negative cases reject bad R4l schema, incomplete E2/E3 counts, unexpected original vendor receipts, changed planned vs prepared statuses, changed requested clock, falsely set `E2_CSim_qualified=true`, unexpected added files, manifest forgery and original-source SHA mismatch. All are source-only; no native Vitis/Vivado execution is introduced.

**Verification status:** fixes were committed after the uploaded run; **the updated R4y source/tests have not yet been run on the author's workstation**. Avoid describing the user's **44-pass / 6-fail** result as a clean test pass or the 24ns packages as created. No previous R4w one-time lock, original Vitis/RTL/ROM, or paper physical evidence was modified. The full separate E2/E3/E4/board experiment remains not executed.

### Exact safe rerun after pulling the repair

```bash
cd ~/Desktop/github_projects/fpgai_compiler
git status --short
git switch research/first-paper-source-memory-bridge-20261106
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q   tests/test_cst_paper_r4y_24ns_pipeline_source_prepare.py   tests/test_cst_paper_r4x_feasible_only_core_latency.py   tests/test_cst_paper_r4w_feasible_clock_sweep.py

make research-r4y-plan

# Preparation failed BEFORE allocating this path in the author's uploaded log.
# Never remove a path if it now exists; inspect it instead of overwriting.
test ! -e build/r4y_24ns_four_arm_sources_20261106_01
make research-r4y-prepare
make research-r4y-audit
```

Run these foreground steps in order and stop on any failure. Source preparation is fresh data creation only, with no GCC/Vitis/Vivado; audit verifies original SHA and common HLS source/testbench/clock receipt. Only then consider a separate create-once native E2/E3 study with real CSim/CSynth.
