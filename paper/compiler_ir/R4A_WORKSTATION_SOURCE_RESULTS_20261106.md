# R4a first-paper row-retained materialization — observed source results

**Recorded 6 November 2026. Evidence tier: user-shared actual workstation terminal output plus source code inspection, not an independent read of local generated binary/raw receipt hashes.** The original evidence lives on the author's workstation and must remain unchanged. Source revision: `fe99d627a5a16ce75e5d4df4542bb0f0f93e6ab0`.

## Completed original E1 execution

- Focused test command, involving `tests/test_row_retained_fanout_cpp.py`, `tests/test_cst_r4a_row_retention_study.py`, and `tests/test_cst_r4b_row_retention_native.py`: **26 passed in 1.61 s**, zero failures (as reported in terminal).
- Quick create-only actual GCC study `build/r4a_row_retained_quick_20261106_01/`: **3 matched host GCC pairs**, aggregate **13 source scratch elements saved**; output reports **zero Vitis/Vivado calls**.
- Full create-only actual GCC study `build/r4a_row_retained_paper_20261106_01/`: **5 matched host GCC pairs**, aggregate **493 source scratch elements saved**; output reports **zero Vitis/Vivado calls**.
- Both studies report the same full Git source SHA: `fe99d627a5a16ce75e5d4df4542bb0f0f93e6ab0`.
- The quick cohort is **strictly a subset** of the paper cohort. These are five unique profiles, **not eight independent observations**. Distinct original GCC invocations occurred in both output directories, but don't pool the repeated small profiles as independent evidence.

## Source capacities — interpretation only

From the committed prespecified profile configuration and the source relation `(m-1)*n`, the **expected per-case scalar scratch savings** in the five paper profiles are 3, 4, 6, 240 and 240, summing to 493. The full H buffer shrinks from `m*n` to `n` cells. These per-case numbers are inferred from experiment code and the observed aggregate, and **must be cross-checked against the existing local case receipts before final manuscript use**.

This study validates on the supported static R2f fanout domain that actual generated host C++ follows the row-staged execution and that the tested numerical intermediates and outputs pass independent NumPy comparisons. It does not validate physically concurrent execution, producer/consumer FIFO ports, or fixed-point cycle-accurate scheduling.

## What has NOT been established

- There are **no newly executed Vitis E2 CSim or E3 synthesis results** at this point, and no measured BRAM18K/FF/LUT/DSP/cycle improvement.
- There are **no clean-DRC Vivado physical E4 results** from this new study.
- A reduction in source slots is **not** an empirical reduction in FPGA BRAM.
- Five hand-selected finite configurations are not statistical generalization; the source retention/fusion mechanism alone is not a novel invention.
- No new Allo comparison, mixed Conv2D/Dense contract extension or hardware source refinement proof was run.

## Immediate next gate: read-only provenance and then one new matched HLS pair

1. Audit the original full five-case E1 root using `cst_r4b_row_retention_native --study build/r4a_row_retained_paper_20261106_01 --audit-native`. A missing-native state should be explicitly UNKNOWN, not 0 MB savings.
2. Dry-run the predeclared medium case index 3, which has the same two arms, same target, same input and requested clock. This tests the synthesis sensitivity of a nontrivial 240-element source difference. Running a single pair first is a tool-readiness pilot, **not a replacement for the full five-pair denominator**.
3. If the original source audit and dry run pass, explicitly opt in to two Vitis invocations for medium case 3 in the **full paper** study directory; never run that pair again in the quick source root. Audit actual source/testbench/Tcl SHA and qualified CSim/CSynth reports.
4. Continue cases 0–4 and preserve all invalid or ineligible results as unknown. Do not filter by favorable BRAM response.
5. The entire paper still requires independent comparator coverage, heterogeneous operator contract evidence and (for a physical claim) routed clean-DRC implementation.

**No source code, native tools, experiment data or RL files were changed when adding this record.**
