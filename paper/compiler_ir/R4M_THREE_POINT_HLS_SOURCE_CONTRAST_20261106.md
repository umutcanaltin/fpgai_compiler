# R4m: source-pinned three-point HLS path ledger — 6 November 2026

**Scientific status:** All R4e/R4l HLS figures below come from the author's original executed, source-hash-attested Vitis HLS 2023.2 studies. R4m is NEW read-only analysis code and tests committed to the research branch; **R4m itself has NOT yet been verified on the workstation**. The original R4l pair has already been executed; never repeat either native run.

## Scientific motivation

Earlier R4e contrasted original full-H/global-producer source order with row-H/row-interleaved source order. This changed both source issue scheduling and H materialization. The new R4k/R4l controls the row-interleaved source loop and only changes source scratch layout/capacity. A three-point path lets us describe the changes separately WITHOUT pretending to isolate physical implementation decisions or claim complete factorial causal effects.

All observations are the **same 32-total/6-integer format**, graph (2 fanout readers, shape 16×4×16×8, mixed mapping), model/weights, source-input CSim reference, and `xck26-sfvc784-2LV-c` FPGA part. Requested clock 12ns; explicit uncertainty 1.5ns. Both original R4e arms and both original new R4l arms passed their respective E2/E3 gates (max CSim abs error ~3.7956994241e-6 for the qualified 32/6 cases).

## Three source configurations and original observed E3 results

| Configuration (source control) | BRAM18K | DSP | LUT | FF | Worst HLS cycles | Estimated HLS period, ns |
|---|---:|---:|---:|---:|---:|---:|
| **A.** R4e original `full_H`: global producer, full H storage | 13 | 23 | 4,990 | 1,798 | 22,192 | 10.302 |
| **B.** R4l `full_storage`: row-interleaved producer/readers, full H storage | 13 | 21 | 5,095 | 2,120 | 22,954 | 9.857 |
| **C.** R4l `row_storage`: row-interleaved source order, row H reuse | 13 | 19 | 4,755 | 1,647 | 22,602 | 9.500 |

The historical R4e `row_H` is exactly the same C++ SOURCE as C: SHA-256 `aa0a7a2f684f3c0e6e64f6b4e3084c8f0dc93ae152318bd5339d9646b4894e9e`, and the two source-attested CSynth report hashes match `fe929aaa9db08990e5f7556d750d7d1eb7ef081c94b855424cccea38eb2644eb` in the author's original output. That is an explicit fourth **receipt for the same third point**, rather than a fourth independent experimental design point.

## Two descriptive source-controlled contrasts

| Source-level contrast, right minus left | BRAM18K | DSP | LUT | FF | Worst cycles | Estimated HLS period, ns |
|---|---:|---:|---:|---:|---:|---:|
| **A→B**, source loop/reader schedule changes, full H storage | 0 | −2 | +105 | +322 | +762 | −0.445 |
| **B→C**, source H materialization changes, row-source order fixed | 0 | −2 | −340 | −473 | −352 | −0.357 |
| **A→C**, historical combined source change | 0 | −4 | −235 | −151 | +410 | −0.802 |

The numerical deltas add: `(A→B) + (B→C) = A→C`. **That arithmetic identity is a report consistency check, not a randomized causal decomposition.** On the observed single case, the row-H memory intervention at unchanged source order improves estimated DSP/LUT/FF, worst cycles and estimated clock, but **does not change BRAM18K 13/13**. In contrast, row-source scheduling with full storage alone improves the estimated period and DSP, but increases LUT/FF/cycles. These mixed effects explain why the historical combined comparison concealed the favorable storage-only source contrast for FF and cycles.

## R4m implementation and anti-fabrication gates

New module: `src/fpgai_research/experiments/cst_r4m_three_point_source_contrast.py`.

New tests: `tests/test_cst_r4m_three_point_source_contrast.py`.

R4m calls **only** the original read-only `r4e.audit_native` and `r4l.audit_native`, verifying original frozen R4a/R4d source lineage and the complete R4k E1 + source/Tcl/testbench inventories. The 3-format R4d denominator remains visible (only 32/6 is natively qualified). It requires:

- Exact 32/6 original full/row and new full/row arm identities; identical original-R4e and new-R4l row-storage C++ SHA; complete source and numeric receipts.
- E2 acceptance plus valid E3 HLS estimates for all FOUR original receipts: original full, original row, new full, new row. If missing, resource deltas stay null, never invented.
- Exact original/new row-H source and CSynth report SHA match **and** all reported metrics agree; otherwise the path decomposition remains UNKNOWN rather than treating tool drift as a storage effect.
- An exact additive delta identity over BRAM18K/DSP/LUT/FF/URAM/worst cycles/estimated clock period; numeric nonfinite/negative/unsupported values are not qualified.
- Source evidence cannot be edited. Optional output is create-only and outside every original R4a/R4d/R4k evidence root; only the new report and its SHA inventory are written.
- No vendor call, no simulation call, no RTL elaboration, no Vivado, no physical E4 result, no claim of real achieved Fmax, power or area.

## Next workstation verification — READ ONLY and safe after executed R4l

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q \
  tests/test_cst_r4m_three_point_source_contrast.py \
  tests/test_cst_r4l_schedule_matched_storage_native.py \
  tests/test_cst_r4k_matched_row_schedule_storage.py

python -m fpgai_research.experiments.cst_r4m_three_point_source_contrast \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --output build/r4m_three_point_HLS_ledger_20261106_01
~~~

Never run `cst_r4l_schedule_matched_storage_native --execute-native` again on the original study; both original packages are already complete. If the optional R4m output path exists, omit `--output` and inspect stdout rather than overwrite. **R4m local tests and analysis are still pending**; the table's numbers are supported by R4e and the user's R4l original log, not by any new R4m execution.

## Remaining paper gates

The data presently cover ONE controlled row-source storage comparison (32/6, two readers). Before generality or physical timing/resource claims, preregister independent graph/fanout/shape and precision replications and complete separate Vivado placed-and-routed E4 trials with clean DRC, reproducible original source/RTL/ROM receipts. Investigate why HLS estimated BRAM18K ties using SHA-attested *new* R4l generated RTL and actual native implementation, without treating source RAM declarations as physical memory.

**Paper remains MAJOR REVISION / NOT SUBMISSION READY.**
