# R4e workstation native 32/6 evidence and R4f review — 6 November 2026

**Evidence tier:** user-shared actual Vitis HLS 2023.2 terminal outputs. No native tool execution took place through GitHub or this document. Source results were generated in the user's existing checkout; original files must remain unchanged.

The new R4d source population was pinned to Git revision **abe6aa37e5c799f27312b7f3a17309c6122ed7cb**. The preserved R4a/R4b original source revision is **fe99d627a5a16ce75e5d4df4542bb0f0f93e6ab0**. R4e code and six targeted tests were synchronized from branch SHA d950424d3ea27a4e33d36744a25291810aae511d. The log reports **6/6 tests passed** in 12.15 s and an R4e native dry-run (two planned 32/6 calls, zero actual vendor calls), after a clean read-only source inventory of all three format pairs. Then two original Vitis processes ran (16s and 15s, exit=0 each) and returned independently qualified E2 and E3 source-attested outcomes.

## Controlled 32/6 HLS case

- Shape: m=16,k=4,n=16,q=8; 2 MatMul readers and original mixed Phi mapping, full H versus row-local H source schedule.
- Format: ap_fixed with 32 total / 6 integer / 26 fractional bits, AP_RND and AP_SAT.
- Part: xck26-sfvc784-2LV-c; requested clock 12.0 ns; explicit uncertainty 1.5 ns; effective HLS timing criterion 10.5 ns.
- Same CSim absolute tolerance 0.02 and same 1×128 output sample for both arms.
- Result: **both E2 true and E3 true** according to original Vitis source-attributed R4e executor; neither is physical E4.

| Original measurement | full_H | row_H | Row minus full |
|---|---:|---:|---:|
| CSim failures (out of 128) | 0 | 0 | 0 |
| CSim max absolute error | 0.0000037956994241 | 0.0000037956994241 | 0 |
| CSim RMSE | 0.00000081167162467 | 0.00000081167162467 | 0 |
| CSim cosine | 1.0 | 1.0 | 0 |
| Estimated BRAM18K | 13 | 13 | **0** |
| Estimated DSP | 23 | 19 | **−4** |
| Estimated LUT | 4990 | 4755 | **−235** |
| Estimated FF | 1798 | 1647 | **−151** |
| Estimated URAM | 0 | 0 | 0 |
| HLS worst cycles | 22192 | 22602 | **+410** |
| HLS estimated period (ns) | 10.302 | 9.500 | **−0.802** |
| Effective-budget margin (ns) | 0.198 | 1.000 | +0.802 |
| Worst cycles × requested 12ns (ns, conditional HLS latency proxy) | 266304 | 271224 | **+4920** |

Original attested HLS report SHA-256: full H 6cfa5a92b42f10bea5b2ab639a96aebbcaa524642d8c3375467956734bcb1a94; row H fe929aaa9db08990e5f7556d750d7d1eb7ef081c94b855424cccea38eb2644eb.

The estimated clock period is **NOT** placed/routed period or achieved Fmax. The conditional requested-period latency proxy is **not** a measurement of actual device time. The observed CSim aggregate equality likewise does not prove bit-for-bit output equality.

**Scientific conclusion for this ONE qualified pair:** the earlier certified source H-buffer saving (240 elements) did NOT translate into estimated BRAM18K reduction. It was accompanied by lower DSP/LUT/FF estimates and more worst-case cycles; improvements are descriptive within this bounded single shape. The row source intervention changes global source loop scheduling as well as array size, so no causal claim that the saved array cells alone reduced logic or DSP is justified. No general hardware superiority or energy savings are established.

## Distinct earlier 16/6 failed experiment

The original R4b case_003 (16 total/6 integer bits) failed source-attributed E2 in BOTH arms with 28/128 failed outputs each and maximum absolute error 0.055395125 versus tolerance 0.02. The original R4d 32/6 new result with zero failures is evidence consistent with numeric representation as a cause, but does not identify the specific operations, saturation or quantization mechanism. It is not a bit-exact independent oracle. The NEW R4d 16/6 and 32/8 packages had NOT undergone native Vitis calls in the supplied terminal output and must remain UNKNOWN, not be counted as failures.

## Newly implemented R4f read-only review (unverified on original workstation)

Code: src/fpgai_research/experiments/cst_r4f_paired_hls_review.py
Tests: tests/test_cst_r4f_paired_hls_review.py

R4f independently calls the R4c and R4e original-evidence audits, checks revision consistency, retains all three format pairs, and derives within-format numerical and qualified E3 tradeoffs. It keeps unqualified and E4 values null. It makes **zero Vitis and Vivado calls**, writes nothing to original R4a/R4d evidence, and optionally generates a new create-only JSON report plus SHA inventory in a separate output directory. The R4f code/tests have not yet been executed on the original workstation.

### Safe commands

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pytest -q tests/test_cst_r4f_paired_hls_review.py

python -m fpgai_research.experiments.cst_r4e_numeric_format_native \
  --study build/r4d_numeric_format_source_20261106_01 \
  --original-study build/r4a_row_retained_paper_20261106_01 \
  --audit-native

python -m fpgai_research.experiments.cst_r4f_paired_hls_review \
  --study build/r4d_numeric_format_source_20261106_01 \
  --original-study build/r4a_row_retained_paper_20261106_01
~~~

Optional new create-only R4f report outside both original roots:

~~~bash
python -m fpgai_research.experiments.cst_r4f_paired_hls_review \
  --study build/r4d_numeric_format_source_20261106_01 \
  --original-study build/r4a_row_retained_paper_20261106_01 \
  --output build/r4f_paired_hls_report_20261106_01
~~~

Do **NOT** rerun the original 32/6 native arms. The other two formats remain separate future native experiments; no automatic Vitis execution should be started before inspecting the current read-only audit. Additional scientific requirements are per-element and per-intermediate fixed-point diagnostics, source-matched synthesized RTL memory classification, better distinct external comparison, cross-operator semantics, and clean-DRC physical evidence. Manuscript status remains **MAJOR REVISION**.
