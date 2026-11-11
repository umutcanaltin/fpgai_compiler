# R4c — original matched CSim failure: attested numerical triage

**6 November 2026.** This is a **read-only** diagnostic of the original R4b Vitis HLS case 003, not a new native run. Original study revision: `fe99d627a5a16ce75e5d4df4542bb0f0f93e6ab0`. Preserve all original `build/r4a_row_retained_paper_20261106_01/` files.

## New observed evidence (user's original workstation stdout)

Both independently packaged Vitis 2023.2 runs, `case_003/full_H` and `case_003/row_H`, reported the **same aggregate CSim metrics** against the original float32 reference:

| Metric | Full H | Row H |
|---|---:|---:|
| `ap_fixed` total/integer bits | 16 / 6 | 16 / 6 |
| Absolute acceptance tolerance | 0.02 | 0.02 |
| Samples × outputs | 1 × 128 | 1 × 128 |
| Failed output comparisons | 28 | 28 |
| Failure rate (per output) | 21.875% | 21.875% |
| Cosine | 0.999989333846 | 0.999989333846 |
| RMSE | 0.0183012798956 | 0.0183012798956 |
| Maximum absolute error | 0.055395125 | 0.055395125 |
| CSim acceptance | FAIL | FAIL |

The largest deviation is **2.76975625×** the prespecified absolute threshold. The RMSE being below that threshold does *not* change acceptance: 28 output-wise absolute errors exceed 0.02. Tool process exit 0 does *not* override the `FPGAI_CSIM_STATUS=failed` marker.

**Interpretation:** Identical aggregates across the two arms are evidence *consistent with* a common numerical issue. They **do not prove** the outputs are element-wise identical; distinct output vectors can share cosine/RMSE/max/failure counts. The available aggregates do not identify fixed-point rounding, coefficient quantization, saturation, testbench errors, or another root cause. Host GCC E1 float32 conformance is not an ap_fixed bit-exact CSim oracle. Both native arms remain E2-rejected, and there is **no E3-qualified BRAM/clock or E4 physical delta** for case 003.

## New read-only, provenance-aware command

This branch adds `src/fpgai_research/experiments/cst_r4c_csim_numerical_triage.py`. It first checks the *whole* original R4a E1 inventory using the existing R4b audit; then reads the original case only through the existing hash/source/testbench/Tcl/native-receipt-aware `summarize_case` path. It never trusts a free-floating log merely because numerical metrics appear in it. Missing/mismatched receipts return UNKNOWN and cannot be promoted to E2.

After pulling the branch and running tests:

```bash
cd ~/Desktop/github_projects/fpgai_compiler
python -m pytest -q tests/test_cst_r4c_csim_numerical_triage.py
python -m fpgai_research.experiments.cst_r4c_csim_numerical_triage \
  --study build/r4a_row_retained_paper_20261106_01 \
  --case-index 3
```

To **optionally** save an independent create-only evidence receipt, not under the original study:

```bash
python -m fpgai_research.experiments.cst_r4c_csim_numerical_triage \
  --study build/r4a_row_retained_paper_20261106_01 \
  --case-index 3 \
  --output build/r4c_csim_numerical_triage_20261106_01
```

The separate output has `diagnostic.json` plus `artifact_sha256.json`. This diagnosis executes **zero vendor tools**. It does not mutate source, threshold, prior study, and does not requalify previous failed cases.

## Next *new* experimental question (not yet executed)

Preregister a distinct R4d numerical mechanism investigation **before** any more HLS resource comparisons:

1. Preserve the original 16/6, 0.02 case as a failed member of the original five-pair denominator. Do not edit prior `tb.cpp`, `manifest.json`, metrics or native log. Never silently expand tolerance.
2. Produce a **new source-pinned** matched two-arm numerical-format sweep on the same case 003 input/weights/mapping, with explicit candidate formats (16/6 original control, 32/6, 32/8) and unchanged 0.02 float32 absolute acceptance threshold. Each precision must generate new source/testbench/Tcl digests and use a new create-only output directory. Distinguish more fractional bits from additional integer range; the [R4d create-only source preparation](R4D_MATCHED_NUMERIC_PRECISION_20261106.md) now implements these **resolver-supported** candidates; no new 32-bit native HLS results have been run.
3. Add independent element-wise outputs (not only four aggregates), per-operation fixed-point observations, and a separately implemented quantized reference to test where error first appears. Also check saturation counts and intermediate ranges. A larger format passing is not by itself proof that *precision* alone caused the previous failure until these mechanisms are isolated.
4. For each new format, run **both** full H and row H with matched clock/part/uncertainty/sample and source SHA receipts, reporting **all** negative and unknown outcomes. Only both E2-passing, timing/resource-attested cases may enter E3 synthesis comparison. Never pool different precisions into one single-factor storage-effect estimate.
5. In later hardware experiments, keep BRAM units and array implementation type distinct from source scratch elements, and include HLS estimated cycles, FF/LUT/DSP, estimated clock and, when available, routed clean-DRC evidence.

**Scientific status: E1 source storage improvement established for the bounded cohort, E2 failed in case 003, E3/E4 memory superiority not established. First paper still requires independent prior-art and broader operator semantics.**
