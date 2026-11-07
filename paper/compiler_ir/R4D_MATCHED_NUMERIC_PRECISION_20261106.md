# R4d — new matched supported-precision study, source package preparation

**Status: IMPLEMENTED ON THE RESEARCH BRANCH; NOT YET WORKSTATION-EXECUTED.** This is a separate, create-only prospective study prompted by the original R4b case-003 E2 failure. It does not amend or discard the original R4a case-003 output and does not claim a successful fixed-point CSim result.

## Scientific question

Can increased fixed-point representational precision eliminate the original 28/128-output numerical CSim failure, while holding the SAME mathematical graph, weights, input samples, Phi mapping, source retention treatment, Vitis part, clock, uncertainty and floating-reference acceptance threshold? If both formats and storage variants pass E2, are the hardware effects of row retention consistent *within each fixed format*?

The architecture resolver's declared supported widths are **8, 12, 16 and 32 bits**. Earlier suggested 20-bit and 24-bit candidates would be outside this compiler contract and are NOT used. The predeclared study is:

| Format total/integer bits | Fraction bits | Purpose |
|---|---:|---|
| 16/6 | 10 | New source-pinned baseline reproducing original source/testbench SHA; original E2 failure remains untouched |
| 32/6 | 26 | Test higher fractional precision at the same integer field |
| 32/8 | 24 | Test increased integer range at the cost of two fractional bits vs 32/6 |

The comparison between 32/6 and 32/8 changes the fixed-point range/quantization tradeoff; **it is not a pure integer-range-only causal experiment**. New numerical precision is not a storage optimization; it affects synthesized operator widths/resources. Any BRAM/latency comparison of full vs row must be made **within the same precision**, never pooled across precisions as a single-factor storage effect.

All candidate formats use **absolute numerical tolerance 0.02** against the same float32 model, no post-hoc acceptance relaxation, `xck26-sfvc784-2LV-c`, requested clock 12 ns, explicit uncertainty 1.5 ns, original `case_003` shape (16,4,16,8), two original reader mappings, original weights and deterministic 1×128-output test sample.

## Code and safety

- `src/fpgai_research/experiments/cst_r4a_row_retention_study.py`: the existing canonical `_case` builder now takes optional `precision_bits` and `integer_bits`. The default **remains exactly 16/6** and must produce the same source CPP bytes as the original. No frozen original run changes.
- `src/fpgai_research/experiments/cst_r4d_matched_numeric_formats.py`: independently calls the ORIGINAL R4b full-source SHA inventory audit, requires the paper cohort and case-003 identity, compares regenerated 16/6 full and row source SHA against original E1 receipts **before creating an output directory**, and checks baseline Vitis model/testbench/Tcl package digests against original receipts. Invalid model/format/target combinations fail closed.
- Each of the three formats emits two original host GCC source/trace checks and two Vitis HLS **source packages only**. Source generated on the existing R2g/row emitter, no duplicated arithmetic implementation.
- A new create-only directory contains `summary.json`, `artifact_sha256.json`, and `format_16_06`, `format_32_06`, `format_32_08`, each containing the `full_H` and `row_H` arm artifacts. Failing partial studies keep `FAILED.json`. Output inside original R4a root or reusing any existing R4d output path is rejected.

## Commands on the original existing checkout

Run from your existing checked-out research branch with clean tracked changes after pulling the latest source. No alternative clone is needed.

```bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pip install --no-deps -e .
python -m pytest -q \
  tests/test_cst_r4c_csim_numerical_triage.py \
  tests/test_cst_r4d_matched_numeric_formats.py \
  tests/test_cst_r4a_row_retention_study.py \
  tests/test_cst_r4b_row_retention_native.py
```

Only after the focused tests pass, dry-plan **without any files or tools**:

```bash
python -m fpgai_research.experiments.cst_r4d_matched_numeric_formats
```

Then run original new E1 host GCC only, with a **new unique output directory**:

```bash
python -m fpgai_research.experiments.cst_r4d_matched_numeric_formats \
  --original-study build/r4a_row_retained_paper_20261106_01 \
  --output build/r4d_numeric_format_source_20261106_01
```

This command launches **zero Vitis, zero Vivado** calls. A passing source study does not mean the 32-bit native fixed-point result is accurate.

## What must be done next (R4e native mechanism experiment)

A [separate R4e guarded native runner](R4E_NUMERIC_FORMAT_NATIVE_GATE_20261106.md) is now IMPLEMENTED on the research branch, with read-only SHA audits and create-once Vitis invocation checks for the six new source packages. It has **not yet been workstation-tested or run against native Vitis**. The R4b runner remains specialized for the original R4a cohort and is not reused under a fabricated schema.

Require actual per-output mismatches and per-intermediate quantized/reference values, saturation checks and a causal explanation before asserting *why* 16/6 failed. Vitis HLS synthesis/resources may qualify only after both E2 passes and full timing constraints. A failed case still belongs in the denominator and missing cases stay UNKNOWN.

**Publication status:** this is a falsifiable, source-pinned numerical experiment. It is not a demonstrated hardware improvement or new numerical algorithm.
