# R3b — Controlled completion and read-only publication evidence handoff (3 Nov 2026)

> **FINAL ORIGINAL R3A COHORT STATUS, 3 November 2026 — FROZEN, DO NOT RERUN:** All six original Vitis cases (`case_000`–`case_005`) have now been attempted on the author's workstation, **6/6 E2**, **4/6 strict E3**, **two attested E3 effective-clock failures** (`case_002/003`, both 9.181 ns vs 9.0 ns). **2/3** original matched E3 comparisons qualified; both showed increased worst HLS cycles/DSP/LUT/FF with tied BRAM18k. The newer [R3c complete source-attested publication tables](R3C_COMPLETE_CNN_HLS_PUBLICATION_RESULTS_20261103.md) supersede this earlier continuation protocol. **All native continuation commands below are historical/archival instructions; DO NOT run them on the complete original folder.** R3c is a new read-only code export, not an original native experiment, and still needs workstation tests. No physical E4 or novelty claim.


> **Latest original workstation result — 3 November 2026, source-pinned cohort
> `f665b62ca81ee181a1447e3eddd23d7eba578186`:** At current code
> `fb6747a`, 45/45 focused and 1,633/1,633 complete Python tests passed.
> The immutable six-case R2z source study passed its SHA audit and kept
> 12/12 original GCC input receipts. Vitis original `case_001` was run once,
> qualified E2/E3 (35 s), and completed the first comparison with
> `case_000`. Original `case_002` was run once (32 s), passed original CSim
> **E2**, but **FAILED_HLS_EFFECTIVE_CLOCK_BUDGET** at E3, so R3b properly
> stopped and preserved the original. Thus **3/6 native attempts, 3/6 E2,
> 2/6 E3 and 1/3 fully E3-qualified matched pairs**; cases 003–005 are
> **unattempted** in the supplied log. All new R3b continuation-after-clock-fail
> and rejected-clock diagnostic code is newly committed and **not yet tested on
> the author's workstation**; do not claim case 002 became E3. See below
> for the optional, source-attested continuation protocol.
>
> **First qualified HLS comparison** (`cnn_pool_dense` scalar control →
> independent Conv/Dense `Phi`): original HLS BRAM18K estimate **1→1
> (Δ0)**, DSP **45→93 (+48)**, FF **2913→4709 (+1796)**, LUT
> **10555→19701 (+9146)**, and worst cycles **624→911 (+287)**.
> At the unchanged 10 ns requested clock, worst-cycle×request latency estimate
> is **6,240→9,110 ns**, about **46% higher**, not a physical device
> measurement. Both arms meet the 9 ns effective HLS timing budget by very
> narrow margins (**8.971** and **8.997 ns** estimated clocks, respectively).
> No general mapping superiority or routed BRAM claim follows.


**Status:** The initial R3b read-only/stop-on-failure workflow was validated
on the author's workstation (45 focused / 1,633 full Python tests at `fb6747a`)
and two further original Vitis runs were performed; the later
**explicit skip-only-for-attested-E3-clock-failure** implementation and
additional tests on current `main` have **not yet been workstation-verified**.
All actual Vitis observations remain in the original source-pinned local
study, not in this documentation or the new code revision. The paper
remains **MAJOR REVISION / NOT SUBMISSION-READY**.

## Verified original workstation evidence (do not reinterpret)

The author's **earlier** console at original source revision
`f665b62ca81ee181a1447e3eddd23d7eba578186` reports:

| Cohort | Original observation | Evidence level |
|---|---|---|
| Focused regressions | 38 passed | Source/test |
| Complete Python regression | 1,626 passed | Source/test |
| R2z full CNN/Dense source | 6 original packages across 3 topologies | Source E1 preparation |
| Generated source host | 12/12 actual GCC input/stage traces qualified | E1, **not bit-exact ap_fixed** |
| Original R2z read-only package audit | 6/6 SHA-attested; no new GCC/Vitis | SHA/receipt audit |
| R3a original native case_000 | Vitis process exit 0, 31s; original CSim E2 true and CSynth E3 true; HLS-estimated BRAM18K = 1 | **One** qualified native HLS case |
| Original R3a case_001…005 | Not attempted in the supplied console | UNKNOWN, not zeros/ties |
| Three R3a matched Phi pairs | 0 pairs with both native E3 arms yet | UNKNOWN |
| Strict new R3a routed E4 | No original routed implementation was performed | UNKNOWN, not a physical result |

The original six-package source study is **frozen** at
`build/r2z_conv_dense_original_source_6_v1/` and carries original source
revision `f665b62...`. Current `main` may move while read-only source
re-derivation verifies its generated C++ hashes; do not mutate original
source, native receipts, build folders, input vectors or HLS Tcl. A new
code revision does **not** turn historical measurements into new runs.

## Safe sequence on the author's licensed workstation

Use a clean checked-out current `main` and an unmodified original study.
The R3b program is not a replacement for the source gate, real Vitis,
or the original R3a numeric/clock qualification.

```bash
cd ~/Desktop/github_projects/fpgai_research
git status --short
git pull --ff-only origin main
python -m pip install --no-deps -e .
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_r3b_publication_handoff.py
python -m pytest -q

# READ-ONLY; prints original evidence, all unknowns and submission blockers.
python -m fpgai_research.experiments.cst_r3b_publication_handoff \
  --study build/r2z_conv_dense_original_source_6_v1
```

The latest original log has **three attempted original native cases**:
`case_000` and `case_001` qualified E3, while `case_002` is a
source-attributed E2-pass/E3-effective-clock failure. The earlier strict
R3b continuation appropriately stopped at that failure. A newly committed
explicit narrow opt-in allows it to **skip the already-frozen, attested
clock failure** and run only the three remaining **unattempted** cases
(`case_003`, `case_004`, `case_005`). It does **not** retry, reclassify,
alter clock/uncertainty, relax numerical acceptance, or impute a matched pair.
It rejects all unverified, partial, non-clock or provenance failures.

First run the current focused/full Python suite and the read-only R3b
snapshot. **Only after they pass and the read-only original case_002
diagnostic confirms the source-attributed timing rejection**, explicitly
opt in to a maximum of **three** new original Vitis calls:

```bash
cd ~/Desktop/github_projects/fpgai_research

python -m fpgai_research.experiments.cst_r3b_publication_handoff \
  --study build/r2z_conv_dense_original_source_6_v1 \
  --continue-native --execute-native --confirm-isolated-native \
  --continue-past-attested-timing-failure \
  --max-new-cases 3
```

The runner uses the existing strict R3a case driver one original case at a
time, re-audits the complete source cohort before and after each invocation,
and preserves case_002's E3 failure. If another E2-passed original attempt
fails *only* the source-attested effective-clock criterion, this special
opt-in can collect the next unattempted case; all other kinds of failure
still block. Omitting the new flag retains the conservative original
stop-on-failure behavior. If partial artifacts or provenance ambiguity
are discovered, **do not** bypass them or rerun the original case.

For a follow-up corrected-clock study, use a separately pre-registered,
new source-pinned cohort and compare it as a distinct protocol, not as
a replacement for case_002.

After all attempted work, create a **fresh** read-only publication export.
A report directory is create-only. The author already created the original
`build/r3b_cnn_evidence_handoff_20261103_v1` after the 3-attempt checkpoint;
**do not overwrite v1**. Use v2 or the next unused name after later experiments.

```bash
python -m fpgai_research.experiments.cst_r3b_publication_handoff \
  --study build/r2z_conv_dense_original_source_6_v1 \
  --report-dir build/r3b_cnn_evidence_handoff_20261103_v2
```

It writes a JSON case/pair population and a Markdown handoff. The two
read-only files contain the E1/E2/E3 denominators, per-case actual
qualification status, HLS **estimated** resources or UNKNOWN, the three
matched contrasts and explicit publication blockers. It runs no vendor
tools. Do not commit native project trees or claim that these locally
generated results are downloadable until a sanitized artifact release is
approved.

## What must still be established for submission

1. A precise, falsifiable novelty claim relative to an **independently
   inspected** competing artifact (Allo, StreamTensor, Filament, CODO,
   hls4ml etc.) on a concrete semantic/operational problem. Mixed CNN
   support alone is not novel.
2. Source-level correctness plus original E2/E3 results for the prespecified
   six-case set and all three pair denominators; failures may be publishable
   as negatives, but never silently imputed as positive.
3. A fair, explicitly matched comparison, or a carefully limited source/HLS
   negative-translation result. Do **not** report HLS BRAM as routed BRAM.
4. A physical E4 claim only with clean DRC, reported timing, original
   source/RTL/report provenance and both matching routed arms. Historical
   R2y DRC-warned runs are not strict E4.
5. A manuscript that includes limitations, reproducible input/precision/
   toolchain settings, all original adverse outcomes and independent
   scientific review. Institutional IP and dataset/release permission remain
   separate from scientific reproducibility.

No code/tool can legitimately declare the scientific paper accepted or
submission-ready merely because a test suite passes.
