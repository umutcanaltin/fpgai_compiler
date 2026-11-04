# R3c — Complete original mixed CNN/Dense HLS evaluation and publication tables

**Observation date:** 3 November 2026. **Original six-case source revision:** `f665b62ca81ee181a1447e3eddd23d7eba578186`.
**Workstation code checkpoint:** `6e87a4c42aaa30f6ad9e9be1ecccd7218c6ede99`.
**Independent R3c export code:** committed *after* the six native measurements; its tests and export have **not yet been executed by the author**. No new original Vitis/Vivado evidence was created by coding.

**Scientific status: MAJOR REVISION / NOT SUBMISSION-READY.**

## Original complete prespecified cohort

The author's latest terminal log shows 30 focused tests passing followed by **1,649/1,649** full Python tests, six original source-attested R3a native Vitis invocations (all exit 0), and six source-attested E2 numerical passes. **Four** qualify under the unchanged effective E3 HLS clock/resource gate. **Two** (`case_002`, `case_003`) are original **E2-pass / E3-timing-rejected**, each with a separately hashed source-attributed Vitis report estimating **9.181 ns** against **9.0 ns** effective budget; each margin is **-0.181 ns**. These are rejected, not absent; their resources/cycles must not be used as E3-qualified data.

The original R2z source study has six source packages (three complete deterministic small CNN/MLP family topologies, scalar vs per-operator independent Phi) and **12/12** actual GCC input/stage E1 receipts. R3b qualified **2/3** original matched E3 pairs, preserving the full **3-pair** prespecified denominator.

| Original pair | Network | Scalar → independent Phi worst HLS cycles | Estimated BRAM18K scalar→independent | DSP Δ | FF Δ | LUT Δ | Status |
|---|---|---:|---|---:|---:|---:|---|
| `case_000→001` | `cnn_pool_dense` | **624→911** | **1→1** | +48 | +1,796 | +9,146 | Both E3 |
| `case_002→003` | `cnn_strided_dense` | **NULL** | **NULL** | NULL | NULL | NULL | Both E2 but both fail effective E3 clock |
| `case_004→005` | `cnn_two_conv_softmax` | **2,365→5,436** | **0→0** | +29 | +22,817 | +39,878 | Both E3 |

The HLS worst-cycles × **requested** 10 ns estimates for the two qualified pairs are **6,240→9,110 ns** and **23,650→54,360 ns**. This is a requested-clock throughput/latency proxy (assuming one inference completes in the stated worst cycles), **not achieved routed latency**. Under the original 10 ns requested / 1 ns uncertainty source protocol, both qualified independent-mapping arms are slower and use more DSP/LUT/FF estimated resources than their scalar baselines, with no BRAM estimate change. These results are **not** a general theorem, evidence that independent mapping is intrinsically worse, or a statement about optimized tuning. Per-operator architectural choice remains a representational feature; its benefit remains unestablished.

The `0` reported BRAM_18K for the two-Conv+Softmax cohort is an *actual parsed HLS estimate* for its E3-qualified cases, not missing data. It does **not** imply zero physical block-RAM primitives after implementation. Rejected `case_002` and `case_003` retain **NULL**, not 0. E3 timing estimates of 8.971/8.997 ns (`case_000/001`) and 8.917/8.973 ns (`case_004/005`) are not achieved Fmax.

## New R3c source-attested read-only export

The new `src/fpgai_research/experiments/cst_r3c_cnn_publication_tables.py` invokes the existing R3b read-only audit and independently checks all original 6-case/3-pair denominators, E2/E3 status, control/intervention SHA lineage, qualified resource and cycle deltas, and source-attested timing diagnostics. It refuses incomplete native cohorts, invalid or inconsistent measurements, and any attempt to claim resource/latency deltas for rejected comparisons.

It creates **new** outputs only; original `build/r2z_conv_dense_original_source_6_v1/`, and existing `build/r3b_cnn_evidence_handoff_20261103_v1` or `v2` exports, if present, are never modified. It performs **0 new GCC, Vitis, Vivado or RL calls**.

After a clean `git pull --ff-only origin main` and successful focused/full Python tests:

```bash
cd ~/Desktop/github_projects/fpgai_research

python -m pytest -q tests/test_cst_r3c_cnn_publication_tables.py
python -m pytest -q

python -m fpgai_research.experiments.cst_r3c_cnn_publication_tables \
  --study build/r2z_conv_dense_original_source_6_v1 \
  --output build/r3c_cnn_original_publication_tables_v1
```

This creates `summary.json`, `case_population.csv`, `matched_pair_population.csv`, `paper_results.md`, and `derived_export_sha256.json`. The last file hashes the four other newly derived results. All six attempted original cases, including clock failures, must remain in their raw data denominators. If the proposed output exists, use a **new** versioned path; do not overwrite.

## Paper claim decision

This cohort supports a **bounded source compiler operational/correctness claim**, and a limited negative finding for untuned nonuniform operator-level Phi under the specific native HLS timing/resource conditions. It **does not** independently establish a publishable novel compiler representation, physical RAM savings, superior HLS performance, arbitrary neural graph correctness, model accuracy, or RL benefits.

**Publication blockers:** a precise independently checked IR mechanism difference against competing artifacts (e.g. Allo, StreamTensor, Filament/CODO), a fair matched competing-code baseline or directly verified counterexample, stronger tensor/temporal semantic semantics, and a final manuscript tightly scoped to source/HLS experiments. Strict clean-DRC E4 routed evidence is absent; do not rewrite earlier DRC-warned physical observations as E4. Institutional release rights, citation/related-work review, and research artifact sanitization require their own decisions.

The original three pairs were planned *before* their native outcomes were known; use all of them in the paper, including the entire failed clock pair. This file is a truthful checkpoint, **not** external peer review or an assertion of acceptance.
