> **2026-11-03 — R2m v2 WORKSTATION VERIFIED:** The user ran **11/11 focused tests** and **1,447/1,447 full tests** (136.67s), then independently re-audited all frozen source-pinned R2i/R2j/R2l data and exported the corrected v2 results to `build/r2m_original_six_physical_paper_v2/`, with **zero new vendor calls**. The original v1 output remains preserved. The main paper now includes actual **BRAM, LUT, FF, DSP and OOC WNS** values. Among the six selected HLS BRAM-positive routed pairs: **6/6 BRAM ties at 1.5 → 1.5**, **6/6 DSP ties**, **3 LUT increases and 3 decreases**, **3 FF increases and 3 decreases**, **4 positive and 2 negative descriptive WNS differences**, and **0/6 strict clean-DRC E4**. The other **30/36 original physical pairs remain UNKNOWN**. Preserve all original source and vendor outputs and both create-only export versions.

# R2m — Final source-pinned six-pair physical evidence tables (read-only)

**Research date: 2026-11-03. The original R2l experiment and both R2m exports have completed; the corrected R2m v2 was independently executed on the user's workstation and is the canonical source for the paper table.**

## Original observed results, frozen

- The HLS source is fixed at `b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`; full original study comprises 72 cases, 36 matched pairs, 24 paired E3-qualified and 12 paired E3-unknown.
- Six **prespecified HLS-estimated** BRAM-positive original indices: 20, 23, 28, 31, 32, 35.
- Pair20 original routed pilot: `build/r2j_fanout_physical_36_pairs_v1`.
- Five additional original matched pairs in `build/r2l_five_positive_physical_followup_v1`. Ten original follow-up Vivado OOC processes completed with exit 0 on the user's licensed workstation, but all resulted in warned routed observations.
- The independent R2l completion output `build/r2l_final_six_pairs_after_followup_v1.json` reports **6 routed BRAM ties**, **0 routed BRAM reductions**, **0 strict clean-DRC E4**, **0 remaining unobserved among selected six**. The other **30/36** full-cohort matched pairs have **no physical observations**.
- Warning-only OOC results may support *descriptive*, not clean-DRC E4, physical resource and slack comparisons. No achieved Fmax, deployed bitstream, measured power, guaranteed benefit or external firstness is inferred.

## New R2m analysis code

`src/fpgai_research/experiments/cst_r2m_original_physical_paper.py` audits original frozen studies using the **existing R2i/R2j/R2l production readers**, then creates the following *new* output folder and files:

- `results.json` — all 36 pairs, each original observed arm's full routed LUT/FF/DSP/BRAM, WNS, DRC rule counts, ROM evidence category, and original four report SHA-256 digests; all E3 unknown and unmeasured physical effects left null.
- `all_36_pairs.csv` — full source and HLS denominator with six measured routed pairs and the 30 unobserved physical comparisons left empty, not zero or tied.
- `paper_table.md` — six matched observed WNS/BRAM/LUT/FF outcomes with explicit warned-vs-E4 labelling and a limited scientific conclusion.

The program rejects incomplete or swapped physical arms, changes to case identities, missing original SHA fingerprints, unverified DRC, incorrect routed resource/timing arithmetic, and any effort to promote warned results to strict E4. It requires all **exactly six** original positive pairs to have been completed; never silently drops failed pairs. No new GCC, Vitis or Vivado calls are made.

## Workstation verification

Use the already cloned local Git checkout. No recloning, no reinstalling tools, and **do not rerun any of the existing 12 Vivado physical cases**.

~~~bash
cd ~/Desktop/github_projects/fpgai_research

git status --short
git pull --ff-only origin main

python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_r2m_original_physical_paper.py
python -m pytest -q
~~~

If any test fails, report the traceback; **do not** modify the original experiments in the `build/` tree.

## Create the complete physical result tables (no vendor invocation)

~~~bash
python -m fpgai_research.experiments.cst_r2m_original_physical_paper \
  --hls-study build/r2i_fanout_native_36_pairs_v1 \
  --original-physical-study build/r2j_fanout_physical_36_pairs_v1 \
  --followup-study build/r2l_five_positive_physical_followup_v1 \
  --output build/r2m_original_six_physical_paper_v1
~~~

This is a **create-only path**. If it already exists, keep it unchanged and choose a new output path rather than overwriting.

Read `build/r2m_original_six_physical_paper_v1/paper_table.md` and `all_36_pairs.csv` to inspect the physical LUT/FF/DSP distribution across all six pairs, the original DRC rule populations and warning-only classification, and whether timing trends are consistent or workload-dependent.

## Paper claim boundary

A defensible negative result is that **all six selected HLS-estimated BRAM-positive pairs tied in original Vivado routed BRAM tile allocation**. This does not refute independently verified source-level scratch lifetime/aliasing correctness. It *does* falsify any guarantee that optimizing source scratch allocation reduces physical BRAM for these studied workloads under their tool/part/clock configuration.

The broader manuscript's proposed novelty remains a bounded, certificate-backed and executable **operator Phi + tensor-element lifetime contract** that must be compared to Allo, HIDA, StreamTensor, ST-Flow and relevant prior schedule/lifetime work. Internal source ablations alone do not establish priority. The physical campaign as currently run establishes **zero** strict clean-DRC E4-qualified pairs; a new clean source-controlled study would be needed to claim E4.

**Do not silently redefine an original R2l matched pair, waive DRC or change clock, sources or packaging after viewing the results.** Any follow-up design correction must be an explicitly new preregistered experiment with both matched arms, fresh immutable identity, and independent semantic equivalence.
