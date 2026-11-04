# R3a — Six original full CNN→Dense Vitis CSim/CSynth cases, strict E2/E3 and 3 matched Phi contrasts

> **FINAL ORIGINAL NATIVE COHORT STATUS — DO NOT RUN THE HISTORICAL COMMANDS BELOW.** The author's 3 November 2026 log confirms `case_000`–`case_005` are all source-attested original Vitis attempts with 6/6 E2, 4/6 strict E3; `case_002` and `case_003` both failed the effective 9 ns HLS budget at 9.181 ns reported (preserved). Both completed E3-qualified pairs had HLS BRAM18k Δ0 and higher worst cycles with independent per-operator Phi. No native rerun or E3 promotion is permitted. The updated [R3c full-cohort publication protocol](R3C_COMPLETE_CNN_HLS_PUBLICATION_RESULTS_20261103.md) is now the active **read-only** analysis path. Historic pending statuses and native commands below record earlier steps and must not be treated as current instructions.


> **Later original checkpoint, 3 Nov 2026:** At analyzed code `fb6747a`,
> the author's 45 focused and 1,633 full tests passed; `case_001`
> qualified native E2/E3, completing one same-network HLS-estimate
> comparison with case_000 (BRAM Δ0, DSP +48, FF +1796, LUT +9146,
> worst HLS cycles +287). Original `case_002` passed E2 but failed the
> unchanged 9 ns effective E3 clock threshold, so the native R3b driver
> stopped. Thus 3/6 attempted, E2 3/6, E3 2/6, 1/3 both-E3 pairs.
> This is a preserved negative HLS result, **not** physical E4.
> For new opt-in completion of **only cases 003–005**, see
> [R3b follow-up](R3B_PUBLICATION_NATIVE_COMPLETION_AND_EVIDENCE_20261103.md)
> and its `--continue-past-attested-timing-failure` flag, after local tests.
> The follow-up code is not yet workstation tested.


**Updated 3 November 2026 from the author's ORIGINAL console:** Source revision `f665b62ca81ee181a1447e3eddd23d7eba578186` passed 38 focused and 1,626 full tests, generated six original R2z source packages, qualified 12/12 real GCC E1 input/stage traces, and passed six-package SHA re-derivation. Native R3a `case_000` was then attempted exactly once: Vitis exit 0 after 31 seconds, **original E2 true and E3 true**, and **HLS-estimated BRAM18K = 1**. This is **1/6 native cases**, not a full matched pair; cases 001–005 are not attempted in the supplied log, and no physical E4 result exists. All original input/case files and receipts must stay immutable. The new [R3b controlled native continuation/handoff](R3B_PUBLICATION_NATIVE_COMPLETION_AND_EVIDENCE_20261103.md) and tests are committed but **not yet executed on the workstation**; run their tests before native continuation. Scientific paper remains MAJOR REVISION.

## Scientific hypothesis and controls

The core test is whether the **same complete CNN-to-Dense network** can be compiled with **independently mapped per-operator Conv2D and Dense architecture choices** without dropping operators or changing model weights and tested numerical behavior; and, if both native variants qualify, what **actual Vitis HLS estimates** differ under those source-level mapping requests. The original R2z functional networks are three distinct, deliberately small source-cohort shapes. Each has a **scalar mapping control** and a **nonuniform Conv/Dense Phi intervention**, yielding **six cases / three matched pairs**:

| Network | Scalar original source | Independent Conv/Dense Phi source |
|---|---|---|
| `cnn_pool_dense` (two Conv layers, pooling, two Dense) | case_000 | case_001 |
| `cnn_strided_dense` (strided Conv, two Dense, batch 2) | case_002 | case_003 |
| `cnn_two_conv_softmax` (two Conv, Dense, Softmax) | case_004 | case_005 |

All six must have **identical graph functional hashes within each pair**, full original stage order and weights, identical **two-input CSim testbench SHA**, original tensor input batch SHA and **identical 10 ns / 1 ns uncertainty HLS Tcl SHA**. Only compiler operator mapping source changes within a matched pair. The canonical code emitter still uses full intermediate arrays, not stream/fusion/in-place lowering.

The parent R2z source campaign (see [R2z protocol](R2Z_MIXED_CONV_DENSE_NEURAL_NETWORK_PROTOCOL_20261103.md)) **must first pass local focused/full tests**, generate and independently audit **6 original source packages / 12 original actual GCC full-intermediate comparisons**. Those E1 source checks do **not** establish bit-exact `ap_fixed` equivalence.

## Original read-only and opt-in native execution

`src/fpgai_research/experiments/cst_r3a_conv_dense_native_hls.py` is a thin, source-pinned research workflow. It reuses unchanged `cst_r2z_conv_dense_network_campaign.audit`, existing `experiments.results.summarize_case`, **the existing `cst_nchw_hls_qualification.qualify_case` with its strict original CSim metrics and 10 ns/1 ns effective-clock HLS timing constraints**, and the canonical vendor-attributed `toolchains.vitis_hls.runner.run_vitis`. The script **does not change any original source or Tcl**. It verifies all six source cases and three matched functional/TB/Tcl/inputs control pairs before any vendor tool is launched.

**The HLS E2 gate:** original completed native CSim, source- and execution-attributed testbench and numeric log, exactly **two samples** and full output cardinality, **zero observed numerical failures and max absolute error within 0.01**, not merely a successful vendor return code.

**The HLS E3 gate:** E2, original completed vendor invocation with verified tool identity and stable C++/testbench/Tcl input hashes, original fresh matching XML SHA, **effective-clock feasibility using the existing 10 ns request/1 ns Tcl uncertainty-aware timing checker**, finite positive worst-case cycles and all required HLS-used BRAM_18K/LUT/FF/DSP estimates. Negative native timing or missing/invalid resources cannot qualify. An invocation that finished but failed any gate stays failed/UNKNOWN and is **not imputed** as 0 BRAM or an equal outcome.

**The matched E3 comparison:** exactly **three** fixed within-network scalar→independent-Phi pairs. Only if **both** have E3 will the report calculate intervention-minus-control differences in **HLS-estimated** BRAM_18K/LUT/FF/DSP and worst-case HLS cycles. E2 and E3 flags remain distinct. HLS resource/latency results are never physical Vivado BRAM, achieved Fmax, board performance or training accuracy. No pooling of historical R2u or R2y results with the new CNN protocol.

Commands (only after current R2z tests and original E1 source results have qualified):

```bash
cd ~/Desktop/github_projects/fpgai_research

python -m compileall -q src/fpgai_research experiments/compiler_ir tests

python -m pytest -q \
  tests/test_cst_r2z_conv_dense_neural_network.py \
  tests/test_cst_r2z_neural_campaign.py \
  tests/test_cst_r3a_conv_dense_native_hls.py

python -m pytest -q

# The original R2z source study EXISTS and is FROZEN; never run --output again.

python -m fpgai_research.experiments.cst_r2z_conv_dense_network_campaign \
  --study build/r2z_conv_dense_original_source_6_v1
```

Read-only native analysis on the original source (no Vitis calls):

```bash
python -m fpgai_research.experiments.cst_r3a_conv_dense_native_hls \
  --audit --study build/r2z_conv_dense_original_source_6_v1

python -m fpgai_research.experiments.cst_r3a_conv_dense_native_hls \
  --contrasts --study build/r2z_conv_dense_original_source_6_v1

python -m fpgai_research.experiments.cst_r3a_conv_dense_native_hls \
  --execute --study build/r2z_conv_dense_original_source_6_v1 \
  --start-case 0 --max-cases 1
```

The preceding R3a `--execute` command **is DRY RUN** and must report 0 vendor calls, even after case_000 has been attempted. The original first native case is **already complete**; do not rerun its case directory. To attempt an unattempted subsequent case after testing the newly committed R3b code:

```bash
# case_000 has ALREADY been executed; do not rerun it.
# After successful R3b tests/status, explicitly attempt the next ORIGINAL case:
python -m fpgai_research.experiments.cst_r3b_publication_handoff \
  --study build/r2z_conv_dense_original_source_6_v1 \
  --continue-native --execute-native --confirm-isolated-native \
  --continue-past-attested-timing-failure \
  --max-new-cases 3
```

**The first original case has been inspected and is E2/E3 qualified per the author's log.** If any later original native case is rejected or has unexpected CSim/HLS behavior, keep original files/receipt/logs intact and investigate; **never retry the attempted case in place**. Original indices 1 and 2 are already attempted and frozen; only indices 3–5 remain unattempted in the supplied log. Source-attested E2-pass/E3-clock failures may be skipped only with the NEW explicit timing-failure flag, without changing the original E3 rejection. All partial or other failed states still block. Source/cohort selection and actual original attempted count must be reported accurately even if results are negative.

No old R2i–R2x source/vendor evidence is changed. Original R2y Vivado `case_000` already completed with DRC warnings, is **not strict E4** and must never be rerun. The new R3a native folder is precisely the R2z prepared source's own fresh source-pinned case subdirectories; no new intermediate source generation or repacking happens when a native case is attempted.

## Manuscript interpretation

If the three pairs satisfy both E3, describe qualified **HLS estimated source-to-HLS impact on six small complete CNN/Dense neural models**, separately reporting E2 numeric failures and any E3 clock/resource rejections. Do not say that a source-level mapping difference *improved physical FPGA efficiency* without qualified, controlled routed implementation. A zero qualified-pair outcome is a negative experiment result, not permission to relax numeric/clock qualification or replace a failed native arm. This supports a bounded compiler-operational evaluation, not an independently established novel algorithm by itself.

See [active paper draft](MANUSCRIPT_DRAFT.md), [R2z original stage study](R2Z_MIXED_CONV_DENSE_NEURAL_NETWORK_PROTOCOL_20261103.md), and [strict original R2y Vivado OOC protocol](R2Y_NINE_CASE_STRICT_E4_PHYSICAL_PILOT_20261103.md).
