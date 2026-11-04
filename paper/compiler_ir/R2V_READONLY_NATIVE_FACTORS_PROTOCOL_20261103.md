# R2v — Read-only analysis of original R2u native HLS memory-factorial evidence

**Dated November 3, 2026. Research status: ORIGINAL R2v WORKSTATION VERIFIED AND FROZEN.** At analysis commit `23e998a0553e3eec6f05c5d3cd09a9b4a81af83e`, the user's workstation passed **26/26 focused and 1,572/1,572 full regression tests**, then executed the R2v **read-only** review on the preserved R2u original source revision `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`. New analysis files are in `build/r2v_r2u_native_factorial_review_v1/`; do not overwrite them. R2v confirmed **32 original invocation receipts / 32 source-attested CSynth XML files, 32 E2 CSim, 30 E3 CSynth**, with `case_004` and `case_012` failing the **effective clock budget**. Of **64/64** planned conditional factor pairs, **57** qualified and **7** were UNKNOWN, with 0 new Vitis/Vivado calls. The review establishes HLS estimates only, not physical E4.

## Workstation-verified original R2t and R2u facts

The user pulled original commit `0c6d85f` and reported **29/29 focused pytest and 1,565/1,565 full pytest**. The R2t original **36/36** source-global contiguous whole-tensor minima were proven and independently source-recomputed, with **0 search-budget unknowns**; **12/36** minima improved upon feasible greedy tensor first-fit, **36/36** exact tensor optima equaled the tensor concurrency lower bound, and **36/36** committed-element exact allocations were smaller than the exact contiguous tensor optimum. This is *source-event E0*, not a novel interval-coloring algorithm or FPGA physical resources.

The R2u original source package/factorial audit confirmed **32/32 original packages** from the **two fixed graph × four binary factor (16 levels) source design**. Then **32 source-attested native Vitis processes** completed with zero nonzero process exits, and the printed original native observation returned **32/32 CSim/E2 qualified** and **30/32 E3-qualified synthesis estimates**. `case_004` and `case_012` failed **`FAILED_HLS_EFFECTIVE_CLOCK_BUDGET`**, with valid CSim and Vitis source/report evidence. They are **unqualified for pairwise E3 resource deltas**, not resource ties or "zero BRAM" cases. Both were the table-to-switch address transformation under automatic memory inference and exact array declaration, in the two storage layouts of graph 0. The original clock target was **8 ns**, with **1.5 ns uncertainty** and **0.01 absolute** numerical CSim tolerance. All physical BRAM and achieved routed Fmax remain **UNKNOWN**.

The original output showed **HLS estimated BRAM_18K varying between 0 and 1 for the two-reader graph, and 2 and 3 for the four-reader graph**. These are ***estimated consumed HLS resources*** for the 30 E3-qualified cases and cannot be equated to routed Vivado BRAM or installed RAM primitives. The original six older warning-disclosed OOC Vivado matched comparisons remain six routed BRAM ties, **0 clean-DRC strict E4**.

## Verified factor-effect results from original source-attested HLS reports

Each pair's sign is **intervention minus control**, and these counts refer only to **E3-qualified HLS BRAM_18K estimates**. The user-reported original R2v summary and per-factor/per-graph audit agreed:

| Factor | Estimated BRAM reductions | Ties | Increases | Unqualified/UNKNOWN |
|---|---:|---:|---:|---:|
| Storage (disjoint → element-exact) | **2** | **13** | 0 | **1** |
| Address (table → switch) | 0 | **14** | 0 | **2** |
| Binding (automatic → requested `ram_2p` BRAM) | 0 | **10** | **4** | **2** |
| Padding (exact → 64-aligned) | 0 | **10** | **4** | **2** |
| **Total** | **2** | **47** | **8** | **7** |

By graph, storage reductions were **0/7 eligible** in the two-reader graph and **2/8 eligible** in the four-reader graph. Neither address lookup arm changed *estimated BRAM* across its 14 eligible comparisons, but **switch/auto/exact cases 004 and 012 failed timing**. Requested binding and padding each increased estimated BRAM in **2 of 6 qualified pairs** in graph 0 and **2 of 8** in graph 1. **All 32 original CSynth XML files were SHA-attested to the original invocations**, including the two E3-rejected runs. This stronger provenance still does not establish the inferred storage primitive, actual RAM port structure, routed Fmax or the reason for any hardware BRAM tie.

## Why a new, strictly read-only factor analysis is needed

The original `cst_r2u_hls_mechanism_execute --execute-native` printed per-case E2/E3 statuses, but the scientific claim requires **the complete 64 comparisons** across all four factors, while retaining both failed E3 cases and the original source/report hashes. Re-running those 32 vendor cases would destroy experiment provenance and is unnecessary.

We add `src/fpgai_research/experiments/cst_r2v_hls_factorial_review.py` and `tests/test_cst_r2v_hls_factorial_review.py`. The new module calls the existing R2u **read-only source-attested audit** with `require_current_source=False` because the original prepared source revision `0c6d85f` remains pinned while the GitHub main branch has advanced. The audit still checks saved original source/testbench/Tcl/package SHAs, regenerates each source arm using the unchanged compiler emitter, verifies original vendor invocation provenance, runs the strict E2/E3 numerical/timing/resource gates, and never calls Vitis or Vivado. The R2v analysis separately demands the **exact original full 40-character source SHA**, the exact 32-case/64-contrast population, complete source case numbering and every factor-level pairing. It recalculates all pairwise resource and cycle deltas from the exact two original cases and refuses forged or duplicate data.

R2v produces a **new, create-only output directory outside the original R2u vendor folder**, containing:

- `summary.json`: original 32 E2/E3 statuses, all 64 factor comparison outcomes, eight per-graph/factor strata, E3-qualified/UNKNOWN denominators, resource change signs, original source provenance, and source-report hash attestation for each present CSynth XML.
- `contrasts.csv`: each of the 64 predeclared comparisons as one line, with factor, graph and controlled source levels implicit from matching cases, E2/E3 gate, qualitative HLS BRAM classification, estimated BRAM/LUT/FF/DSP/URAM and worst-case cycle differences. **Unknown comparisons have blank numerical deltas, never fabricated zeros**.

The original source-attested read-only run **confirmed 57 E3-qualified / 7 UNKNOWN conditional contrasts** out of 64. The pytest module includes a **synthetic case array based on the pasted per-case BRAM numbers solely to test the arithmetic and rejection logic**; it is not a replacement for original vendor artifacts.

Per-factor results are signed as **intervention minus control**:

| Controlled factor | Control | Intervention |
|---|---|---|
| Storage | Disjoint | Element-exact |
| Address | Constant address table | Switch-case address decoder |
| Requested binding | Automatic HLS inference | Requested `ram_2p` + `bram` |
| Declared capacity | Exact element count | Capacity rounded up to 64 |

For example `bram_18k_b_minus_a = +1` means the intervention **increased** the HLS BRAM estimate by one; `−1` means the intervention **reduced** it. The analyzers never infer physical implementation from a source directive. A successful experimental arm may contain **zero HLS BRAM_18K** because registers/LUT memory are inferred; this is not equivalent to a hardware resource saving unless paired LUT/FF/DSP/latency/timing and actual RTL are analyzed. Because graph and factor interactions matter, no pooled number alone establishes a universal advantage. Document **all eight individual strata (2 graphs × 4 factors)** and the missing pairs.

## Exact verification and original read-only report

On the existing workstation checkout **without rerunning a single Vitis job**:

```bash
cd ~/Desktop/github_projects/fpgai_research

git status --short
git pull --ff-only origin main
git rev-parse --short HEAD

python -m compileall -q src/fpgai_research experiments/compiler_ir tests

python -m pytest -q \
  tests/test_cst_r2v_hls_factorial_review.py \
  tests/test_cst_r2u_hls_mechanism.py \
  tests/test_compiler_ir_series.py

python -m pytest -q
```

**Historical R2v commands already executed; preserve immutable v1 output and do NOT rerun these exact paths:**

```bash
python -m fpgai_research.experiments.cst_r2v_hls_factorial_review \
  --study build/r2u_hls_factorial_32_original_v1 \
  --output build/r2v_r2u_native_factorial_review_v1

cat build/r2v_r2u_native_factorial_review_v1/summary.json
```

The new study path is **strictly create-only**; never delete or overwrite prior results or the original R2u folder. If the new analysis fails, retain the original evidence and report exactly which SHA/source/qualification/denominator check failed. It may reveal unexpected UNKNOWN source bindings, malformed reports, tool warnings or absence of a strict match; **do not silently downgrade tests to pass**.

## Publication-level limitations and next obligations

A verified E3 factorial can establish **specific source-level treatment effects on estimated Vitis area and scheduling for these two static graphs** only. It still cannot establish physical BRAM, hardware Fmax, power, independent numerical equivalence beyond the qualified testbench, an actual RAMB primitive binding, or a general IR/compiler novelty claim. The (8) ns clock/0.01 threshold is intentionally different from older original 32 ns/0.1 studies and therefore **not directly pooled** with them.

After R2v, inspect the original **attested Vitis HLS generated RTL and memory-binding reports** in a separate *read-only* evidence process: identify named storage arrays, inferred implementation, resource promotions, declared padded elements surviving synthesis, and address-decoding LUT logic. Then, if an appropriate controlled physical study is justified, create a **new source-pinned Vivado OOC campaign** using exactly qualified RTL inputs with clean DRC/timing gates; do not claim old physical BRAM ties explained by new HLS estimates alone. Formal/capability comparison against Allo/StreamTensor/ST-Flow/CODO and the unimplemented fully in-place emitted C++ control remain publication blockers.

See [R2t/R2u original protocol](R2T_R2U_ALLOCATION_AND_HLS_MECHANISM_PROTOCOL_20261103.md), [manuscript](MANUSCRIPT_DRAFT.md), [formal source-event proof scope](FORMAL_SOURCE_EVENT_REFINEMENT_AND_TENSOR_BASELINE_20261103.md).
