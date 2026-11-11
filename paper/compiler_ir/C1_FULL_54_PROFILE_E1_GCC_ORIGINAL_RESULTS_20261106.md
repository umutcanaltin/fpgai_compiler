# Original 54-profile C1 E0→executed generated-C++ E1 study — verified source evidence

**6 November 2026 | Author-workstation ORIGINAL, SHA-audited, fully executed source-only E1 study.** The first compiler C1/C2 manuscript remains **MAJOR REVISION / NOT SUBMISSION READY** pending independent novelty comparison and verified physical feasibility. This ledger does NOT rerun or modify the original experiment, its signed E0 provenance, GCC outputs or its manifest.

## 1. Original completed measurements and source provenance

| Evidence | Author-observed original | Interpretation |
|---|---|---|
| Original E0 input | `build/paper_c1_seeded_metamorphic_20261106_01`; revision `637f2745721b32d6f5a7e02c4659b0623bc9eee7`; SHA report `8fa3848fcb0993e44a83391f595861943d75e3f09b669bd93b1b27e0222f3978` | Original 54 new-shape bounded fanout profiles / 486 selected E0 source-event traces; read-only audit reverified 0 oracle disagreements. These events belong to **9 selected controls per static profile**, not to 486 E1 executions. |
| Original full E1 result | Source revision `3bd1e36ff49864d89fa4c676e61da8ba0b936fce` under `build/paper_c1_e1_cpp_bridge_full_20261106_01` | **54** original static profiles matched against original E0 canonical event identities; all 54 qualified across both actual generated-C++ source scratch layouts. |
| Original focused regression | **17/17** tests passed in **1.68s** on author's machine | New C1 source and GCC bridge focused regression, not a claim that a full repository suite ran. |
| Actual GCC source execution | **108 GCC compilations; 108 genuine GCC executions** | Two source layouts for each of 54 profiles: `disjoint` and `per_element_optimal`; existing source-refinement checker qualified emitted C++ runtime memory/access and numerical checks for selected input. Zero Vitis/Vivado calls. |
| Original canonical source-E0 binding | `original_c1_event_hashes_matched=true`; E0 SHA as above | New source-code revision attested to all original E0 canonical typed-event hashes before each new GCC profile was executed. Original E0 event experiment and original files not rewritten. |
| Source-level scratch allocation contrast | **54/54 profiles had positive scratch-unit-slot savings** under reuse layout; **0 ties** | This counts source allocation units, not physical BRAM tiles/bytes or timing. The stdout does not show per-profile/absolute slot savings; don't invent them. |
| Read-only full E1 audit | **271 artifacts** independently verified; `original_manifest_sha256 = a79e700208e5d7cada93dd1f488809b6622f52cf6b14fa7b7d7b05a8d624d83e`; exact E1 source revision matched | 54 × (2 emitted `.cpp`, 2 actual stdout, 1 case JSON) + 1 study summary = 271 manifest-indexed artifacts. **The manifest is a local SHA checksum inventory, not an independently signed timestamp/provenance authority.** |
| Qualification limits | `board_qualified=false`, `physical_compute_latency=null`, `new_compiler_or_vendor_calls=0` during read-only audit | Actual GCC host execution is **E1**, not fixed-point Vitis CSim E2, HLS timing E3, clean/timing-qualified Vivado E4 or board throughput. |

**Interpretation:** This closes a meaningful empirical bridge between original model-level source trace certificates and actual GNU-C++-executed code for **the same 54 admitted static MatMul fanout profiles**. Both source layouts had matching functional/architecture/input identities; each case's source C++, observed stdout, original typed-event lineage and standalone summary entered the SHA audit. The test's actual GCC validator implements scratch address/last-use and intermediate/final numeric conformance checks to the independent NumPy reference; those are **bounded source-level observations**, not an all-input formal semantics proof.

**Crucial causal/novelty qualification:** The contrast is `disjoint` versus an explicitly optimized **fixed-order `per_element_optimal`** source scratch allocator on deliberately constructed multi-reader fanout examples. This choice creates an *expected opportunity* to reduce slot counts. A sign-only **54/54** result is therefore a validated **implementation correctness and attainable source-allocation contrast**, not evidence that an optimization algorithm discovered an unforeseen improvement, not a comparison to the best independently optimized compiler/allocator, and not generalizable to arbitrary models. This is why the new per-profile independent original read-only analysis must report absolute/percentage effect sizes and the value, schedule and witness checks, including whether different reader mappings show materially distinct behavior. Only an equal-functionality external memory allocator/Allo comparator and/or qualified hardware resource study could support a stronger resource-superiority claim.
## Original SHA-verified source effect sizes — now executed and quantified

The author ran `make research-c1-e1-review` successfully at repository SHA `a0d4db117247f08b0c85849f3fbd5d8c0cdac34a` after **40/40 targeted tests passed**. The tool rechecked original E0 report SHA `8fa3848fcb0993e44a83391f595861943d75e3f09b669bd93b1b27e0222f3978` and immutable original E1 manifest SHA `a79e700208e5d7cada93dd1f488809b6622f52cf6b14fa7b7d7b05a8d624d83e`, **271 source artifacts**, 54 profiles and 108 genuine GCC executions; this was **read-only with zero new compiler/vendor calls**.

**Across 54 distinct original fanout source programs:** median **15.5** source scratch unit slots saved; minimum **4**, maximum **45**; median **50%** per-profile source slot reduction (range **38.461538%–55.555556%**). The sum of distinct program capacities was **1,999 disjoint source units versus 1,009 reused units** (990 fewer in *aggregated distinct examples*, not the physical capacity of a single FPGA). Observed maximum final numerical absolute error **2.3781921365539915 × 10⁻⁷**, intermediate absolute error **2.408715820134688 × 10⁻⁷**. Both source layouts recorded **6,625 actual address witnesses**, preserving matched generated-source address observation counts.

| Fanout readers | Φ mapping | 9 original profiles: median saved source slots | Median per-profile reduction | Min–max source slots saved |
|---:|---|---:|---:|---|
| 2 | scalar | 14 | 45.454545% | 4–17 |
| 2 | mixed_phi | 12 | 50.000000% | 4–23 |
| 3 | scalar | 14 | 50.000000% | 7–27 |
| 3 | mixed_phi | 18 | 47.058824% | 8–33 |
| 4 | scalar | 24 | 50.000000% | 8–45 |
| 4 | mixed_phi | 32 | 50.000000% | 13–45 |

**Interpretation:** The output is a validated *source-level* allocation contrast and compiler execution-refinement check, not a statistically independent FPGA performance estimate or memory component count. `per_element_optimal` is chosen expressly to reuse source slots, so the positive-saving sign should not be presented as a surprising optimization discovery. Neither grouped sample count nor 990 summed source slots establishes BRAM, physical inference latency, core II or power. The separately completed original R4l/Vivado hardware study has its own distinct graph and tool-stage evidence.

## 2. Scientific limitations and honest negative controls

- **No cherry-picking**: 54/54 distinct predeclared C1 profiles completed across two source layouts. These are fixed-seed, synthetic family members under one topology, not 54 statistically independent applications. We retain the complete original population even if later extended test families produce counterexamples.
- **No precise source-slot effect size yet:** The author's summary supports 54 positive reductions, **not** mean/median saved slots, percent storage reduction, bytes, power, BRAM or per-reader effects. The per-profile source records saved inside the immutable original `summary.json` can be analyzed **read-only** to obtain the numeric source-slot distribution and maximum numerical errors. Do not retroactively change or rewrite original 271 artifacts.
- **Source E1 is not actual hardware scheduling:** GNU g++ execution and instrumented scratch witnesses cannot prove cycle-level Φ issue order, hardware block RAM port contention, place-and-route timing, energy or board DMA throughput. Original R4 routed source-matched storage contrast gives descriptive BRAM 7→4 at **both failing timing and clean DRC**. Physical compute and end-to-end latency remain **UNKNOWN**.
- **Independent novelty is open**: Allo PLDI 2024 already detects MatMul FIFO producer–consumer access-order mismatch, and CODO analyzes dataflow violations. The source-attested combined Φ/last-reader/actual-witness capability must be evaluated on a **matched, representable external baseline**; generic source ordering verification itself is not new.
- **C2 is separate negative evidence**: The original read-only source-cost analysis found lazy caching faster than eager in 108/108 paired CPU workloads under known-certified request streams. [Source-cost ledger](C1_E1_SMOKE_C2_SOURCE_COST_ORIGINAL_RESULTS_20261106.md). This E1 result does not establish an RL benefit or rescue a claim of eager source cost advantage.

## 3. New read-only exact-original postanalysis (source-created, unexecuted)

**NEW code, NOT YET author-workstation tested:**

- `src/fpgai_research/experiments/cst_paper_c1_original_e1_readonly_descriptives.py` consumes only the original full E1 study; locks the exact known **manifest SHA `a79e700…`**, its revision `3bd1e36…`, original E0 report SHA and 54/108/271 population; then invokes the **existing original E1 audit** to re-hash every original source C++/stdout/case/summary artifact before analyzing.
- `tests/test_cst_paper_c1_original_e1_readonly_descriptives.py` creates a synthetic complete 271-artifact/54-case test fixture, verifies a legitimate descriptive result and rejects manifest SHA drift, changed source C++, duplicated cases, non-positive savings, missing actual scratch-access witnesses and invalid numerical error fields. **Synthetic test fixtures are not original experiment results.**
- The analyzer reports **per-profile source slot counts and savings/percentages**, median/range and aggregated source slots across distinct designs (explicitly **not one deployed design**), separate rows for 2/3/4 readers × scalar/mixed-Φ mapping, actual scratch witness counts, and maximum independent numerical errors. The stdout is read-only and no source/native compiles are made. It will not compute real BRAM, II, physical clock, inference latency or hardware power. All output metrics remain unknown until the author runs this exact SHA-pinned analysis.

For compact terminal output, run `make research-c1-e1-review`, which uses `--summary-only` and preserves the overall result plus all six reader/Φ subgroups. Omit `--summary-only` on the module CLI to print every one of the 54 per-profile source-effect rows. Both modes are **read-only** and source-SHA-pinned; neither creates or changes an original report. The new display option and tests remain workstation-unverified until the author runs them.

## 4. Safe workstation command: NO new original experiment

```bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pytest -q \
  tests/test_cst_paper_c1_original_e1_readonly_descriptives.py \
  tests/test_cst_paper_c1_heldout_cpp_bridge.py
python -m fpgai_research.experiments.cst_paper_c1_original_e1_readonly_descriptives \
  --study build/paper_c1_e1_cpp_bridge_full_20261106_01
```

Do **not** rerun the original full `--output` mode: it has already completed with 108 generated-GCC executions. If the read-only analyzer fails a checksum, source revision or paired-identity gate, investigate and preserve original evidence; do not recalculate a replacement hash or alter a case to make a table pass.

## 5. Main paper readiness

C1 now has a real 54-profile E0-to-GCC E1 implementation bridge and no observed failures on the selected original population. The remaining high-value scientific task is **not more similar E1 synthetic examples**. It is (i) an independently executed, pinned Allo/CODO representative semantic comparator to locate the genuine contract novelty, and (ii) if the article asserts hardware utility, a **new, preregistered physically feasible timing/resource/latency qualification** of source-fixed alternatives with full unfavorable results. No source GCC slot saving alone establishes FPGA BRAM benefit. C2 may be reframed as useful policy-neutral compiler infrastructure rather than a proven second algorithmic contribution.
