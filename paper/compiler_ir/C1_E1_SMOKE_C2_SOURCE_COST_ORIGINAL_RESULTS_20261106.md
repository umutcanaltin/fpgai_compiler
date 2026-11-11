# Original C2 source-compiler cost and C1 held-out E1 GCC smoke — verified workstation findings

**6 November 2026; source-only/E1 evidence. Author-pasted original terminal stdout and saved-file SHA audits; do not rerun or overwrite original output folders.** These results concern a frozen original C2 CPU source-checker workload and a new two-profile emitted-C++ GCC smoke. They do not demonstrate policy search quality, native FPGA latency or superiority to independent external compilers.

## 1. Provenance and reproducibility

| Item | Exact identity and author verification |
|---|---|
| Original C2 source benchmark | Git revision `637f2745721b32d6f5a7e02c4659b0623bc9eee7`; original local output `build/paper_c2_fair_source_cost_20261106_01`; schema `fpgai-research/C2-direct-lazy-eager-source-compiler-cost/v1` |
| Original C2 SHA-pinned report | `be2adaa4acf30196a2ccb17114f2caa0af3e6679dc2685872e27cb086ca86ec1`, author `--study` verified previously and original separate read-only `cst_paper_c2_original_readonly_analysis` now completed. Analysis used **zero source compiler calls** and **zero vendor calls**. |
| Original C2 sampled workload | Six fixed R3f source-parent graph cohorts, three seeds (111/222/333), two *certified-only* request streams, 1/8/32 requests per stream: **108 paired workload conditions × 3 methods = 324 observations**. An explicit **48-call full-catalog eligibility preflight** was excluded from the timed methods. All compiler source decision/result identities matched. |
| Generated-C++ E1 source bridge | New original two-profile smoke at revision `5bc8e9c1ec749c1c3dcf09bf85bace91503ee0c7`; output `build/paper_c1_e1_cpp_bridge_smoke_20261106_01`; two selected configurations (scalar and mixed-Φ); **4 genuine g++ compilations and 4 GCC source executions** (disjoint and per-element reuse layouts per profile); no vendor calls. |
| E0 anchor for E1 | Original 54-profile/486-trace E0 report SHA-256 `8fa3848fcb0993e44a83391f595861943d75e3f09b669bd93b1b27e0222f3978`, revision `637f2745721b32d6f5a7e02c4659b0623bc9eee7`. The bridge reported `original_c1_event_hashes_matched=true`. |
| New E1 read-only audit | Same source revision, `profiles=2`, `source_gcc_compilations=4`, `source_gcc_executions=4`, **11 source artifacts** checked; immutable E1 manifest SHA `51e022a07e35ab4ee96aed25d4a7bb990d97d49b54e98a07d977a6ed0bce4435`; `board_qualified=false`, `physical_compute_latency=null`, no new compiler/vendor calls during audit. |
| Repository release hygiene | At revision `5bc8e9c…`, **30 targeted tests passed in 1.60 s**, all 168 local public Markdown links resolved across 31 documents, curated **472-file** compiler preview selected from 895 tracked files; `clean_room_tested=false`, `history_security_reviewed=false`, `redistribution_license_approved=false`, `ready_to_publish=false`. |

## 2. Original C2 CPU wall time — full reported condition table

All displayed CPU timings are **median milliseconds** in 18 matched, **correlated** observations (6 fixed source graphs × 3 seeds) per workload condition. Figures come from the original saved-report SHA-verified read-only analyzer; they are not independent real-world ML workloads, FPGA inference latency, or policy-training benchmarks.

| Request pattern | Requests | Direct median ms | Lazy median ms | Eager median ms | Direct source compiler calls, median | Lazy calls, median | Eager calls, median | Lazy faster than eager (pairs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Repeat one known-certified action | 1 | 9.236882 | 9.359742 | 36.830830 | 1 | 1 | 8 | 18/18 |
| Repeat one known-certified action | 8 | 73.121406 | 9.904325 | 38.252520 | 8 | 1 | 8 | 18/18 |
| Repeat one known-certified action | 32 | 293.073960 | 12.250369 | 42.556626 | 32 | 1 | 8 | 18/18 |
| Cycle among known-certified actions | 1 | 9.190855 | 9.303959 | 37.249710 | 1 | 1 | 8 | 18/18 |
| Cycle among known-certified actions | 8 | 72.550611 | 33.683635 | 38.421565 | 8 | 3.5 | 8 | 18/18 |
| Cycle among known-certified actions | 32 | 290.607527 | 36.257298 | 43.026149 | 32 | 3.5 | 8 | 18/18 |

**Paired descriptives for all 108 workloads:** lazy faster than eager **108/108**, eager faster than lazy **0/108**. At Q=1, direct faster than eager 18/18 in each stream (36/36); at Q=8 and Q=32, eager faster than direct 18/18 in each stream (72/72). For lazy versus direct, lazy faster **74/108** and direct faster **34/108**, with no ties. The exact per-method total recorded source compiler transitions across all 324 measured method rows were **direct 1476**, **eager 864**, **lazy 198**; original preflight of **48 calls** was excluded and was not silently charged to a particular method. The counts are repeated/trial method-call observations over fixed source states, not independent search episodes.

**Interpretation:** For this source-only certified-request workload, eager precompilation can outperform a deliberately wasteful repeat-direct policy when many actions are queried, but it loses to simple on-demand lazy caching in every measured paired case. This is a **negative primary algorithmic speedup result**, important for publication honesty. The eager method still provides *complete up-front knowledge of all candidate statuses*, whereas lazy must discover them when requested; that information asymmetry was NOT evaluated for blind search utility, and the experiment deliberately selected certified actions in advance. No reward or RL agent was evaluated. No statistically independent general-population confidence interval or significance test should use 108 correlated trials as independent samples.

**Correct C2 paper framing:** Describe C2 as a bounded, source-pinned, policy-neutral certificate/action interface; do **not** advertise an eager catalog CPU optimization or source call benefit over lazy memoization. Keep C2 as supporting engineering infrastructure unless a genuinely different, independently checked multi-state search/unknown-boundary functionality is demonstrated in a new study with equal total screening budget and blind source actions.

## 3. Original matched C1 E0→E1 GCC smoke

The new bridge verified **2 source profiles × 2 scratch layouts**, with **4 g++ compilations/executions**, exact original C1 E0 canonical trace binding, actual generated-C++/scratch and independent NumPy output qualification (as asserted by the R2h bridge receipt checker). Both profiles exhibited **positive source scratch slot reductions** under `per_element_optimal` relative to disjoint storage: `source_slot_savings_count=2`, `source_slot_tie_count=0`. **The author's displayed summary does not provide per-profile slot counts or precise byte savings**; do not invent amounts or ratios. The bridge's manifest audit verified **11 source artifacts** and source provenance, with physical compute latency unknown and board unqualified.

**Scientific limitation:** This is a small, preselected **two-profile original E1 smoke**, not the 54-profile E1 cohort and not an independent formal source-to-RTL proof. Even if the full 54 profiles pass, address-trace and NumPy checks are bounded to source model assumptions, actual chosen inputs, emitted GCC semantics and a selected fanout family. The observed scratch slot saving is a **source allocation proxy**, not an actual BRAM tile reduction, timing-closed clock, throughput or board DMA runtime. The original R4 routed OOC storage pair **failed both timing and clean DRC**; strict E4 and achieved physical latency are still NULL/UNKNOWN.

## 4. Next original experiment / stopping rule

**Now authorized to stage a different, create-only full E1 study**, provided the original smoke remains frozen and a clean tracked checkout plus real `g++` are available. Use [the existing protocol](C1_E0_E1_HELDOUT_CPP_BRIDGE_GATE_20261106.md), with `--output build/paper_c1_e1_cpp_bridge_full_20261106_01` and **without `--quick`**. It should generate **54 matched source profiles × 2 layouts = 108 separate GCC source compiles and runs** under one new source SHA and SHA-backed receipts. If any original canonical E0 SHA, C++ source/NumPy/scratch witness or pairing check fails, **stop, preserve the FAILED receipt** and report the counterexample rather than changing the original results. Do not label this full study completed in advance.

**External novelty remains P0:** Allo's PLDI 2024 paper already verifies producer–consumer FIFO order mismatches between two MatMul kernels. A successful source-E1 bridge is implementation evidence, but not in itself new verification science. The reviewer needs a matched, runnable Allo/CODO comparison or an independently validated contract/utility result showing what is genuinely different.
