# Scientific strengthening protocol — C1 meaning, C2 cost, and falsifiable baselines

**6 November 2026 — newly implemented source-only studies, NOT YET WORKSTATION-VERIFIED.** This protocol was created in response to a skeptical review of whether the original C1/C2 and R4 results provide a scientific contribution. All previously qualified original evidence is immutable. The active C1/C2 manuscript remains **MAJOR REVISION / NOT SUBMISSION READY** and draft PR #2 remains draft. No Github Actions; no new Vitis/Vivado runs.

## Editorial diagnosis — what the existing results do and do NOT establish

| Research question | Existing evidence | Scientific weakness to falsify | Next strengthened gate |
|---|---|---|---|
| C1: does value/lifetime safety plus per-operator Phi issue-order differ from value safety? | R3j 24 profiles/216 traces and original predicate recombination: 24 value-safe but issue-nonconformant controls; 0 combined-contract accepted | The comparison is `safe` versus `safe AND conformant` on deliberately selected controls; partly tautological, restricted static fanout, no independent source-to-RTL refinement | **New seeded held-out metamorphic population** + genuinely independent source-event oracle. Report shape/mapping coverage, positive and negative control classifications, disagreements, and non-executable gaps. Do not claim arbitrary soundness. |
| C1: is issue-order conformity a meaningful hardware obligation? | The source-level Phi READ-order predicate is checked, but source-level matching order does not prove PE/SIMD real execution schedule, legal memory port use or latency impact | **Important:** a value-safe alternative read order may be **functionally correct**; rejecting it is nonconformance to a *declared schedule*, not proof of a data hazard or bug | Define a concrete checked schedule/resource contract and emitted-source-to-event refinement with independently observed output/address/issue traces. Show a hardware-relevant invariant or narrow novelty to a source schedule-conformance interface. |
| C2: does eager compiler certification offer practical efficiency? | R3h 48 precompiled candidates, 0 additional source calls to choose cached accepted state | Must pay 48 source transitions upfront; old timing is unrandomized, and a **lazy memoized checker** is a strong, simple competitor | **New paired randomized CPU benchmark** across 3 strategies: direct per-query, lazy on-demand, eager full catalog, with exact same source parent/actions and verified resulting source certificates. Include eager loses/ties and null results. |
| C2: does three-valued certification give scientifically novel behavior beyond typed API? | R3f 21/24/3 and R3k held-out target limit statuses | The mask and direct replay call the **same compiler**, so agreement is a designed invariant; no search performance/learned-policy improvement | Independently derive source legality and run matched E0/E1 qualification under adversarial perturbed graph/action generation; compare meaningful false certifications and UNKNOWN behavior to independent baseline. |
| External validity and novelty | Allo source-verifies schedules; CODO checks dataflow violations; ST-Flow explores spatial/temporal mapping | No executed, matched Allo capability case. Source import failure or unavailable representation is not proof of verifier weakness | Run preexisting source-pinned R3i Allo study in isolated environment with supported safe and intentionally altered-value examples. Distinguish expressivity, source-equivalence, semantic contract and physical performance. No superiority unless matched validated. |
| Performance, latency and physical validity | R4l E3 matched HLS cycles 22,954→22,602, HLS BRAM13/13; R4p routed OOC BRAM tiles 7→4 but LUT/FF/WNS worse | Both designs FAIL 12ns timing and strict clean DRC. No feasible clock, achieved Fmax, realized compute/board latency | Keep physical compute/end-to-end latency UNKNOWN. Only run new physical clocks/throughput if a specific paper hypothesis cannot be answered otherwise; preregister feasible timing, compatible cycles and baseline. Never rerun original R4p/R4u. |

## E-A — C1 seeded held-out finite metamorphic challenge

**Code:** `src/fpgai_research/experiments/cst_paper_c1_seeded_metamorphic.py`. New source-only test file: `tests/test_cst_paper_c1_seeded_metamorphic.py`. Requires existing `sample_fanout`, `resolve`, `generate_fanout_events`, `verify_fanout_events`, `verify_fanout_issue`, and independent `reference_event_oracle`. **No RTL/Vivado.**

- Predeclare seeds **111, 222, 333**, reader counts **2,3,4**, mapping policies **scalar, mixed_phi**, and three seeded shapes per reader/policy. Shapes exclude all four frozen R3j shapes. This is **54 newly generated static profiles × 9 controls = 486 source traces** if the full study completes; the small `--quick` mode is only a test pilot.
- Nine exact per-profile classes: canonical; value-safe but declared-Φ-issue-order-nonconformant; premature H read; premature producer commit; wrong PE/SIMD owner; missing final read; duplicate producer contribution; overlapping live-reader source scratch; unknown output tensor.
- **Success criterion:** independent oracle and production source event/schedule verifiers agree for every admitted variant; all canonical traces pass both; all safe-order controls pass value safety but fail configured issue conformance; all other selected negative controls fail value safety. A disagreement is a **scientific counterexample to investigate**, not to overwrite or suppress.
- **Limit:** new shapes within SAME MatMul fanout topology are not new operator families or independent real workloads; seeded mutation labels are selected, not a statistically unbiased error rate. A value-safe but different issue order may be functionally correct and must NOT be called an unsafe data hazard merely because it violates a requested source scheduling rule.
- Preserve exact generated original event SHA, scratch offset SHA, source revision, all 486 row-level outcomes, and create-only output. No rerun on the same output; no base graph edits after seeing the result.

## E-B — C2 direct, lazy and eager source cost with matched requests

**Code:** `src/fpgai_research/experiments/cst_paper_c2_fair_source_cost.py`; new tests: `tests/test_cst_paper_c2_fair_source_cost.py`. Reuses original R3f frozen six supported graph cohorts (three MatMul-chain depths and three fanout sizes), original action tuples, `unified_transition` and `CompiledActionCatalog`, without changing the compiler.

- Three **genuine, measured source-compiler strategies**: direct recompile every requested action; lazy on-demand compile on first use and memoize by original state/action; eager compile all eight candidates and select from in-process source-verified catalog. Inputs/indices, source decision status/certificate/C++/next-state identities must match across all methods.
- Seeds **111, 222, 333**, request counts **1, 8, 32**, request distributions (a) repeat one known-certified action and (b) cycle among known-certified actions. Randomized deterministic strategy execution order, and per-graph/per-stream/per-seed timings and **actual source transition counts**. Full cohort is **6 graphs × 3 seeds × 3 query counts × 2 streams × 3 strategies = 324 measured method rows**, each with source identity and SHA. A fresh `--quick` report is NOT a paper result.
- The experiment has an explicit **eight-transition preflight per graph** to identify certified actions; this is EXCLUDED from all timed methods and fully disclosed. Certified-only queries are a convenient, **preselected consumer workload** and may differ from realistic blind exploration. Do NOT infer blind action acceptance, learning efficiency, or a global optimizer speedup.
- **Null-hypothesis-friendly reporting:** eager construction is expected to consume 8 source transitions even for one query, direct consumes Q, lazy consumes number of distinct requested actions. Any observed CPU benefit is workload-dependent and should not be called intrinsic. The lazy comparator makes hidden savings impossible; if eager loses to lazy, publish that. Do not substitute nanosecond wall times for physical inference latency.
- **Limits:** same six R3f cohorts rather than independent held-out applications, no CPU isolation or randomized process restarts in this source pilot, source compiler may contain incidental process caches, and source-checking is E0 not E3/E4. Use per-cohort distributions and avoid naïve aggregate confidence intervals treating correlated seeds as independent graphs.

## E-C — mandatory novelty and independent source/RTL refinement

**Allo** [PLDI 2024](https://doi.org/10.1145/3656401) already composes compute/memory/communication/type customizations and supports source/schedule equivalence. The existing [pinned isolated R3i runner](R3I_ISOLATED_ALLO_CAPABILITY_RUNBOOK_20261104.md) remains the correct starting point, but currently is not equivalent to every possible source-event ownership contract. Run safe/all-output-correct and intentionally wrong-output controls using the official supported API and archive exact source and verifier logs. Classify each test as `REPRESENTABLE_VERIFIED`, `REPRESENTABLE_REJECTED`, `UNSUPPORTED_REPRESENTATION`, `TOOL_SETUP_FAILED`, or `INCONCLUSIVE`. Unsupported representation and tool setup failures are NOT negative verification results.

**CODO** [ISCA 2026](https://doi.org/10.1109/ISCA66397.2026.00018) provides coarse/fine dataflow violation checking and performance-aware scheduling. **ST-Flow** [PACT 2026](https://doi.org/10.1145/3838684.3846865) handles spatial/temporal dataflow and hardware reuse. Their overlap precludes a claim of 'first contract-aware compiler' or 'first design-space explorer'. A novel contribution must be specified as a narrowly falsifiable static semantic object + proof obligation + policy-neutral transition and shown not to be just an ordinary schedule verifier reused under a new name.

The genuinely hard correctness gate is emitted executable source refinement: from a set of typed `CONTRIBUTE/COMMIT/READ` events and offsets to independently instrumented *actual generated C++* value reads/writes and issue ordering, with dynamic boundary cases and independently checked numerical outputs. We do **NOT** yet have a complete source-to-RTL or memory-bank/port timing proof. If this cannot be completed, narrow paper claims to the bounded static Model IR certificate and do not call it verified physical hardware safety.

## Reproducible workstation gate: NEW output names; CPU only

```bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m compileall -q \
  src/fpgai_research/experiments/cst_paper_c1_seeded_metamorphic.py \
  src/fpgai_research/experiments/cst_paper_c2_fair_source_cost.py
python -m pytest -q \
  tests/test_cst_paper_c1_seeded_metamorphic.py \
  tests/test_cst_paper_c2_fair_source_cost.py
test ! -e build/paper_c1_seeded_metamorphic_20261106_01
test ! -e build/paper_c2_fair_source_cost_20261106_01
python -m fpgai_research.experiments.cst_paper_c1_seeded_metamorphic \
  --output build/paper_c1_seeded_metamorphic_20261106_01
python -m fpgai_research.experiments.cst_paper_c2_fair_source_cost \
  --output build/paper_c2_fair_source_cost_20261106_01
python -m fpgai_research.experiments.cst_paper_c1_seeded_metamorphic \
  --study build/paper_c1_seeded_metamorphic_20261106_01
python -m fpgai_research.experiments.cst_paper_c2_fair_source_cost \
  --study build/paper_c2_fair_source_cost_20261106_01
```

If the first full run fails a scientific assertion, **stop** and preserve original negative evidence. Do NOT delete partial output, rewrite mutation expectations until passing, or switch to new seeds without recording the counterexample. New code/test preparation is NOT a verified experiment until the user runs it and checks the original receipts.

## Scientific quality gate

Submission to a serious compiler/architecture conference is conditional on: (1) a defensible novel semantic object relative to executed/pinned Allo/CODO/ST-Flow capabilities, (2) independently validated C1 contract coverage beyond hand-selected trivially rejected cases, including value-safe alternate orders, (3) calibrated C2 cost versus the lazy baseline and baseline hardware/language interfaces, (4) accurate timing-qualified latency or explicit UNKNOWN and no physical superiority claims, (5) transparent full run denominators, adversarial negative results, source and receipt hashes, no selective deletion and original revision pinning, and (6) independent coauthor review, reproducibility, licensing and rights clearance. If these gates cannot be met, choose the narrower compiler-contract workshop or engineering reproducibility paper instead of overstating results.

## Engineering change inventory and dependency impact

| New or changed path | Dependency connection | What changed or remained unchanged |
|---|---|---|
| `src/fpgai_research/experiments/cst_paper_c1_seeded_metamorphic.py` | `cst_fanout_models.sample_fanout`, `architecture.resolver.resolve`, `model_ir.fanout_contract` source event APIs, `cst_r3g_c1_independent_event_semantics.reference_event_oracle` | NEW independent-from-R3j seeded *configuration and mutation generation*. Exactly 54 unique static profiles across three seeds (cross-seed duplicate static graph/mapping profiles rejected), 486 expected trace outcomes. Each source acceptance and issue-order obligation compared to existing two checkers. CLI creates original source-revision-pinned create-only output or a `FAILED.json` counterexample receipt. Original R3j files unchanged. |
| `tests/test_cst_paper_c1_seeded_metamorphic.py` | Above C1 driver | NEW held-out shape/seed uniqueness, real source-event oracle check, expected mutation outcomes, SHA audit, synthetic source discrepancy and failed-attempt preservation tests. |
| `src/fpgai_research/experiments/cst_paper_c2_fair_source_cost.py` | Frozen R3f `_case`/`COHORTS`, `model_ir.compiled_action_catalog.build_catalog/select_certified`, and original `unified_transition`/source state hash | NEW fair direct, lazy-on-demand and eager catalog actual source compilation. Strict complete source decision identity, eager own compiler-transition receipt, 3-seed balanced execution order. Explicit eight-call per-cohort setup exclusion and source-matched 324 planned method rows; creates report SHA and create-only counterexample receipt on failures. Original R3f/h and production API unchanged. |
| `tests/test_cst_paper_c2_fair_source_cost.py` | Above C2 driver | NEW synthetic and real source-smoke tests, exact per-method compiler-call counts, matched result hash per trial, balanced three-seed order, tamper/partial failure handling. |
| `paper/compiler_ir/SCIENTIFIC_STRENGTHENING_C1_C2_PROTOCOL_20261106.md` | Active C1/C2 manuscript and scientific submission lock | NEW preregistration, scientific hypotheses, baselines, negative outcomes, experiment and file connectivity. |
| `paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md` | C1/C2 paper | Updated **interpretation**, not new results: a safe alternate issue order is not necessarily functionally wrong; HLS per-arm estimated cycle×period latency proxy disclosed apart from unavailable physical clock. Source studies listed as UNEXECUTED. |
| `paper/compiler_ir/FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md` | Evidence-to-claim gate | Updated novelty skepticism, C2 eager versus lazy compile-call inequality and new unexecuted experiment provenance. Original R3/R4 signed results intact. |
| `paper/compiler_ir/README.md` | Research documentation index | Linked new strengthening protocol as primary planned experiment, not completed original result. |
| Draft PR #2 | This source research branch | Summary updated with new falsification protocols, exact experiment status, no release or merge. |

**Files deliberately unchanged:** frozen original `build/r3*`/`build/r4*` evidence, original R4p/R4u permanent attempt locks and Vivado/DCP reports, existing R3j/R3f/R3h source experiment implementations, `model_ir` production compiler code, backend implementations, and `main`. No GitHub Actions, no external package installation, no vendor processes. All new output paths are separate. Any scientific mismatch must be preserved and reviewed, not automatically corrected by weakening its expected oracle checks.

**Important non-superiority control:** For any fixed eight-action parent catalog and a certified-only request stream with D distinct actions, eager source compile calls = **8** and lazy cache source compile calls = **D ≤ 8**. This means the eager method cannot claim a source-call efficiency improvement over lazy caching in this setup. Measuring wall time and surfacing negative results is the point; any useful C2 novelty must be established in *independent semantic/certificate capability* or downstream policy utility, not manufactured as an eager-cache speedup.

## Independent-competitor probe review: interpret R3i correctly

The existing `src/fpgai_research/experiments/cst_r3i_external_allo_capability.py` has **NOT YET executed** the competitor artifact on the author's workstation. A source inspection confirms it defines **three tiny two-reader MatMul fanout programs in Allo's DSL**: `safe`, `reordered` (the consumer execution order changes but value computation is intended equivalent), and `overwritten` (the shared producer is zeroed before its second consumer, intentionally changing the output). It calls the pinned Allo `verify` whole-output source/schedule equivalence on self, reordered and overwritten. Positive/negative outputs are also checked against NumPy. The process is isolated and pinned, but this is **not the same `Phi` event language** as our compiler; even a successful Allo verification outcome is only evidence about observable whole-output equivalence, not about whether Allo explicitly stores an operator-local value-read order certificate.

**Competing hypotheses:** If Allo judges `safe` and `reordered` equivalent, that is compatible with source functional equivalence despite declared `Phi` order nonconformance (not a weakness). If it rejects `safe` versus `overwritten`, it establishes genuine detection of a wrong-output mutation (again NOT a weakness). If it cannot represent a particular `Phi` ownership/reuse semantic under its public API, the valid result is **unsupported representation / inconclusive external equivalence**, not `Allo failed correctness`. Errors must remain errors, not negative capability results. This independent study is P0 because same-compiler source predicate ablation alone cannot establish priority over Allo's composable and verifiable schedules.
