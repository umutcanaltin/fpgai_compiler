<!-- Historical scientific review and internal research provenance; not public user guidance. -->

# Independent Scientific Review Request — FPGAI CST Model IR

> **New independently reported workstation evidence since this review brief was drafted (2026-11-03):** R2n original source revision `6296f99eb5d52371bffb9755989a340aadbe7ff6` completed **9 action cases: 3 actual source E1-qualified, 3 rejected, 3 E0 UNKNOWN**; the independent JSON receipt audit agreed. Subsequently, R2o original source revision `ce8083c0ad5b827232e1e67322811b25edd3ac79` passed **27/27 focused and 1,484/1,484 local pytest** and completed **12 source actions (three 2/3/4-reader fanout graphs × four action roles), 9 actual GCC E1-qualified, 3 compiler rejected, 0 UNKNOWN/failures**, with original source/manifest/stdout SHA read-only audit agreement and **102/210/372** runtime source-address witnesses per qualified action on the respective fanout graphs. No new original Vitis, Vivado or RL training calls. See `paper/compiler_ir/R2N_C3_E1_QUALIFIED_DECISION_IR_20261103.md` and `R2O_C3_MULTI_READER_DECISION_SOURCE_PROTOCOL_20261103.md`. New R2p common-typed C3 facade, tests, six-action experiment and audit **exist in source but ARE NOT YET workstation-verified**; see `paper/compiler_ir/R2P_UNIFIED_C3_SOURCE_INTERFACE_PROTOCOL_20261103.md`. Review whether these bounded source results genuinely strengthen C3 *scientific novelty*, rather than just implementation breadth. Keep all original C1/C2/physical conclusions and the no-GitHub-Actions requirement unchanged.

**For:** Claude (or an independent senior reviewer in compilers, FPGA/HLS, programming languages, verification, and learning-guided design)  
**Date:** 2026-11-03  
**Repository:** https://github.com/umutcanaltin/fpgai_research  
**Paper:** `paper/compiler_ir/MANUSCRIPT_DRAFT.md`  
**Scope:** Read-only, skeptical **publication-quality scientific audit**, not code generation or an endorsement.

> **Important operational constraint:** We do **not** have or use GitHub Actions tokens. GitHub Actions/CI is not available for this review. **Do not call GitHub Actions, request CI tokens, inspect workflows or job logs, require CI to be enabled, rerun workflows, or call CI failures a publication blocker.** Base assessment on source, local experiment logs/results provided by the author, Git commit history, independent analysis and literature. Ordinary `git log` and `git show` are appropriate. The author's workstation has run the described local tests; those are not equivalent to independently executed tests by Claude.

## 1. Mandate: be a critical peer reviewer, not an advocate

Please evaluate **whether this is genuinely publishable scientific research**. We want a difficult, specific, evidence-grounded review:

- Is the principal scientific idea new or a repackaging of established scheduling, dependence, liveness, storage reuse, graph/dataflow, and source-verification concepts?
- Are the definitions, hypotheses, assumptions, proofs/proof sketches and falsification tests logically valid? Can you find counterexamples?
- Does the current implementation **actually realize** the proposed IR semantics, or are important relationships only metadata, source hashes, generator replay, or restricted cases?
- Are experiments scientifically fair and sufficient to support each claim? Inspect denominators, confounders, original source revisions, negative results, validity tiers and uncontrolled assumptions.
- Is the paper publishable **as an IR-semantics/certificate/compiler paper even without running direct competitive benchmarks**? We do not assert that competitors are slower. **Related work and novelty positioning still require scrutiny**, and benchmark comparisons are necessary if you decide a specific claim needs them.
- **Do not omit the secondary RL-/search-compatible Model IR contribution.** Test whether that claimed secondary research advance is technically sound and meaningful versus existing compiler-RL interfaces, and how strongly it is currently demonstrated.
- Is the manuscript honestly written and internally consistent? Identify every sentence/paragraph or numerical statement that is wrong, overgeneralized, ambiguous, post-hoc, unsupported or stale.
- Recommend the **minimum concrete experiments, counterexamples, formalizations and manuscript edits** needed for a realistic submission. Prioritize by scientific risk rather than volume of code.

**Please do not answer with generic praise. You are free to conclude "not publishable as written" or "insufficient novelty" and must explain exactly why and how to remedy it.** Do not change code or documentation during this review.

## 2. Intended scientific story — each contribution needs its own verdict

### C1 — Compositional Space–Time Model IR (primary)

Given a typed neural-network Functional DAG, independently chosen per-operator spatial/temporal mappings \(\Phi_v\), and explicit tensor identities, the IR should derive **actual logical-element** obligations across operator boundaries:

1. required producer reduction contributions and their completeness before COMMIT;
2. source value READY/availability and consumer-specific logical READ/access maps;
3. multiple consumer READs and final-reader RELEASE, not a single-consumer shortcut;
4. source-level physical scratch addresses/banks and safe aliasing only after last use;
5. coherent relationship between \(\Phi_v\), actual emitted C++ access order and independently checked reference semantics.

The scientific question is whether **mapping-conditioned tensor-element readiness/lifetime/address refinement** forms a distinct, usable compositional contract, not simply a new name for classical liveness, type checking or schedule/storage optimization. **A changed \(\Phi\) hash or different C++ SHA is not sufficient causal evidence.** Inspect if the actual source event traces and candidate legality change for the right reason.

### C2 — Certificate-backed source refinement and memory planning (primary)

The implementation proposes finite, independently checkable certificates/counterexamples covering reduction completion, ownership, ordering, reader-specific accesses, last reader, source storage reuse and generated-source conformance. Check that:

- The *formal proposition and its assumptions* can be stated without circularly assuming a correct compiler.
- Independent verifier and generator do not unknowingly share the same flawed oracle.
- “Proof”, “certificate”, “exact optimum”, “safe”, “independent” and “sound” are used only for the stated, **bounded** schedule, event and source-domain models.
- Runtime evidence is from actual GCC-executed reads/writes/numerical calculations where claimed, not only a generator replay.
- Interval coloring or simple minimal live slots is **prior art**; any novelty lies in the explicit mapping-conditioned, compositional certificate/refinement and validated consequences, not rediscovery of the classical interval result.
- E0/E1 source semantics are not promoted to RTL correctness, RAM-port feasibility, clean physical timing, hardware Fmax or guaranteed resource savings.

### C3 — Policy-independent, RL-/search-compatible certified Model IR (secondary; MUST BE REVIEWED)

The research explicitly proposes the **same underlying typed mapping-conditioned compiler contract** as an optimizer-independent state/action/observation and legality interface, usable by a human, enumeration, heuristic, Bayesian search, or a future RL/GNN policy:

`S = (FunctionalGraph, target_constraints, local_Phi, tensor_exchange, storage_schedule, certificate_status)`

`a = (scope, target, field, candidate_value)`

`transition(S,a) -> Accepted(new_state, source_certificate) | Rejected(old_state, counterexample)`

`certified_mask(S, candidate_actions) -> BOUNDED_CERTIFIED | REJECTED | UNKNOWN`

`observation(S) -> versioned variable-size operator/tensor/edge graph, no raw weights or leaked post-HLS outcomes`

The compiler owns legality, source lowering, provenance and certificates; **the RL agent owns its policy, experience, reward, optimization objective and training**. The IR must not implement an agent-specific bypass. An UNKNOWN action is **not** a certified legal action. A SOURCE-admissible action is **not** automatically bounded-certificated, HLS feasible or routed. There is no claim of a novel RL algorithm, trained RL superiority, better rewards, transfer learning, sample efficiency, or hardware-optimal action mask.

**Review C3 critically:**
- Are immutable source state, typed action, compiler transition, source admissibility mask, conservative three-way certificate mask, graph observation and provenance genuinely implemented?
- Does the candidate action mask call *the same real source compiler/checker*, or a lossy independent proxy?
- Is C3 coupled to **C1/C2's full multi-reader \(\Phi\)-conditioned tensor-element certificate**? Or does the implemented decision path remain limited to static complete-design row chains? State the exact supported domain.
- Do graph renaming tests establish only bounded name-insensitive features, rather than universal graph permutation equivariance or generalization?
- Are all fields an RL algorithm needs—including constraints and available devices—actually present? Is physical feedback kept separate and source-pinned?
- Does C3 qualify as **(A) an independently defensible secondary scientific contribution, (B) a useful engineering/design property, or (C) a future paper requiring new evidence?** Justify the classification and identify the smallest source-only experiment that would make it stronger.
- How does this differ substantively from CompilerGym, RL4ReAl, IronMan, DRiLLS, CASCADE or known design-space/action-mask work? An RL-friendly API, GNN input or Gym wrapper is **not automatically novel**.

**Do not train agents or unfreeze `rl_environment/`.** The prior online post-HLS RL research is intentionally frozen while the compiler-IR paper is assessed. Improved RL sample efficiency on unseen networks is an **unexecuted future R5 empirical hypothesis**, not evidence in this paper.

## 3. Actual reported original experimental results — audit, do not assume

The user reports an existing laptop checkout (local code and preserved licensed-tool `build/` data) and the following **verified-by-user-workstation** outputs. You may treat the displayed logs as reported observations, but do **not** claim to have personally validated raw Vivado/HLS files if those files are not available in your review environment.

**R2h source/E1:** 72 original GCC compiled/executed programs, 36 matched disjoint-vs-optimized scratch pairs. The reported 36/36 matched source cases reduce **C++ source scratch scalar slots**, with numerical and address/refinement checks in the supported experimental domain. This does **not** directly mean reduced actual FPGA BRAM.

**R2i native E2/E3:** Original source revision `b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`. 72/72 actual Vitis HLS CSim E2-qualified cases; 48/72 CSynth E3-qualified; **24/36 matched E3 pairs**, **12/36 E3 unknown**, not zeros or ties. Within the 24 qualified pairs: **six HLS-estimated BRAM_18K savings, eighteen ties, zero HLS-estimated BRAM regressions**, and reported HLS worst-case cycle improvements. Check that the paper never substitutes E3 estimates for real routed utilization or achieved Fmax.

**R2j original routed pair20 + R2l prospective five-pair follow-up + R2m paper exporter:** Six prespecified original matched HLS-positive indices **20, 23, 28, 31, 32, 35**. Original pair20 reused; ten *new* original Vivado OOC calls produced the five additional pairs. All twelve source-attested routed arms completed routing and met the **requested clock**, but disclosed DRC warnings. **Zero strict clean-DRC E4 matched pairs.** Other 30/36 physical pairs unobserved.

| Pair | Cases | HLS-estimated BRAM_18K saved | Routed BRAM tiles disjoint → optimized | Routed LUT | Routed FF | Routed DSP | OOC WNS ns |
|---|---|---:|---|---|---|---|---|
| 20 | 040/041 | 1 | 1.5 → 1.5 | 649 → 684 | 396 → 407 | 7 → 7 | +26.356 → +27.304 |
| 23 | 046/047 | 1 | 1.5 → 1.5 | 656 → 693 | 396 → 407 | 7 → 7 | +26.941 → +27.008 |
| 28 | 056/057 | 1 | 1.5 → 1.5 | 870 → 889 | 501 → 503 | 8 → 8 | +25.491 → +27.133 |
| 31 | 062/063 | 1 | 1.5 → 1.5 | 878 → 819 | 497 → 494 | 8 → 8 | +26.359 → +27.054 |
| 32 | 064/065 | 1 | 1.5 → 1.5 | 883 → 880 | 504 → 489 | 8 → 8 | +27.205 → +26.863 |
| 35 | 070/071 | 1 | 1.5 → 1.5 | 869 → 858 | 504 → 499 | 8 → 8 | +27.119 → +26.121 |

**Six out of six selected routed BRAM ties**, **six DSP ties**, zero routed BRAM savings/regressions; **LUT increases in 3/6 and decreases in 3/6** (mean optimized minus disjoint +3 LUT); **FF increases in 3/6 and decreases in 3/6** (mean +0.167 FF); **WNS improves in 4/6 and worsens in 2/6** (mean optimized minus disjoint +0.335333 ns, median +0.381 ns). These are **descriptive warning-disclosed OOC results**, *not strict E4*, achieved Fmax, throughput, power or a statistically representative estimate of all FPGAs.

**R2m v2:** Latest workstation log shows **11/11 focused tests** and **1,447/1,447 full Python pytest tests** passed, followed by source-only final table export to `build/r2m_original_six_physical_paper_v2/` with **zero additional vendor calls during that exporter invocation**. Output includes `results.json`, `all_36_pairs.csv`, `paper_table.md`, with original Vivado report digests and DRC rule populations. Original experiment artifacts and v1/v2 exports must not be overwritten.

### Why the negative physical result matters

Our candidate finding is **not** “we save physical BRAM.” Instead it is: in this **selected six-pair HLS-positive cohort**, the observed source scratch savings and estimated HLS BRAM reductions **did not translate into routed BRAM tile reduction**. Ask if this is a publishable insight about abstraction/evidence separation, a useful negative result requiring more explanation, or an underpowered selected example. Can you distinguish physical memory granularity, HLS binding, other fixed overhead, DRC warnings and implementation effects *without inventing a root cause*?

### Non-negotiable evidence tiers

- **E0** — bounded IR/event/source-level semantics/certificates, explicitly scoped.
- **E1** — actual generated C++ and independent GCC-executed numerical/trace/address evidence for specified cases; not all E1 traces are equally independent.
- **E2** — original fixed-point Vitis CSim and source/numeric qualification.
- **E3** — original Vitis CSynth source-bound resource/cycle/estimated timing; unknowns preserved.
- **E4 strict** — source-attested fully routed, timing-met and **DRC-clean** Vivado designs. Warning-only OOC routed observations are **descriptive**, NOT E4.
- **Board bitstream, achieved maximum Fmax, real power, latency on hardware, RTL proof and trained RL gains** are **not established** by the six-pair study.

**Please audit denominator consistency and the statement that the six physical cases were preselected by positive HLS effects (selection bias).** Do not extrapolate to 36 physical ties or all neural networks.

## 4. Priority reading map — code and history, not just the manuscript

Read in this order and use specific file/line or commit references in your verdict.

### Paper and criticism already recorded

1. `paper/compiler_ir/MANUSCRIPT_DRAFT.md` — especially abstract, candidate C1/C2/C3, formal definitions, experimental method, **actual R2i/R2m quantitative results**, limitations and conclusion. The file contains dated *historical* notes that may be stale; find contradictory statements.
2. `paper/compiler_ir/NOVELTY_THESIS_AND_PROOF_OBLIGATIONS.md`; `PAPER_CONTRIBUTIONS_AND_RL_INTERFACE.md`; `COMPOSITIONAL_SPACETIME_IR_NOVELTY.md`.
3. `paper/compiler_ir/RL_READY_IR_NOVELTY_AND_EXPERIMENTS.md`; `CERTIFIED_DECISION_IR_SPRINTS.md`. **Do not rely only on the early “UNIMPLEMENTED” sprint table**: later appended records document additional bounded implementations. Reconcile claims to actual code.
4. `paper/compiler_ir/CLAUDE_REVIEW_ACTION_REGISTER_20261103.md` — **a prior Claude review** and F1–F9 remediation register. Identify which earlier issues are genuinely resolved by subsequent original experiments, which remain partial or unsupported. Historical verdicts are not automatically today's verdicts.
5. `paper/compiler_ir/CURRENT_SCIENTIFIC_STATUS.md`; `PUBLICATION_READINESS.md`; `REPRODUCIBILITY_LEDGER.md`; `R2M_FINAL_ORIGINAL_PHYSICAL_PAPER_EXPORT_20261103.md`; `ORIGINAL_SIX_PAIR_PHYSICAL_RESULTS_20261103.md`.
6. `paper/compiler_ir/PRIOR_ART_NOVELTY_GATE.md`; `CST_FOUNDATIONAL_STORAGE_PRIOR_ART.md`; `PRIOR_ART_PINNED_ARTIFACTS.md`.

### Implementation and experiments to challenge

- `src/fpgai_research/model_ir/decision_interface.py` — immutable full state; action; `transition`; `action_mask`; `certified_action_mask`; `graph_observation`. Examine **actual semantic coverage** and UNKNOWN handling.
- `src/fpgai_research/model_ir/{contracts,semantics,composition,phi_refinement,mapped_value_contract,mapped_event_schedule,fanout_lifetime,fanout_allocator,fanout_phi_source,matmul_phi_order,branch_matmul_contract,branch_exact_allocator,graph_access_maps,graph_phi_contract,graph_edge_witness,element_scratch_allocation,independent_phi_issue_reference}.py`. Inspect current fanout/reader/producer source-order checking against the paper's formal story.
- `src/fpgai_research/compiler/cst_pipeline.py`; `src/fpgai_research/codegen/hls/`; `src/fpgai_research/validation/`. Source evidence must be tested against **real generated C++** and independent mathematical operations, not just IR hashes.
- `src/fpgai_research/experiments/{cst_decision_campaign,cst_rl_interface_study,cst_phi_counterfactual_effects,cst_r2i_fanout_native_effects,cst_r2l_physical_completion,cst_r2m_original_physical_paper}.py` and corresponding `tests/test_cst_*`.
- Compiler-IR series `src/fpgai_research/experiments/compiler_ir/series.py` and original source-pinned receipt protocols. **No fresh vendor execution or deletion** is needed for this review.

### Git commit history: verify development and experimental order

Use `git log --oneline --all --decorate`, `git log -- <path>`, `git show <commit>` on the checked-out repository. We specifically want to know whether key hypotheses and negative controls **preceded** evaluation, rather than were fitted post-hoc.

Relevant prior Git commit milestones (verify content; titles are not evidence):
- `f7e785e2817d` — policy-neutral complete-state action interface.
- `c356e4929c27` — weight-private, variable-size graph observations.
- `7067dd9c89e0` — three-valued conservative certified mask.
- `901214cfb6d7` — action mask bound to bounded mapped-value E0 event verification.
- `ec01472a9c5c` — separate source-only R3/R4 RL-interface study.
- `b2d5dcc4f18fad7cb1cd2c342101dd33185e817f` — original R2i source revision for 72-case experiment.
- `f64766fdd807855a53ffc63b9f8d18b400238fd2` — pre-review current baseline with the verified R2m-v2 publication text.

**No GitHub Actions or tokens**: do not inspect, invoke, enable or troubleshoot Actions. Original CPU tests and licensed tool outcomes were performed in an existing local workstation clone; if you do not possess the original `build/` reports, you must clearly label those numbers **“reported by author, raw artifacts not independently inspected”**.

## 5. Required adversarial tests and hypotheses to challenge

Your critique must specifically address:

1. **Originality / prior art.** Polyhedral/affine schedule-storage optimization, classic storage allocation, interval coloring, Allo, HIDA, StreamTensor, ST-Flow, Wavelet/Calyx, CompilerGym and RL4ReAl already cover parts of this terrain. Identify exact overlap; suggest a truthful narrower contribution if needed. Do **not** assume absence of competitor features from README-only inspection.
2. **Mapping causality.** Holding Functional graph, input/weights, numeric model and scratch request fixed, does changing only a supported reader \(\Phi\) cause an independently predicted change in **actual** READ/last-use/alias legality? Or just hash/source shape differences?
3. **Reduction completeness.** Are values COMMITted only after every contributing reduction is complete? Do missing/duplicate/wrong-owner contributions fail with independently useful counterexamples? Is this true for two independent readers?
4. **Last use and overwrite.** Can the producer reuse an address before a second reader has completed? Do actual emitted C++ reads reflect the certificate rather than a generator-provided schedule replay?
5. **Soundness / proof gaps.** Which invariants are actually proved mathematically, which are bounded exhaustive tests, which are empirical sample checks, and which remain merely assertions? Could the verifier and source emitter be wrong in a common way?
6. **Memory objective confusion.** Distinguish scratch scalar slots, live intervals, mutable accumulator arrays, source C++ buffer declarations, HLS estimated BRAM_18K, Vivado routed BRAM tiles, LUT, FF and DSP. Where can a paper claim exact optimality, and where can it not?
7. **Numeric model.** What does host floating-point checking establish compared with actual fixed-point Vitis CSim? Were exact source transformations independently tested under the claimed arithmetic assumptions?
8. **Hardware physics.** Zero strict clean-DRC E4; what is a legitimate descriptive observation? Can any WNS change be called improved frequency? Could DRC warnings or OOC methodology bias interpretation?
9. **Experimental science.** Selection on six positive HLS outcomes; finite benchmark variety; 12 undefined E3 pairs; 30 unobserved physical pairs; any p-hacking, post-hoc hypothesis modification, weak baselines or cherry-picked cohorts.
10. **RL interface independence.** If two proposers offer the same action, is certificate equality a meaningful check or nearly tautological? Are independently legal masks derived from C1/C2 or just repeated compilation? Does UNKNOWN exclude uncertified actions? How to test causal action legality without training an RL agent?
11. **C3 limits.** No trained RL performance, cross-graph transfer, reward advantage, or hardware feasibility should appear as achieved results. Do current graph features provide relevant constraints and prevent post-HLS leakage? Is there a partial-state/STOP semantic model? If absent, say so.
12. **Generality.** Does the paper extrapolate from static MatMul/fanout to arbitrary Conv/Attention, dynamic shapes, asynchronous clocks, generic hardware devices or physical bank-port legality?
13. **Writing accuracy.** Identify contradictions between top chronological notes and main manuscript body, outdated “no native” statements, unsupported firstness/superiority, and mismatched terminology.
14. **Formal contribution.** Propose one or two *clean theorem-level statements* with precise assumptions and a falsification example. Clearly differentiate the mathematical theorem from independently checked finite implementations and vendor observations.
15. **Scientific adequacy without competing compiler execution.** Is a well-scoped semantics + independent verification + original HLS/physical results study publishable? If not, identify which *specific scientific question* cannot be answered without executing an external comparator, rather than automatically demanding a leaderboard.

## 6. Requested review deliverable (strict format)

Produce a substantial independent report with the following sections and evidence citations:

1. **Overall verdict** — `PUBLISHABLE_NOW / MAJOR_REVISION / NOT_YET_SCIENTIFICALLY_DISTINCT`; level of confidence and the most realistic target venue/type. Do not guess acceptance probabilities.
2. **Contribution verdict table** — C1, C2, **C3 RL-compatible IR**, and negative HLS-to-physical translation result. For each: exact claimed novelty, implemented evidence, known related overlap, strongest counterexample, publishable wording, and current pass/partial/fail status.
3. **Claim-by-claim forensic review** — quote or point to exact manuscript sections with `file:line` references. Classify each nontrivial claim as `SUPPORTED / PARTIAL / UNSUPPORTED / FALSE / UNKNOWN`, state what evidence would fix it.
4. **Hypotheses and formal semantics audit** — identify unproved assumptions, circularity, invalid inference, need for new proof/lemma, and explicit failing examples.
5. **Experimental-methods audit** — E0/E1/E2/E3/E4 distinctions, all denominators, source/clock/numeric parity, fixed selection of six, actual BRAM/LUT/FF/DSP/WNS values, DRC warnings and negative results.
6. **RL-compatible IR independent assessment** — assess R1–R4 implementation, C3's dependence on full C1/C2 semantic relation, and whether R5 is needed to substantiate *interface* versus *learned improvement* claims. Propose exact, reproducible minimal tests with pass/fail criteria and no RL training.
7. **Related-work novelty matrix** — exact prior mechanisms and what remains genuinely different; separate known facts from paper/artifact questions not yet answered. A direct external compiler speed benchmark is not mandatory unless a particular claim requires it.
8. **Top 10 prioritized blockers** — severity `FATAL/MAJOR/MINOR`, referenced evidence, specific remediation, estimated scope `short/medium/large`, whether new original GCC/Vitis/Vivado evidence is necessary. No invented runtime estimates.
9. **Minimal publishability plan** — ranked **three to six** concrete new experiments/proofs/edits that would most improve scientific value while preserving all frozen original data. Include clear kill conditions: what result would force us to abandon or narrow the novelty thesis.
10. **Proposed precise paper title, abstract, contributions, and conservative claims** — draft only after your critical analysis; retain RL suitability as a secondary, carefully qualified contribution, and include actual physical resource and WNS results with proper evidence-tier labels.
11. **Missing evidence** — exact file/receipt or local `build/` path needed for facts you could not independently verify. Do not pretend to see unprovided workstation artifacts.
12. **Final scientific decision** — one paragraph: “If this were my submission, I would / would not submit it yet, because …”

If all you can inspect is documentation, state **“documentation-only review”** and do not pretend the source code or artifacts were audited. Prioritize correctness over positivity. Do not ask to enable GitHub Actions or supply tokens.

## 7. Read-only local commands (optional; no Actions)

From an already checked-out repository, **do not clone or reset anything just for the review**:

```bash
cd ~/Desktop/github_projects/fpgai_research
git status --short
git log --oneline -n 120
git log --oneline -- src/fpgai_research/model_ir/decision_interface.py
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_rl_interface_study.py tests/test_cst_r2m_original_physical_paper.py
```

The local test commands are optional if a suitable Python environment exists; they use **no licensed vendor calls**. The author has already run the complete **1,447-test** local suite and R2m-v2 export; there is no need to rerun Vivado, Vitis, frozen original GCC experiments, or `rl_environment/` studies merely to perform an independent review.

**End of independent review request.**
