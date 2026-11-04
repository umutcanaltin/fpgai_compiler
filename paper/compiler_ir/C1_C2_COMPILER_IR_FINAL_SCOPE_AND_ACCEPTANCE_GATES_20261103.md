# C1 + C2 research-paper scope and falsifiable acceptance gates

**Decision: 3 November 2026.** This manuscript has two and only two primary proposed scientific contributions. Trained RL agents, reward design, learning curves, sample efficiency of learned policies, and closed-loop native-HLS learning belong to a **separate later paper**. The current `rl_environment/` is frozen. No external paper or artifact has accepted or validated these claims.

> **Workstation-verified C1/C2 original checkpoint, source `fc40369f9493d93cc78c4886ae2e32c5d1f64093`:** 42/42 focused + **1,705/1,705** full regression; R3d **3 E0-certified, 2 rejected, 1 UNKNOWN** of 6; R3e **5/5 finite oracle agreements and 25/25 selected semantic mutants detected**; R3f **21 E0-certified, 24 rejected, 3 UNKNOWN** of 48. All three original receipts were read-only audited with 0 vendor calls; folders are frozen. [Full audit checkpoint](R3D_R3E_R3F_ORIGINAL_C1_C2_WORKSTATION_RESULTS_20261104.md). R3g systematic independent event oracle and R3h compiled-once C2 catalog were subsequently **workstation-verified**: 133 source event trace variants/0 oracle disagreements and 48 source compiler actions/0 additional cached selection compiles at source revision `3309076814714e61906363e21e3d50b932519f9d`; 1,751/1,751 full pytest passed. [Separate frozen results](R3G_R3H_ORIGINAL_C1_C2_WORKSTATION_RESULTS_20261104.md). The **new R3j, R3k and R3l** studies are coded but **not yet workstation-executed**. No improved trained RL, no E3 physical proof and no external novelty comparison.

## Final intended research question

Can one bounded, compositional hardware Model IR connect **operator-local architecture `Phi`**, typed tensor-element production/read/last-use contracts, and original source identity, while exposing the *same real compiler legality relation* to any search proposer as a weight-private graph observation and explicit `REJECTED / UNKNOWN / CERTIFIED` action result?

This is a narrower question than "we invented typed hardware IR" or "our system beats reinforcement learning environments." The latter would require a trained, equal-budget agent study and is **out of scope**.

## Formal bounded semantics (claimed only where supported)

For a complete model state `S = (F, A, X, L)`:

- `F`: typed static Functional Model IR with named operators/tensors and model constants.
- `A`: complete operator-local architecture requests (PE/SIMD/etc., where each bounded domain supports them).
- `X`: supported tensor exchange and scratch layout policy, with source-level temporal event order.
- `L`: target/source admissibility limits (not native HLS physical feasibility).
- Typed action `a` changes one supported parameter. The transition `T(S,a)` must use the **ordinary compiler source emitter and contract verifier**. Unsupported, unchanged, ill-typed, or nonadmissible actions are rejected without mutating `S`.
- `Cert_E0(S,a)` is an explicitly *bounded* event/value/ownership certificate. `UNKNOWN` means no accepted E0 proof and must never be treated as proven safe.
- A finite `M_E0(S, C)` mask for a specified candidate catalog `C` is true **only** for E0-certified transitions. This is *not* an HLS timing, resource, numeric, or physical DRC mask.

C1's central finite source-event invariant is:

1. No tensor element is read before all its producer terms are complete and committed.
2. A scratch slot cannot be overwritten while any required consumer still needs the old value; the last-reader event releases it.
3. Per-operator `Phi` metadata for the supported source iteration domain matches the intended producer/consumer issue order; a source-safe order need not match the required `Phi` order.
4. Every certified source candidate carries model/state/source event and emitted C++ hashes, with original CSim/HLS and physical evidence **separately** tiered.

The finite interval packing bound is for **one fixed serialized source event order**, not simultaneous physical RAM-bank ports, pipeline concurrency, clock crossings, arbitrary DAGs or general hardware safety. The present implementations cover specific static MatMul row chains and 2–4-reader MatMul fanout with Add chain/Relu; CNN/Dense source integration is validated separately, but **the same C1/C2 decision-contract coverage is not yet demonstrated for all complete CNN topologies**. Do not conflate these paths.

## Implemented research design and original evidence

| Gate | Existing original evidence / newly prepared experiment | What it can validate | Not a valid conclusion |
|---|---|---|---|
| C1-E0 finite event contracts | Existing R2f/R2r counterexample/source mutations, bounded event verifiers | Readiness, all-reader lifetime and `Phi` issue rejection in specified source models | General formal RTL proof |
| C1-SLOT independent reference | Existing independent exhaustive interval coloring; **new R3e** five predeclared tiny source cases, five distinct adverse event/alias mutations each | Fixed-order exact slot bound on an exhaustively enumerable subset, typed mutant rejection | Optimal physical memory layout |
| C1-E1 actual source | Frozen original R2g/R2h/R2o source GCC numeric and address witnesses | Selected actual emitted C++ read/write conformance | All-input bit-exactness |
| C2 deterministic API | Existing R2n/R2o/R2p typed source decisions, complete action patches, SHA receipts | Same compiler oracle for search and engineering | Trained agent improvement |
| C2 mask vs proposal | **New R3d** exact three actions in each of two supported domains; blind/catalog source acceptance, unknown E0, certified E0; real compiler replay | Count invalid/unknown/provable actions and mask cost, without switching compilers | Faster optimization or cheaper overall search |
| C2 graph observation | Existing R2p typed weight-private graph plus **new R3d** rename/order-stable bounded structural fingerprint | Feature invariance to node/edge ID changes and list ordering on tested graphs | General graph isomorphism or proven GNN transfer |
| CNN native evidence | Frozen original R3a and verified R3c read-only export: six E2, four E3, two strict paired E3 | Fully materialized mixed-network compilation and negative HLS comparison | Better E3/E4 hardware or improved RL |
| Prior art | Public related-work matrix (Allo, StreamTensor, ST-Flow, CODO, Filament, hls4ml) | Exact overlapping scientific mechanisms to investigate | An exclusive-newness finding |

**Important:** The R3d/R3e source code and tests were added to GitHub `main` on 3 November 2026, but **have not yet been executed on the author's workstation**. The five cases and twenty-five mutants are planned test denominators, not measured success counts. All prior original native/E1 source receipts remain untouched.

## Required falsification and acceptance criteria

**C1 verification gate:** all five R3e predeclared tiny cases must agree with a **separate exhaustive minimum-color oracle** for unit-slot scratch, the five chosen adversarial mutations per case must be rejected for the *expected specific reason*, and the alternative `Phi` issue order must be rejected **even when the altered source event order remains otherwise readiness-safe**. Original selected GCC source witnesses and model/CPP SHA must be referenced separately. Any mutation that unexpectedly passes is a failed gate, not deleted from the denominator. This is a **bounded experimental check**, not a general mathematical proof.

**C2 verification gate:** on exactly the same action catalog in each domain, `unified_action_mask` and direct `unified_transition` results must match case-for-case; **UNKNOWN is never legal**, rejected changes leave the original state unchanged, and all compiler calls spent computing the mask are counted. The restricted, hand-specified six expected outcomes do **not** constitute an independent general legality oracle. The graph feature fingerprint must be stable under ID/list permutations and sensitive to changed typed tensor dimensions in the prespecified examples. Timing overhead is recorded as raw CPU measurements, not a claimed statistical benefit. No train/test leakage or learned optimization statement is possible because **no agents are trained**.

**External novelty gate (not completed):** independently inspect and pin an actual competing implementation, preferably Allo and then StreamTensor/CODO or ST-Flow. Prepare identical bounded **legal and illegal** operator-memory source designs, specify semantics and matching precision/hardware controls, and record whether the rival expresses/checks the *same particular invariant*. "Our tool has a certificate" is not by itself a proof rivals lack an equivalent representation. Unsupported/missing features must be documented as observations for a **specific pinned artifact**.

**Causal coverage gate (implementation added, not workstation-verified):** R3f now covers **six source graph cohorts, 8 actions each = 48 prespecified complete-design proposals**, with exact mask-vs-blind compiler replay, per-domain status counts, explicit screening cost and SHA-linked graph observations. This is an additional fixed catalog, **not exhaustive all-architecture legality**; broader systematic action grids, independent semantic oracles and held-out source graph families are still required. Independently compute confusion/coverage/false-positive denominators, E0 UNKNOWN rates, source-compiler CPU cost and complete action-space size. Show which difference results from contracts rather than renamed API fields or different compiler backends. Assess observational equivalence/renaming across genuine independently generated models, not only synthetic relabeling of one feature graph.

**Generalizability gate (not completed):** demonstrate the same C1/C2 contract mechanisms on a useful held-out or mixed CNN/NN source family, *or explicitly narrow the final paper claim* to static MatMul chain/fanout design spaces. A CNN CSim/HLS test without a C1/C2 certificate does not close this gate.

**Editorial gate (not completed):** replace obsolete prospective text in manuscript sections 5.12 and 5.13 with verified R3c results; write a formal bounded method/proof statement, explicit counterexamples, causal ablation and uncertainty; include source revisions, toolchains, exact population denominators, original failures, related work, release/IP review and independent scientific reviewer feedback.

## Historical R3g/R3h original validation commands — already completed

R3d, R3e, R3f, R3g and R3h **have already run**. The following R3g/R3h commands are **archived for reproducibility**; do **not** reissue their create-only `v1` commands into existing folders. For the next run, use the **R3j/R3k/R3l** section below instead:

```bash
python -m pytest -q \
  tests/test_cst_r3g_c1_independent_event_semantics.py \
  tests/test_cst_compiled_action_catalog.py \
  tests/test_cst_r3h_c2_compiled_once_selection.py
python -m pytest -q

python -m fpgai_research.experiments.cst_r3g_c1_independent_event_semantics \
  --output build/r3g_c1_independent_event_oracle_v1
python -m fpgai_research.experiments.cst_r3g_c1_independent_event_semantics \
  --study build/r3g_c1_independent_event_oracle_v1

python -m fpgai_research.experiments.cst_r3h_c2_compiled_once_selection \
  --output build/r3h_c2_compiled_once_selection_v1
python -m fpgai_research.experiments.cst_r3h_c2_compiled_once_selection \
  --study build/r3h_c2_compiled_once_selection_v1
```

The completed R3g/R3h scripts were **source-only and create-only**. No Vitis/Vivado and no RL training occurred. The R3h masked action catalog still performs one source compiler transition for every candidate; cached selection is an API convenience, not proof of total search-speed improvement.

## Reproducibility commands: source-only new studies, NO agent or vendor execution

From the **existing** `fpgai_research` checkout after pulling `main`:

```bash
python -m pytest -q \
  tests/test_cst_r3d_c1c2_finite_contract_ablation.py \
  tests/test_cst_r3e_c1_fanout_counterexamples.py
python -m pytest -q

# New deterministic, create-only C1 and C2 evidence. Fresh output paths only.
python -m fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation \
  --output build/r3d_c1c2_finite_contract_gate_v1
python -m fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation \
  --study build/r3d_c1c2_finite_contract_gate_v1
python -m fpgai_research.experiments.cst_r3e_c1_fanout_counterexamples \
  --output build/r3e_c1_bounded_counterexamples_v1
python -m fpgai_research.experiments.cst_r3e_c1_fanout_counterexamples \
  --study build/r3e_c1_bounded_counterexamples_v1
python -m fpgai_research.experiments.cst_r3f_c2_variable_graph_coverage \
  --output build/r3f_c2_variable_graph_coverage_v1
python -m fpgai_research.experiments.cst_r3f_c2_variable_graph_coverage \
  --study build/r3f_c2_variable_graph_coverage_v1
```

Each code path uses the source compiler only; **zero original Vitis/Vivado, trained RL, or new GCC experiment invocations**. The R3d masks compile candidate sources as part of E0 checks; their cost is not zero. Do not overwrite an existing output or rerun the frozen R3a/Vitis cohort.

## R3i isolated prior-art capability probe — optional independent toolchain

The new pinned `R3i` Allo source probe is prepared but **not executed**. The pure research-suite tests guard opt-in, exact checkout SHA, numerical negative reference and output preservation. They do **not** import or run Allo in the normal `fpgai_research` environment. Read its protocol at any time without the competitor toolchain:

```bash
python -m fpgai_research.experiments.cst_r3i_external_allo_capability --protocol
```

For the separately isolated actual external test, use the independent [pinned artifact runbook](C1_C2_PINNED_PRIOR_ART_RECONNAISSANCE_20261103.md) and a different toolchain environment. A missing PAST dependency or failed source transform remains **UNKNOWN**, not a negative Allo capability outcome. This probe compares whole-output source equivalence, **not** the same typed E0 source-event proof, so full external novelty parity remains open.

## Next source-only experiments and cross-cohort release (4 November 2026)

**R3j C1 held-out semantics:** 24 predeclared factor-combinations of 2–4 readers, four distinct small MatMul shapes and two independent `Phi` mapping policies. Nine source traces per profile (one accepted, one source-safe/issue-invalid, seven source-unsafe): **216 planned finite reference comparisons**. The independent oracle and ordinary source event checker must agree for each. **This does not expand the supported graph topology** beyond the static fanout family. Results must not be cited until executed.

**R3k C2 source target-limit intervention:** 12 matched source model pairs, identical functional graph and identical eight typed candidate actions, with `max_parallel_units=1` versus `4` as the sole intervention; **192 planned source action decisions**, plus equal direct compiler replays. It tests legality sensitivity to source bounds, not HLS hardware resources. No trained agents and no vendor calls.

**R3l read-only evidence release:** aggregating **only existing** original R3c (CNN E2/E3) and R3d–R3h source reports, retaining each original source SHA and external-competitor `NOT_ESTABLISHED` status. It validates all saved SHA digests, refuses imputed failed-pair E3 numbers and writes a **new create-only** derived publication table with `submission_ready=false`. Optional R3j/R3k can be included only if their original folders exist. Source data and existing saved receipts are not modified.

From your repository, pull `main` and run the focused suite and full regression **before** the new source studies:

```bash
python -m compileall -q src/fpgai_research
python -m pytest -q \
  tests/test_cst_r3j_c1_heldout_semantics.py \
  tests/test_cst_r3k_c2_target_limit_intervention.py \
  tests/test_cst_r3l_c1c2_publication_release.py
python -m pytest -q
```

Create **new** original outputs, never overwrite the older R3c–R3h data:

```bash
python -m fpgai_research.experiments.cst_r3j_c1_heldout_semantics \
  --output build/r3j_c1_heldout_semantics_v1
python -m fpgai_research.experiments.cst_r3j_c1_heldout_semantics \
  --study build/r3j_c1_heldout_semantics_v1

python -m fpgai_research.experiments.cst_r3k_c2_target_limit_intervention \
  --output build/r3k_c2_target_limit_intervention_v1
python -m fpgai_research.experiments.cst_r3k_c2_target_limit_intervention \
  --study build/r3k_c2_target_limit_intervention_v1

# Read-only release, after all studies; only derived output is created.
python -m fpgai_research.experiments.cst_r3l_c1c2_publication_release \
  --build build \
  --include-optional \
  --output build/r3l_c1c2_publication_release_v1
```

Do **not** run R3l with `--include-optional` before R3j/R3k finish if you intend to report them as verified; omitted optional records remain explicitly `UNEXECUTED` rather than fabricated data. The release always says **MAJOR REVISION** while the independent external baseline and substantive novelty review are open. No original Vitis/Vivado rerun or RL training is required for these source-only steps.

The [isolated pinned Allo external comparison runbook](R3I_ISOLATED_ALLO_CAPABILITY_RUNBOOK_20261104.md) documents exact artifact source/CPython 3.12/PAST constraints and the approved narrow source-equivalence probe. **Its actual comparative results remain NOT_RUN.**

## Separate later agent paper

The future paper may train shared GNN/actor–critic policies on the exact C2 feature/mask API with identical seeds, budgets and held-out graphs, comparing source-certified vs ordinary states, masked vs unmasked actions, and native HLS-qualified latencies/costs. Those are **unmeasured claims** and must not be smuggled into this compiler IR manuscript.

**Current decision: MAJOR REVISION / NOT SUBMISSION-READY** until new C1/C2 workstation gates, independent prior-art comparison, broader action/graph coverage and manuscript reconstruction are finished. A passing test suite is implementation confidence, not peer-reviewed scientific priority.
