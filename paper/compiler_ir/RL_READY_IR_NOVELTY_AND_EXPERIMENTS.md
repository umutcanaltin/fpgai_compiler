# RL-ready, certificate-backed CST Model IR — design and scientific hypothesis

**Date:** 2026-11-02. **Status:** original design contract plus later bounded implemented source-level interface; no trained RL or experimentally established RL superiority. **Location:** compiler research IR, `fpgai_research`. The existing `rl_environment/` and its L0 original-Vitis study remain **frozen and unchanged**.

## Scientific position

The primary compiler-paper contribution remains **mapping-conditioned tensor-value production, readiness, last-use and safe physical-address refinement certificates for independently mapped operators**; see [NOVELTY_THESIS_AND_PROOF_OBLIGATIONS.md](NOVELTY_THESIS_AND_PROOF_OBLIGATIONS.md).

**Second research hypothesis (RL-compatible, not RL-exclusive):**

> A compositional FPGA Model IR that exposes its mapping-conditioned legality constraints as a *policy-independent, certificate-producing transition system* can be used by a learning agent, classical search or a human editor through the **same legal action and state interface**, while preserving functional/numerical semantics and enabling a shared policy to transfer across variable-size network graphs.

The candidate advance is **one operational legality/certificate semantics driving compiler composition, legal action generation, safe state transitions and training observations**. An RL/GNN input tensor or Gym wrapper by itself does not qualify as novelty.

### Closest learning-based prior art to disprove/compare against

- **CompilerGym (CGO 2022):** compiler RL environments already expose actions, graph/feature observations and reward spaces; `RL-ready IR` is not new. https://github.com/facebookresearch/CompilerGym
- **IronMan (GLSVLSI 2021):** GNN + RL multi-objective HLS design-space exploration and C++/directive transformation. https://doi.org/10.1145/3453688.3461495
- **RL4ReAl (CC 2023):** semantics-preserving compiler RL with constrained legal action spaces and modular compiler-agent interface. https://doi.org/10.1145/3578360.3580273
- **CASCADE (ISCAS 2026):** MLIR compiler with RL-guided logic synthesis exploration across HLS/logic transformations. https://doi.org/10.1109/ISCAS66217.2026.11561911
- **DRiLLS (ASP-DAC 2020):** RL logic-synthesis optimization sequences. https://doi.org/10.1109/ASP-DAC47756.2020.9045559

**Kill condition:** if the joint compiler safety relation can already be expressed, checked and used to restrict graph-wide learning actions equivalently in a prior-art system, novelty must be narrowed; merely offering observations/action masks does not justify a claim of firstness.

## Architectural separation — do NOT mix the IR with an RL policy

The compiler owns:
1. **Canonical immutable typed IR**: functional graph, tensor identities/shapes, operator-local `Phi_v`, tensor-edge completion/read/last-use constraints, layout/bank/address and supported device resource limits.
2. **Semantic action catalog**: mapping edits, composition/exchange edits, numeric edits only where safe conversion exists; action names/domains derived from typed IR, not an agent-specific vector.
3. **Constraint propagation and legal transition**: `apply(state, action) -> Accepted(new_state, certificate) | Rejected(counterexample, reason)`; never silently "repair" an illegal action, and never mutate original Functional IR.
4. **Observation projection**: a deterministic, versioned, variable-size operator/tensor/edge graph usable by GNN or other optimizers; full provenance and normalized optional resource/constraint features.
5. **Evidence binding**: same typed state/action hashes and certificate IDs at E0, compiled source E1 and independently qualified original Vitis E2/E3. Native tool results are observations, not fictional constraints or guaranteed clocks.

The RL environment owns:
- policy, network architecture, trajectories, exploration, replay, episode resets, reward function and training metrics;
- a *separate adapter* consuming the compiler's action/observation interface;
- no private fast path to bypass compiler legality or change Functional graph;
- no requirement for a particular RL algorithm, GNN model, pretrained policy or search method.

**Invariance rule:** if an operator mapping is selected manually, by exhaustive enumeration, random search, Bayesian optimization or RL, the resulting IR and compiler correctness certificate must be the same for the same canonical state/action pair.

## Proposed formal learning-compatible transition model

Let
`S=(G, H, Phi, C, M, K)`,
where `G` is Functional DAG, `H` is hardware capacity/target and user constraints, `Phi` is the independently editable operator mapping, `C` mapping-derived tensor edge contracts, `M` the storage/schedule plan, and `K` the bounded certificate/evidence status.

An action is a typed, immutable patch
`a=(target_scope, target_id, field, proposed_value)`.

Examples:
- Operator-local: update PE count, SIMD lanes, tile, precision *only if supported*, serialization, pipeline request.
- Tensor boundary: choose a concrete exchange/layout/physical slot plan with verified compiler lowering.
- Global: numeric error limit, target device/resource budget as *task constraints*, not freely rewritten by the agent; clock is a separate HLS request if experiment controls permit.
- `STOP` terminates a complete assignment without modifying the model.

```
transition(S,a):
  validate_action_schema_and_target(S,a)
  proposed = immutable_patch(S,a)
  resolve_operator_Phi(proposed)
  derive_all_affected_tensor_value_contracts(proposed)
  verify_dependency_reduction_and_last_use(proposed)
  verify_physical_storage_alias_and_address(proposed)
  verify_numeric_and_supported_lowering(proposed)
  => ACCEPT(new_S, source_bound_certificate) or REJECT(counterexample)
```

**Semantic mask:** `mask(S,a)=1` iff the corresponding candidate is **proven accepted within the stated IR/verifier bound**, not because its eventual synthesized HLS resource/timing is predicted feasible. `UNKNOWN/BOUND_EXCEEDED` must not be treated as legal. Do **not** mark global device timing/BRAM as satisfied from static source arrays; actual HLS may still fail.

**Partial decision states:** for sequential action construction, provide an explicit `UNASSIGNED` value and canonical completion obligations. Only candidate actions validated under the partial-state semantics can be marked valid, and the lowerer must reject incomplete assignments. Do not mislabel "could become completable" as proved complete. Under a full-design action space, the state is a complete legal design and each accepted patch preserves its contract.

**Markov qualification:** with a fixed graph/task, constraints, tool version, current full assignment and previous measurement observations included explicitly, the *pure compiler transition* is deterministic. Real HLS results may be noisy/partially observed; **do not claim a fully Markovian physical evaluation process without evidence**.

## Graph observation schema for variable-size networks

Node types: operator, tensor, exchange boundary, hardware target. Edge types: produces, consumes (including multiple distinct reads), mapping-of, candidate-alias, availability-before-read, lifetime-overlap and device-support/capacity. Support models with different depth/width; no required fixed layer count and no privileged layer-index-only feature.

Minimal per-operator public features: kind (MatMul/Gemm/Relu/Conv/etc only if implemented), input/output shape and dimension-normalized statistics, tile, PE/SIMD, serialization, precision and pipeline request, resolved `Phi` projection, local resource estimates marked estimates.

Minimal per-tensor/edge features: rank/shape, dtype, number of readers, mapped production/completion/last-use status, physical lifetime/scratch bank/slot, legal transfer policy and evidence provenance.

Minimal hardware/task features: device resource capacities (LUT/FF/DSP/BRAM/URAM when available), fixed numeric error limit, supported target primitives, user-provided constraints and device identity. Missing measurements need explicit `MISSING` plus mask; no zero imputation. Exact trained tensor weights should **not** be inserted into public GNN observation by default; weight shape/precision suffice for initial network-general policy study.

Observation requirements: deterministic canonical serialization, schema version, hashed state and graph, stable tensor/operator references, topological permutation equivariance for graph features, consistent normalization across network sizes, no ground-truth post-HLS QoR leaked into pre-evaluation observations, and no use of evaluation/test architectures to fit policy normalization.

## Reward and feedback separation

The IR must **not** calculate or learn the reward. The environment attaches measured outcomes only after source-pinned original Vitis HLS CSim/CSynth has passed the appropriate qualification gates, including bounded cycles, reported timing and numerical accuracy.

The RL research track previously chose **latency optimization under constraints**, not a generic fabricated weighted resource reward. Keep any trained policy/reward study isolated from this first compiler paper until explicitly unfrozen. Available FPGA resources may be spent to reduce qualified latency; hard resource and numeric bounds remain constraints, not reward bonuses.

A reward using HLS-requested clock as achieved Fmax is prohibited. Any cross-design latency score must use explicitly qualified timing provenance and document whether it is an HLS estimate or physical measurement. Failed native cases retain failure categories and complete denominators; no phantom zero-latency rewards.

## Exact falsifiable experiments

**R1 — deterministic compiler/search equivalence (highest priority before RL training).**
For each fully specified static graph/target, independently enumerate small legal actions, check `mask`, and compare `transition` acceptance against a separately implemented exhaustive/bounded reference. Every accepted action must be source-compilable and numerical-correct where supported. Every masked-out case returns a concrete reason or documented bound. Require 0 false-legal acceptance in the complete bounded domains; report incompleteness explicitly.

**R2 — mapping-conditioned action-mask causality.**
Hold Functional model, current boundary request and candidate edit constant; vary **only** the operator `Phi` in a way that changes actual value production/read completion order. The legal action set or certificate must change for the correct semantic reason, not merely due to a different `Phi` hash. Compare existing row-atomic mask vs full Phi-derived mask.

**R3 — graph representation invariance.**
Apply graph-preserving renamings/reorderings, tensor ID canonicalization and variable layer-count/width changes. Check equivalent actions preserve legal outcomes and certificates, while features change only in the expected canonical/permutation-equivariant way. Include multiple readers and different resource budgets.

**R4 — policy-independent use (no RL training).**
Use at least uniform random legal search, a deterministic greedy heuristic and one externally supplied action list through **exactly the same** transition interface. Compare illegal action rates, number of source/host-valid designs, coverage and validation overhead. This tests the IR, not a trained policy.

**R5 — actual RL ablation (later RL track, explicitly separate).**
After unfreezing, compare the *same* shared GNN policy, unseen network splits, HLS budget and numeric constraints using (a) raw operator-only graph, (b) graph + edge contracts, and (c) graph + contracts + exact semantic mask. Report legal proposal rate, unique valid architectures per HLS call, sample efficiency to qualified latency, source/numerical failures and cross-network generalization. Evaluate against random/heuristic search with **identical real-HLS budgets**; repeat seeds and count all failed/unknown cases. A gain is a hypothesis, not an asserted result.

**R6 — related-work comparator.**
Directly compare the semantics and available artifacts of CompilerGym, IronMan, RL4ReAl, CASCADE and applicable Allo/ST-Flow solutions. A new benchmark wrapper or fixed-size GNN feature tensor is not enough.

## Acceptance and publication scope

The initial compiler paper may reasonably include R1–R4 as evidence of a **search-/learning-compatible verifiable compiler IR**, without training an RL policy; describe this as a *secondary design contribution*, not a demonstrated RL speedup. To claim **RL sample efficiency or cross-network transfer as an empirical novelty**, R5 with source-attributed real-HLS outcomes and fair baselines is mandatory.

**Current implementation status:**
- Existing immutable Functional/Architectural IR, local `Phi`, typed exchange hashes, atomic row schedule checker and emitted GCC source exist.
- **Not yet present:** a complete certificate-backed mapping-conditioned per-value `transition` interface, exact action masks, multi-reader lifetime logic, device-aware admissibility or the R1–R4 experiments. These are **new sprint gates**, not already completed software.
- Existing RL environment is a separate frozen research track; **no RL code is changed by this design document**.

## Source implementation milestone — E0/E1 bounded complete-design interface (2026-11-02)

**Implemented in** `src/fpgai_research/model_ir/decision_interface.py`:

- Immutable `DecisionState` with complete per-operator `ArchitectureRequest`, exact Functional hash (including original weights), row-block/whole exchange plan and resolver `TargetLimits`.
- Typed `DecisionAction` for actual architecture settings and supported exchange policies; source-compiled `transition(state, action)` returns `Accepted` with source-bound certificate or `Rejected` with error and unchanged state. The same actual `resolve` + `compile_cst` path determines acceptance for any proposer.
- `action_mask` invokes `transition` for each *supplied finite action*, so it cannot disagree with the compiler on those exact proposals. `candidate_actions` is bounded and deliberately incomplete; no claim of exhausting legal hardware configurations.
- `graph_observation` has variable-size operator/tensor/edge nodes, Phi spatial probes, device resource caps, no raw trained constants or native-HLS performance leakage. Renamed equivalent graph observations are tested.
- `experiments/cst_decision_campaign.py` offers a create-only, seeded, policy-neutral experiment: all candidate actions from fixed small-domain models, a **restricted independent arithmetic oracle** for PE/SIMD/B decisions, exact compiler masks, selected actual GCC numerical checks, and a one-action random-vs-simple-heuristic interface demonstration. No trained RL, no native Vitis.
- The study is included in `experiments/compiler_ir/series.py` quick and paper profiles and conservative automatic evidence index, with targeted `tests/test_cst_decision_interface.py` and `tests/test_cst_decision_campaign.py`.

**Do not promote this result to the primary scientific novelty:** this v1 transition runs the existing **whole-row atomic** row schedule verifier. Its Phi snapshots and hashes do not derive per-element reduction completion or multi-consumer last use, and a successful source emission does not prove actual resource/timing feasibility. The primary publication gates L4–L5 (per-Phi value readiness and fan-out source-refined lifetime certificate) remain **unimplemented**. R5 (trained RL benefit) remains deferred with the original RL source frozen.

Full numbered sprint and experiment acceptance plan: [CERTIFIED_DECISION_IR_SPRINTS.md](CERTIFIED_DECISION_IR_SPRINTS.md). Specific source-pinned CI results should be recorded only after successful execution.


## 2026-11-03 revised executable C3 paper acceptance, not RL training

The heading and initial design-status descriptions above are **historical**.
The research compiler currently implements `DecisionState`,
`DecisionAction`, source-bound `transition`,
`graph_observation`, a generic SOURCE admission `action_mask`,
and the conservative three-outcome `certified_action_mask` for
bounded complete-design row-chain graphs. `UNKNOWN` is never
promoted to legally certified. Device timing and RAM-port feasibility
are explicitly **NOT** established by either mask.

The source-only `cst_decision_campaign` covers R1 on its
restricted independent algebraic oracle. The original
`cst_phi_event_campaign` tests R2 Phi-owner dependence but
does not claim arbitrary full-fanout, per-cycle hardware legality.
The newly implemented `cst_rl_interface_study` records three
static MatMul-chain depths, anonymous graph rename controls,
identical compiled result certificates for identical proposer
actions, and conservatively unknown-but-source-admissible edits.
It is **R3/R4 bounded protocol evidence only**: no trained agents,
learned improvement or unseen-network transfer.

The `cst_phi_counterfactual_effects` secondary analysis
distinguishes source-event/owner hashes from actual source last-use
and C++ scratch changes under reader-specific Phi interventions.
If it reports zero lifetime-changing treatments, that is a
**negative finding**, not justification to claim strong Phi
causality from differing source hashes.

Both new studies are wired into **quick AND paper**
`experiments/compiler_ir/series.py`; the independent evidence
index refuses to merge their results unless their original hashes,
source revision, and source-GCC causal CSV lineage qualify.
These new changes **have not yet been verified on the workstation**.
Use [the source-pinned experiment launch](FINAL_RESEARCH_EXECUTION.md).

Paper wording is permitted as a **policy-independent RL-compatible
compiler representation**, not as a novel reinforcement-learning
algorithm or demonstrated learning/synthesis acceleration. R5
comparison against random, standard RL, masked RL, and tensor-edge
feature ablations, using real bounded Vitis outcomes, remains a
different future empirical project. No frozen RL source changes.
