> **Research-track freeze (2026-11-01):** This RL research contract is preserved for later work, but new RL development, training campaigns and agent additions are paused while [the compiler-first FPGAI IR paper](../paper/compiler_ir/README.md) is prioritized. An already-running source-pinned L0 campaign may finish and be audited without modifying its local source; do not pull new remote `main` into that checkout until completion. The new compiler IR may expose a *future* graph/action interface compatible with RL; that is a design/interface property, not measured RL learning or improvement. Nothing below has been silently redefined.

**Prospective Gym learning remediation (31 October 2026):** The first independently audited original-Vitis pilot at `eb4a799` ran 100 training episodes ×10 calls, but qualified only 71/1,000 actions, had five positive rewards, confounded frozen deterministic with adaptive stochastic sampling, and duplicated four baseline search trajectories. This does not establish learned generalization. The *next fresh source-pinned* study, defined in [SPRINT_RL_LEARNING_REDESIGN.md](SPRINT_RL_LEARNING_REDESIGN.md), retains the exact latency-only objective and reward while adding explicit original-HLS failure-gate diagnostics, trial-history-attentive graph policy, true compiler-legal global and local candidate proposals, and auxiliary binary original-HLS feasibility prediction trained alongside RL (not as replacement reward). New matched frozen stochastic and adaptive arms yield 100 online train + 1 validation + 8 controls =109 episodes capped at 10 evaluations each, total maximum 1090. New source is not yet workstation/native qualified; preserve the historical 1,070-case dataset and its independent audit.

# Research question — FPGAI Gym-based Graph RL for latency-only HLS design-space exploration

**Canonical prospective research contract | 31 October 2026 | Design locked; 31/31 focused and 658/658 full tests verified at `0cc2a22`. Later 100-episode campaign controls and strict 30-HLS-attempt cap require retest; no new original Gym Vitis qualification.**

This file is the single source of truth for the **new research direction**. It supersedes contradictory objectives in older RL v1–v9 experiments, but does **not** revise the original paper, immutable HLS receipts, historical commits, or evidence. The original compiler is reused as a dependency, not rewritten.

**Paper evidence contract:** The restarted source-pinned native study captures the exact executable reward source SHA, the numeric `-0.05` infeasible / `-0.005` evaluation penalty / logarithmic best-HLS-ns improvement formula, per-proposal actions and original Vitis numerical/resource/timing receipts, and full policy-gradient diagnostics for the same shared GNN. Audited exports include reward and feasibility by training episode, per-step/within-task best latency curves and loss/entropy by policy update. Do **not** treat different tasks' raw absolute latency as one transferable learning curve or claim learning superiority from a single seed/test task. See [PAPER_EVIDENCE_SPEC.md](PAPER_EVIDENCE_SPEC.md).

## 1. Main research question

> **Can one shared graph-neural-network reinforcement-learning agent, trained online from random initialization in its own Gymnasium environment, learn a transferable, online-adaptive search strategy that minimizes the numerically valid, FPGA-resource-feasible, HLS-timing-feasible estimated end-to-end latency of previously unseen neural networks, using only original Vitis HLS synthesis feedback?**

Supporting questions:

- **RQ1 — Representation:** Can one shared GNN encode varying layer counts, layer types, shapes and connections without retraining a separate encoder for each network?
- **RQ2 — Transfer:** Does accumulated training experience on different networks improve search quality on unseen networks under identical native HLS evaluation budgets?
- **RQ3 — Online adaptation:** Does updating the shared policy after each fresh, verified HLS evaluation improve subsequent decisions compared with a frozen version of the policy trained on previous Gym episodes?
- **RQ4 — Search efficiency:** Does graph-based RL reach the same or lower feasible estimated latency with fewer real HLS calls than random, local, evolutionary and Bayesian search?
- **RQ5 — Clock/architecture interaction:** Does jointly choosing the requested global HLS clock period and per-layer architecture provide better HLS-qualified estimated latency than fixed-clock search?

**Out of scope for this environment:** post-HLS Vivado synthesis/placement/routing, measured/routed Fmax, board performance, energy/power objectives, resource-minimization or Pareto objectives. Other academic research tracks are not part of this Gym environment.

## 2. One episode = one immutable acceleration task

The selected next experiment is **100 online training episodes** with one shared policy, **one frozen validation episode**, and **six frozen/adaptive/baseline test episodes**, all on one seed. Each new episode is capped at **30 original Vitis HLS attempts**, with a maximum of **3,210 total**. These are strict invocation budgets, not cycle/resource reward terms. An optional assumed average compile time yields a forecast **only**: no 8-hour or other campaign wall-clock limit is applied. Periodic source-pinned private checkpoints and completed-episode journals preserve progress, but auto-resume is not implemented. See [EXPERIMENT_100_EPISODES.md](EXPERIMENT_100_EPISODES.md).


The task is a real neural network (the *optimization problem*, not the policy network) plus device constraints:

```
EpisodeTask(
    functional_model_graph,       # entire true model: layers, connections and trained constants
    target_fpga_part,
    available_resources,          # LUT, FF, DSP, BRAM, URAM; usable caps, not nominal guesses
    max_numerical_error,          # clearly defined maximum output-error metric
    # internally generated bounded clock-action envelope, NOT a user input
    fixed_numerical_test_inputs,  # private immutable samples/reference
    original_vitis_call_budget,
    split_and_seed,
)
```

The task graph, tensor values, reference numerical cohort, resource budgets and error bound remain **fixed during the episode**. A new episode may change all of them. Episode transitions do **not** reset the GNN/policy weights.

The task's **model weights and test inputs are used internally for actual numerical validation**, but raw numerical weights and activation statistics are **not initial agent-observation features**. Privacy-preserving hashes identify the exact instance.

## 3. Gymnasium observation contract

### 3.1 Static network graph — variable number of nodes/edges

**Each operation/layer node** exposes only:

- **Type identity:** e.g. MatMul/Dense, Conv, ReLU, Attention; a supported type vocabulary with unknown/type masks.
- **Dimensions:** tensor input/output shapes, hidden sizes, input/output channels, kernel sizes, strides, paddings or attention-head dimensions *where applicable*. Type-specific absent values require explicit masks; zero alone must not ambiguously mean "not applicable".
- **Graph structure:** directed producer/consumer edges, including branches, merges and skip connections. Tensor-edge shape and dtype can encode their structural identity.

The total layer count and hidden-layer organization are recoverable from the graph; optionally supply an explicit global count. There is **no fixed maximum topology in the mathematical representation**. Implementation may use bounded padded tensors with valid-node/edge masks when needed by a specific RL library.

A **single shared, trainable GNN** (e.g. message-passing or graph transformer) receives all network graphs and returns both variable-length per-node embeddings and one **fixed-dimensional global embedding**. A GNN over hardcoded mean features without learned graph-message passing does **not** meet this target requirement.

### 3.2 Global constraints

Provide `[available_LUT, available_FF, available_DSP, available_BRAM, available_URAM, max_output_error]`; the normal experimenter supplies **no clock values or clock grid**. The Gym environment derives its bounded clock exploration envelope internally (and records it for audit). **Never pass a desired operating clock, default requested clock, assumed maximum frequency or a user-specified list of periods as a normal task input.** The shared policy selects a requested global HLS period with every architecture action. The shared GNN policy learns a **real-valued Beta distribution** over requested HLS periods, conditioned on the graph and proposed architecture. An internally fixed **2–32 ns engineering safety interval** bounds the action; proposals are rounded to **0.001 ns** for exact Vitis Tcl generation, identity and source-pinned replay. This is a trainable high-resolution floating actor, **not** an enumerated user/internal clock grid, not mathematically unbounded frequency search, and not physical Fmax. Only original HLS results qualify timing. Numeric error bound is a hard constraint on configurations.

### 3.3 Dynamic search state

After each audited step provide:

- current proposed/accepted full-network hardware mapping, including the requested clock;
- last actual original Vitis HLS timing, cycle latency, estimated resource consumption and CSim error with missingness/status masks;
- remaining resource headroom **for agent decision context only**, never as a resource-saving reward;
- best *feasible* previously observed configuration and its estimated latency;
- remaining original HLS evaluations and rejection/failure state.

No synthetic or proxy QoR values appear as actual HLS observations.

## 4. One shared GNN + one shared DRL policy

**One policy, one graph encoder and one set of model weights across every episode.** These parameters persist through Gym resets.

Architecture proposal:

```
typed computational graph -> shared trainable GNN
FPGA capacity + error limit + source-defined floating-clock safety bounds -> constraint encoder
last HLS observations + current/best configurations -> state encoder
joint representation -> shared actor and critic
actor -> legal complete-configuration decoder
```

The same policy must handle different operator counts and hidden sizes. Both per-node embeddings and a pooled global embedding remain available to the decoder. Graph-level pooling alone is insufficient if it destroys operator-local decision context. Use legality masks for unsupported choices.

This common policy makes transfer possible, **not guaranteed**; generalization is experimentally measured.

The RL algorithm is pluggable (actor–critic/PPO and suitable alternatives can be evaluated). The Gym environment performs no gradient updates itself. A separate trainer initializes the GNN and policy **from scratch**, updates their shared parameters online from original HLS transitions, and saves resumable checkpoints.

## 5. Action space

A substantial agent action proposes **one complete new hardware candidate** for the entire network:

- **Global precision / integer width**, initially global until mixed-precision edge-cast correctness is supported by the compiler.
- **Operator-local** parallelism, PEs, SIMD, tiling, serialization/reuse and pipeline-II controls, to the extent the real compiler can actually materialize them.
- **Global requested HLS clock period** sampled from an architecture-conditioned, trainable Beta distribution by the shared policy. Its floating-point result (canonical 0.001 ns precision) is selected jointly with one complete legal architecture. No predefined clock grid, experimenter-specified clock value or supplied list is part of the normal research API. A fixed clock is used only as an explicit registered ablation.

The variable-size action may use a shared autoregressive per-node decoder. It must generate complete, typed, compiler-legal Model IR architecture requests, not edit HLS C++ ad hoc. All legal actions receive the **same feasible configuration rules** across RL and baseline algorithms. A clock-period change requires a fresh native HLS synthesis, even if the HLS source text is identical.

No separate reward for resource savings; resource-intensive mappings are welcome if they fit and are faster.

## 6. Exact native HLS step

Each accepted **new** candidate triggers **one original, explicitly authorized Vitis HLS program invocation**, configured to run CSim and C synthesis. No Vivado.

```
Gym.reset(task) -> observation, info           # must run zero vendor tools
agent.choose(observation) -> full mapping + requested clock
Gym.step(action):
  validate architecture/clock and novelty
  real FPGAI Functional -> Architectural -> Lowered IR -> HLS C++
  stage fresh immutable source/part/clock/testbench/manifest
  run original Vitis HLS CSim + CSynth explicitly
  independently audit vendor invocation and source/hash binding
  parse whole-top cycles, HLS timing estimate, resources, numeric error
  qualify constraints -> compute native latency/reward or infeasible feedback
  return observation, reward, terminated, truncated, info
trainer performs online update before next policy decision
```

A failed, missing or interrupted native case remains immutable; do not silently retry or invent metrics. `--plan`, `reset`, audits, reports and structural tests must launch zero vendor processes. Explicit native opt-in, timeout, total call cap, source pin and create-only output are mandatory.

## 7. Exactly one objective: lowest HLS-qualified estimated latency

**Primary objective**

```
minimize L_HLS_estimated_ns = HLS_whole_top_worst_case_cycles * requested_clock_period_ns
```

Record the best qualified **requested target frequency** observed so far (`1000 / selected_period_ns` MHz) as a diagnostic, not a second reward term. It is **not** an exact optimum or physical Fmax: the bounded 0.001 ns-precision policy can only demonstrate the lowest latency and highest HLS-qualified requested target **observed among actual native evaluations**; it cannot establish an unrestricted optimum or physical Fmax. The above is a **conditional HLS timing estimate**, not real routed or measured FPGA latency, not maximum physical Fmax. Do not compute it as cycles divided by a falsely claimed achieved frequency.

**Hard feasibility requirements**

1. Original source-bound numerical CSim error `<= epsilon` on the frozen validation cohort.
2. Each original HLS-estimated resource usage `<= available FPGA budget` for LUT, FF, DSP, BRAM and URAM.
3. **HLS timing gate (Vitis HLS 2023.2):** original CSynth XML `EstimatedClockPeriod` must be **no greater than `TargetClockPeriod − ClockUncertainty`**. Vitis HLS subtracts its reported uncertainty from the requested target to form the *effective synthesis scheduling budget*; the 2023.2 default uncertainty is 27% when not specified. Require all three fields to be present and unique, and reject negative scheduling margin, missing/ambiguous values or clock/source mismatches. Never qualify a design merely because `EstimatedClockPeriod ≤ TargetClockPeriod`, or double-count the uncertainty in the ns latency objective. The original target period—not the shorter scheduling budget—remains the clock period multiplying worst-case HLS cycles for qualified estimated latency. These are HLS-only scheduling estimates, not routed physical Fmax.
4. Valid original vendor receipt, source/testbench/clock/part binding, successful synthesis and supported architecture.

Resource usage may reach **100% of declared usable capacity**. No area penalty, area weight, energy reward, multi-objective Pareto reward or hidden preference for smaller designs. Timing feasibility is estimated at HLS level only. The same explicit timing parser must be validated against actual installed Vitis report format; do not claim timing qualification from arbitrary numeric-looking log strings.

**Candidate reward** for a verified feasible design, using previous best feasible latency `L_old` and updated `L_new`:

```
r_t = log(L_old / L_new) - lambda_eval   # with L_new <= L_old; zero improvement -> -lambda_eval
```

If there is no prior qualified baseline, use a documented feasibility/initialization reward with no fabricated `L_old`; infeasible candidates get a documented finite penalty and no fabricated latency. Evaluation cost penalty is **only for consuming expensive Vitis calls**, not for resource consumption. If the baseline is infeasible, report that status rather than forcing a bogus speedup ratio.

Use a positive-latency, dimensionless, cross-task normalization for reporting and learning; total native wall time is an explicit secondary *search cost*, never hardware latency.

## 8. From-scratch online training in our Gym environment

**No separate pretrained DRL model and no offline pretraining stage.** Train one shared trainable GNN and DRL policy **from random initialization in our own Gymnasium environment**, using only the real, independently verified HLS results returned by its actions.

1. **Initialize from scratch:** record training seed and random GNN/actor/critic initialization; no imported policy weights.
2. **Online training across episodes:** each episode is a different training neural network. After each qualified or infeasible original Vitis evaluation, the trainer applies an online RL update. Do not reset policy weights on Gym `reset()`.
3. **Checkpoint during this training:** periodically save the currently trained shared policy/optimizer state with model, environment, source, seed and native-call provenance. Checkpoints are *outputs of this same Gym training*, not external pretrained models.
4. **Frozen unseen-network evaluation:** clone a selected training checkpoint and disable updates on heldout networks; call it *frozen transfer* rather than implying an external pretrained model.
5. **Separate online adaptation test:** start from an identical saved in-project training checkpoint, allow updates on new test tasks, and compare under identical original HLS budgets. Do not contaminate the frozen-policy arm.
6. **Optional continual adaptation:** persist online updates across a specified test-task sequence only as a separately declared experiment; do not claim its test networks were unseen after adaptation.

Do not train on final held-out test tasks before the corresponding evaluation. Validation may be consulted for checkpoint selection but should not be used for gradient updates in frozen evaluation. Random initialization, per-step online updates, training-generated checkpoint source/data provenance and frozen evaluation must be logged.

## 9. Registered research experiments

**E0 — Engineering qualification.** Test variable graph sizes/types/connectivity, fixed-dimensional global embedding, shared GNN parameters, legal decoder, global clock action, Gym reset/step semantics, native provenance, numerical error, resource ceiling, HLS estimated-period parsing and fail-closed behavior. Unit tests with fixtures do NOT count as original Vitis data.

**E1 — Controlled from-scratch online Gym training.** Train one shared GNN/policy from random initialization over different training networks with multiple seeds, recording every original HLS call and its complete training cost. Compare trained checkpoints with random initialization under equal test budgets; do not import pretrained weights.

**E2 — Transfer by network architecture.** Frozen heldout test networks with unseen hidden sizes, depth and combinations of supported operators. Branches, CNN, attention and cross-FPGA claims are prohibited until real compilation/support is demonstrated. Separate by topology *and* model/seed identity.

**E3 — Online adaptation.** Same in-project training checkpoint, heldout task sequence and native evaluation budget; frozen policy vs online-adaptive policy. Isolate histories to prevent test contamination.

**E4 — Clock/architecture ablation.** Fixed requested clock vs joint legal clock + architecture search, with identical tool budgets and strict HLS timing evidence.

**E5 — Optimization baselines.** Random, greedy/local, evolutionary and Bayesian/TPE search plus no-GNN and no-prior-Gym-training ablations. Each receives the same *original Vitis* call budget and feasibility rules. Clearly separate the original Vitis call cost of online Gym training from evaluation-time search cost.

**E6 — Generalization/robustness.** Multiple seeds, increasingly deep supported networks, resource budgets and numerical tolerances, then distinct FPGA parts when qualified. Do not extrapolate a small K26 MatMul/Gemm pilot to arbitrary architectures.

Primary plots: best **feasible HLS-estimated ns** vs original HLS evaluations; unseen-validation performance vs training episodes; frozen transfer vs online adaptation; feasibility fraction; normalized per-task latency improvement; matched random and stronger baseline comparisons; HLS tool wall time; distributions and confidence intervals. Never use only policy loss or a 1-network tie to claim transfer.

## 10. Current repository reality and change gates

As of the independently audited 31 October 2026 smoke (old source revision `d8bfdcc`):
- 16/16 original Vitis cases qualified *under the OLD cycles-only, fixed-clock criteria*; 450.353 s vendor-process time; 2 online updates.
- One held-out network produced DRL 43 cycles vs random 43 cycles (a tie).
- Independent replay audit completed with zero original-file modifications and zero new vendor launches.
- This verifies the prior source/native pipeline, **NOT** the new graph-GNN, clock-choice, HLS timing gate, from-scratch shared-GNN training or latency-ns reward.

The **retained legacy** `agent.py` / `study.py` / cycle-only evidence do not meet the new research contract. The new `gym_env.py`, `gym_agent.py`, `timing.py`, `reward.py`, `gym_study.py`, `gym_audit.py` path passed 21 focused and 648 full CPU tests at `01c7f87`, but later strict effective-HLS-budget changes need a fresh workstation run. No source-pinned original **new** native Gym cases are yet qualified:
- `agent.py` pools handwritten graph features, **not** a learned message-passing GNN.
- The **historical** `post_hls_env.py` was cycle-only. It now has a candidate-clock path and a stricter HLS timing and estimated-latency-ns gate, **not yet validated with actual post-CSynth XML on the installed Vitis version**.
- The historical `tasks.py` had a fixed clock; for new Gym studies `clock_ns=None` explicitly removes that input. The standard task has no clock input or choice list, and `gym_env.py` accepts `(legal_architecture_slot, learned_floating_clock_ns)` directly. The trainable Beta clock head and discrete architecture head share one policy and receive native feedback. Only source-defined 2–32 ns bounds and 0.001 ns canonicalization constrain the numeric action. Users do not pass a period or clock grid to the normal Gym study CLI. The historical pilot's clock property survives solely for compatibility.
- `study.py` is a retired legacy cycles-only runner (native execution blocked); the new `gym_agent.py` and `gym_study.py` supply the unverified shared trainable GNN and from-scratch online-training source path.
- `cohort.py` currently generates 1–4 MatMul/Gemm operators only.

**Do not reuse historical smoke metrics as if they satisfied this new specification.** New CPU fixture tests and initial source features were added; they still require execution, fixes, original Vitis timing-grammar inspection and new source-pinned independent native audit before experiments.

## 11. Cleanup, history and publication

The active `main` branch (after PR #4) contains **only the new Gym research direction** and retained compiler dependencies. Older proxy/v4–v9 study drivers, post-route experiments and old paper experiment assets may be removed from this branch **only after checking direct compiler/HLS/runtime import dependencies**. Preserve compiler semantics, current Gym/native audit tests, and original Vitis evidence provenance. Git history and the older named branches serve as the historical archive: no history rewrite, no forced branch deletion, and no changes to already completed private native studies.

Repository publication is a separate later decision. Keep private until IP ownership, an explicit license choice, full-history secrets/model-weights scan and institutional consent are complete.
