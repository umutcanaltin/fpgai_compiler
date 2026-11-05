# FPGAI HW-RL-05 — auditable online learning across neural networks

**Prospective scientific protocol, version 0.1 (5 November 2026).**
Implementation branch: `feature/rl-hardware-knobs-environment-20261104`.
No completed HW-RL-05 multi-network real-Vitis experiment has yet been
run. This specification does not confer publication readiness or prove
algorithm superiority. Keep compiler-paper novelty and research claims
separate from this distinct RL manuscript.

## Research scope and positioning

**Proposed paper direction:** *Auditable Graph-Conditioned Reinforcement
Learning for Compiler-Grounded FPGA Neural-Network Design-Space
Exploration*. A shared GNN acts on the exact typed Functional Model IR
(tensor/operator nodes and directed dependencies) and actual existing
compiler architecture knobs. A joint masked categorical choice and
architecture-conditioned Beta clock actor proposes an entire hardware
state. Every accepted action is synthesized by original Vitis HLS 2023.2
and independently re-audited before learning. We test generalization
to *different network graphs*, not optimization of a memorized model.

**Do not claim first use of RL in HLS DSE:** prior work includes
IronMan-Pro (IEEE TCAD 2022/2023, DOI 10.1109/TCAD.2022.3185540),
gnn4hls (ACM TODAES, DOI 10.1145/3570925),
SENNA (ACM TECS 2025, DOI 10.1145/3705731),
a Q-learning/evolutionary HLS search (ISCAS 2025,
DOI 10.1109/ISCAS56072.2025.11043387), and Chariot
(IEEE TCAD 2026, DOI 10.1109/TCAD.2026.3732102).
The substantive differentiation must be measured: one *source-linked*,
*multi-operator*, *clock-aware*, shared on-policy optimizer with hard
source/CSim/area/timing receipts and cross-network transfer.
Graph-policy methods already exist, so representation alone is not
proof of novelty.

## Falsifiable questions

- **RQ1: Transfer.** With disjoint train/validation/test network
  graphs, does a shared trained policy exceed the matched random
  baseline's *HLS-qualified discovery rate* at a fixed Vitis budget?
- **RQ2: Algorithm choice.** Do the same GNN encoder and action
  distribution yield different real-native learning curves under
  one-step actor-critic, episodic REINFORCE-with-value-baseline,
  and clipped PPO? Is any difference reproducible over seeds?
- **RQ3: Representation.** Is message passing or observed-audit
  trial-history attention beneficial versus exactly matched
  `--no-message-passing` / `--no-trial-history` ablations?
- **RQ4: Clock co-design (future).** Does jointly selected requested
  clock offer a benefit versus a preregistered clock-only/fixed-clock
  baseline? **Not implemented as a comparable HW-RL-05 treatment yet.**

Negative or null outcomes must be reported; policy-gradient execution
or lower training loss does not establish optimization improvement.

## Exact environment and reward contract

One Gym episode has one frozen real Functional Model IR, immutable
weights and CSim samples, a fixed device part, exact five usable resource
caps and numerical error threshold. Observations contain the typed graph,
per-operator source-legal hardware action targets, constraints,
missing-quantity masks and only *previous independently audited* native
feedback. No model weights are exposed as policy features.
Architecture actions are compiler-resolved; unsupported banking, DMA,
clock domains, physical routing and board changes are not in the menu.

The architecture-conditioned clock actor samples a real Beta latent
`u in (0,1)`, maps it to the bounded `[2,32] ns` interval and
canonicalizes to **0.01 ns** for native Vitis 2023.2 XML attribution.
This is not a fixed clock list. The latent probability is used in
policy gradients; the recorded clock is the actual canonical Vitis
requested target. The discrete action is masked by real source
compilability, not guessed HLS feasibility. The current action changes
one source-backed architecture request (or clock only), retaining the
complete previous configuration for all other operators.

Original E3 qualification requires:

```text
CSim passes && max_absolute_error <= epsilon
&& (LUT, FF, DSP, BRAM18K, URAM) <= exact user budgets
&& original HLS estimated clock <=
   original requested clock - original explicit clock uncertainty
&& actual original source/part/numeric/report hashes agree
```

For qualified cases, estimated whole-network latency is
`L(ns) = HLS_whole_top_worst_cycles * requested_clock_ns`.
For rejected cases it is **null**, never a numeric estimate counted
as a qualified performance point. The unchanged reward is
`-0.05` for an independently audited infeasible case, `-0.005`
for the first qualified case, `-0.005` without improvement and
`ln(previous_best_ns/new_best_ns) - 0.005` on improvement.
There is no area/energy reward; all available resources are usable.
No routed FPGA timing, physical Fmax, power or board latency claim.

**Observed HW-RL-04 negative case, not a learning curve:** sampled
operator-5 tile `[1,4,27]` at 6.55 ns; estimated HLS clock
5.935 ns and uncertainty 1.77 ns; effective budget 4.78 ns,
margin **−1.155 ns**. CSim error 0.0003907345, all resources pass,
raw 780 cycles, `qualified=false`, latency null and reward −0.05.
There was one genuine gradient update, but no saved checkpoint.

## Implemented comparisons and algorithmic specifics

| Method | Gradient update | Policy | Evidence |
|---|---|---|---|
| `actor_critic` | one-step TD actor, critic and original-HLS feasibility auxiliary after each audited step | shared GNN + masked categorical/Beta | original independent replay before optimizer |
| `reinforce` | one episode of discounted reward-to-go with a learned value baseline; single optimizer step after full episode | same GNN/actions | independently re-audit each original native step before buffering |
| `ppo` | finite-episode discounted returns; real clipped importance ratios recomputed from stored original action/latent after each PPO epoch; value, entropy and feasibility auxiliary | same GNN/actions | exact original Vitis audit before storing, no vendor calls during optimizer epochs |
| `random` | none | uniform legal discrete action + continuous uniform clock, independent RNG | same original Gym native/CSim/timing gates |

Implemented no-GNN and no-history *ablation flags* disable those
components without changing Gym dynamics. The PPO implementation uses
full-episode on-policy batches (not minibatches), finite returns rather
than GAE, and a learned critic; describe those implementation choices
precisely. Both REINFORCE and PPO update only at episode boundaries;
actor-critic updates after each audited step.

Private evaluation-only `shared_hardware_gnn.npz` checkpoints
remain restore-only and contain **no optimizer state**. Separately,
HW-RL-05 now writes `training_state/episode_*/training_tensors.npz`,
containing the shared GNN, the exact Adam moment tensors, torch CPU
sampling RNG, source SHA and episode/task metadata. This separate
snapshot is validated with `allow_pickle=False` and SHA checks.
It is **not an automatically resumable original-Vitis campaign**:
the remaining original call budget, previous immutable case journal
and source identity must all be reconciled before continuation.
Interrupted native campaigns remain provisional; never relaunch
into an existing case folder.

## Experimental design (preregister before big native runs)

1. Fix FPGA part, compiler/Vitis version, timing uncertainty policy,
   hardware caps, numerical cohort and per-network original HLS call
   budget before testing. Preflight sources first. Use isolated
   create-only output roots for each method, seed and split.
2. Define multiple true network graph families supported by the
   compiler: small MatMul/Gemm and mixed Conv→activation/pool→dense,
   including distinct widths, channels and operator counts.
   Freeze disjoint ONNX fingerprints for train/validation/test;
   separately document graph/size distribution and any overlap in
   architecture families. The manifest refuses exact ONNX-byte reuse
   across splits. Declare sample generation seeds.
3. Start with **one** prospective 1-network×2-call native smoke per
   method, then independent replay. Only after all methods pass
   CPU and native audit gates, register a larger schedule such as
   100 training network episodes × 10 original Vitis steps, with
   explicit authorization for each method/seed. Do not start the
   full study on source modifications alone.
4. Use at least 3 independently initialized seeds (for example
   111, 222, 333) and **identical task schedules and native budgets**
   across methods, with separate independent random streams.
   Repeated networks are counted and identified (they are **not**
   independent distinct-network generalization).
5. Evaluate frozen private in-project checkpoints with no parameter
   update on untouched validation/test networks, matched against
   `random` and no-message-passing controls, and report stochastic
   or deterministic evaluation selection explicitly. No hidden
   architecture search or target-clock grid.
6. Additional evolution/TPE, fixed-clock, no-feasibility-head,
   normalization and cross-device ablations are **planned**, not
   implemented or validated in HW-RL-05. Do not claim comparisons
   against methods that were not run.

## Primary scientific measurements and safeguards

Report **feasible discovery rate at the fixed budget** (denominator
all preregistered tasks), first feasible call index, full original
resource/numeric/timing rejection funnel, tool calls, original Vitis
wall-time, and best qualified HLS-estimated ns per network. When
either method has no feasible result, mark that pair **censored**;
never silently discard it or assign infinite/zero fabricated latency.
Only compare conditional latency ratios for *both-feasible* matched
tasks, accompanied by discovery rates and all censoring categories.
Use paired per-task comparisons, seed-level scatter and bootstrap
confidence intervals where sample sizes support them; report raw
data without artificial monotonic smoothing. The observed incumbent
minimum curve may be monotone by definition but raw reward need not be.

No scientific claim without complete source-pinned independent campaign
replay (`--audit` returns `AUDITED_COMPLETE`), fixed budgets and
frozen heldout results. A complete campaign log before independent
replay has status `COMPLETE_PENDING_INDEPENDENT_CAMPAIGN_AUDIT`.
The source-only `--plan`, CPU fixtures and policy loss are **not**
hardware benchmark results. The repository's current replay checks
original sources, native reports, rewards, task identities and the
journal but does not reconstruct PyTorch optimizer updates.
Treat training proof and native case proof separately.

## Task manifest and commands

The new CLI accepts a source-controlled JSON manifest. Paths are
relative to the *manifest's directory*; each task has
`id`, `split`, `onnx`, `samples`, `part`,
`resource_caps=[LUT,FF,DSP,BRAM18K,URAM]`, and
`max_absolute_error`; `initial_clock_ns` and
`max_parallel_units` are optional (14.00 and 8 defaults).
Example of a **local single-train-network smoke**, not a valid
cross-network scientific study:

```bash
cd ~/Desktop/github_projects/fpgai_compiler
mkdir -p build
python - <<'PY'
import json
from pathlib import Path
doc = {"schema":"fpgai-research/hardware-rl-tasks/v1", "tasks":[{
    "id":"cnn_train_smoke_001", "split":"train",
    "onnx":"demo/cnn_pool_dense.onnx",
    "samples":"rl_hw02_cnn_samples_seed123.npy",
    "part":"xc7z020clg400-1",
    "resource_caps":[53200,106400,220,280,0],
    "max_absolute_error":0.1,
    "initial_clock_ns":14.0
}]}
Path("build/rl_hw05_tasks_smoke.json").write_text(json.dumps(doc,indent=2))
PY

# Safe dry-run; zero vendor calls and no case directory
fpgai-rl-hardware-campaign \
  --tasks build/rl_hw05_tasks_smoke.json \
  --method ppo --episodes 1 --calls-per-episode 2

# Separately, only after all CPU tests pass and native opt-in approved:
fpgai-rl-hardware-campaign \
  --tasks build/rl_hw05_tasks_smoke.json \
  --method ppo --episodes 1 --calls-per-episode 2 \
  --max-total-vitis-calls 2 \
  --vitis /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls \
  --output-root build/rl_hw05_ppo_smoke_seed101_01 \
  --execute-native

# Independent read-only exact-source replay (never reruns Vitis)
fpgai-rl-hardware-campaign \
  --tasks build/rl_hw05_tasks_smoke.json \
  --output-root build/rl_hw05_ppo_smoke_seed101_01 \
  --audit
```

Checkpoints live under `<campaign>/checkpoints/episode_000000/`.
For frozen evaluation, use the same method and network dimensions with
`--evaluation --checkpoint PATH --split test`, requiring a manifest
with genuinely disjoint test models. Full training resume is not
implemented; using an evaluation checkpoint as an optimizer continuation
is prohibited.

## Next acceptance gates

- Workstation PyTorch/Gymnasium/ONNX import and all focused/full CPU
  suites must pass, including `tests/test_rl_hardware_campaign.py`.
- Validate source-only `--plan`, installed CLI from outside checkout,
  missing/duplicate source and invalid budget tests.
- One 2-call fresh original Vitis native smoke *per algorithm*, including
  original replay, completed episode journal and source-pinned case identity.
- If all pass, implement read-only pairwise multi-method comparison,
  exact safe optimizer/RNG checkpoint resume, novelty-aware action masking,
  bounded crash recovery, and additional baselines before the main study.
- Only then schedule the large original Vitis campaigns and draft a
  manuscript around independently measured outcomes.

## Audited matched heldout comparison (HW-RL-05D source-added)

A second installed command `fpgai-rl-hardware-compare` takes two
**completed frozen validation/test** campaign roots from the same
source SHA, seed, original network schedule, source hashes and per-task
original Vitis budget. It independently replays every original native
case in each campaign; then it writes a new create-only pairwise
`comparison.json` and `paired_native_hls.csv`. No new Vitis calls or
optimizer steps are performed.

```bash
fpgai-rl-hardware-compare \
  --tasks build/rl_hw05_tasks_heldout.json \
  --baseline build/rl_hw05_random_test_seed111_01 \
  --candidate build/rl_hw05_ppo_test_seed111_01 \
  --output build/rl_hw05_paper_pairs_ppo_vs_random_seed111_01
```

These paths are illustrative future experiments, **not existing
measured cases**. The comparator refuses train-vs-test leakage,
unequal schedules/budgets and incomplete native runs. It reports
qualified discovery rates and all four censoring categories, first
feasible call and conditional HLS latency ratios. It does not infer
statistical significance from a tiny smoke or fake training data.
The implementation/tests still require workstation verification.


### Source-level reproducibility gates added 5 November

The standalone `hardware_campaign --audit` now reconstructs not
only the emitted source states and original numerical/HLS receipts,
but also the *exact compiler action catalog index* recorded for each
trial. The original case path, source SHA, requested clock, reward,
qualified latency/null censoring and per-episode summaries are
checked without any vendor tool call.

The paper comparator only grants an `INDEPENDENT_NATIVE_CASES_AUDITED`
result after two full successful original-native replays. A direct
unit-test summary over synthetic dictionaries is **unverified** and
is rejected by the paper export gate. No source-level audit
automatically proves that a PyTorch optimizer produced superior
heldout hardware performance; effect sizes must be measured from
frozen, source-matched original Vitis runs.

### Methods and losses

Let `a_t` be the compiler-legal masked discrete action and
`u_t` the original policy-sampled Beta clock latent. Its canonical
Vitis clock `c_t` is recorded exactly; the policy-gradient log
probability is `log pi_theta(a_t|s_t) + log Beta_theta(u_t|a_t,s_t)`.
The three policy objectives differ only in how they use independently
observed reward data:

- One-step actor-critic: TD target `r_t + gamma*V(s_(t+1))` for
  nonterminal steps, plus value loss and the shared observed
  feasibility auxiliary.
- REINFORCE with learned baseline: complete episode discounted
  `G_t = sum_(k>=t) gamma^(k-t)*r_k`; actor loss
  `-mean[log pi_t * (G_t - V(s_t).detach())]`, with critic
  squared error and feasibility auxiliary, once per full episode.
- PPO-Clip: old rollout log probabilities are frozen; each optimizer
  epoch re-evaluates the original observation/action/latent under the
  current GNN, applies the standard
  `min(rho_t*A_t, clip(rho_t,1-eps,1+eps)*A_t)` objective,
  plus critic, entropy and the same feasibility auxiliary.
  Current PPO is full-episode, not minibatch/GAE PPO.

No algorithm uses a learned QoR surrogate as its reward. The
`torch` RNG is private to the active shared-actor run and the
random search baseline uses its own independent NumPy generator.
None of these mathematical definitions claim an empirical advantage
before actual frozen heldout benchmark results.

