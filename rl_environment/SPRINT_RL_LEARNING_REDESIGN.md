# FPGAI Gym research sprint — feasibility-sparse 100 × 10 experiment remediation

**31 October 2026. Status: source implementation added to `main`, but CPU/native workstation verification pending.** Historical source-pinned pilot `eb4a7991a5ec870f2b543412dc60dde2da237c41` is an immutable independently audited baseline. **Do not edit that study's native cases, code snapshots or recorded reward.** This revision deliberately changes the prospective RL policy/observation/search protocol, so it **requires a brand-new output directory**.

## Motivation grounded in actual audited old pilot

- 100 original training tasks × 10 original Vitis evaluations: **71/1,000 feasible**, **929 rejected**, only **5 positive improvement rewards**. First 50 feasibility 6.6%, last 50 7.6%; no reliable upward learning claim.
- All four original baseline arms followed the *same* ten actions on the single held-out test task after no feasible incumbent; not four independent methods.
- Original frozen policy used deterministic argmax/Beta-mean, while adaptive test sampled architecture/Beta clock, confounding online adaptation; adaptive's first action was already feasible before any adaptation.
- Original report was 926 numerically in-range/resource-in-range trials with no accepted HLS timing field. **This does not prove timing failure**: earlier evidence/CSim/missing worst-case cycles may prevent the timing parser from running. Identify precise gate failures from original case files **before starting another large study**.

## Scientific objective stays exactly the same

Minimize **whole-network HLS-estimated, numerical/resource/HLS-timing qualified latency**:

```
qualified_latency_ns = source-attributed HLS worst-case top cycles * policy-requested clock period
R = -0.05 for unqualified; R = -0.005 for first feasible / no improvement;
R = ln(previous_best_ns / current_best_ns) - 0.005 for feasible improvement
```

No unreported surrogate HLS clock/latency, area/power optimization, fabricated numerical results or relaxed timing criteria. The policy still outputs a floating global requested period from 2–32 ns at 1 ps canonical Tcl resolution and compiler-legal full-network architecture.

## Sprint LR-1 — root-cause visibility without additional native calls

- Original `post_hls_env._receipt()` continues to validate source identity, source-pinned invocation, CSim numerical result, worst-case cycles, all five resource caps, and uncertainty-corrected HLS timing. Now records **a nonexclusive explicit list** of failed gates, original CSim status, raw worst-case cycles if parsed, source-native validity, warnings and timing exception text. Scientific qualification and latency remain unchanged.
- Read-only `qualification_funnel.py --study <old path>` reconstructs original test tasks, recovers each original native source receipt and aggregates failed-gate counts, with sample case paths; invokes **zero Vitis/Vivado** and modifies **zero original files**. An older source revision remains auditable with its original code; this diagnostic is an investigative report, NOT a substitute for its independent historical audit.
- **Storage update from user (2026-10-31):** old experiment directories were deleted to reclaim disk space. The prior independently verified audit receipt and exported analysis CSV/figures remain historical evidence, but **the old native cases are unavailable for reinspection/replay** unless separately backed up. Do not require the old 1,070-case qualification funnel as a prerequisite and do not claim it has been rerun. Instead, run a new original-HLS 22-case smoke first; inspect its source-pinned `qualification_gates_failed` and read-only funnel. If the dominant problem remains missing Vitis reports/cycles, fix that problem before launching the expensive full run.
- **Stop gate:** do not spend 1,090 new native invocations until a 22-case smoke produces independently audited cases and a meaningful rejection-gate breakdown, and the current free disk and observed smoke-case sizes support the larger campaign.

## Sprint LR-2 — state/action redesign

- Each observation still contains real variable-sized (currently padded to 4 operators) Functional Model IR GNN graph, directed edges, five resource caps, numerical tolerance and search status, but **also contains up to 30 authentic prior trial rows** with per-operator normalized architecture indices, requested clock, CSim error/availability, resource utilization, timing estimate/availability, qualified latency/availability and original E3 status. A learned attention pooling encodes this full bounded history; missing HLS values are explicitly masked, NEVER filled with imagined outcomes.
- Candidate architectures now include **global legal random-walk proposals and local refinements** as well as the baseline. The initial action is *not forced* to the baseline; every proposal passes real compiler legality checks before Vitis.
- Keep the joint action `(one discrete compiler-legal complete architecture, continuous learned global period)`. No clock grid, no per-network policy, no synthetic simulator QoR.

## Sprint LR-3 — learn from failed evaluations WITHOUT changing scientific reward

- The same shared actor/critic/GNN now contains a **native-feasibility auxiliary head** operating on candidate graph/history features plus requested clock.
- The head is trained with **actual original source-attested binary HLS qualification** labels, weighted BCE loss, coefficient **0.2** and positive-label class weight **8**. It influences architecture selection logits with strength **0.5** using the candidate's current predicted Beta-mean clock; the sampled clock is still sent to real Vitis. This is a **supervised auxiliary representation-learning signal**, NOT a surrogate latency, NOT a proxy HLS reward, and NOT independent pretrained DRL.
- One original online step still updates actor+critic. Logged total loss now equals actor + critic + entropy + 0.2 × supervised feasibility BCE; audit checks this against recorded native labels. If a binary label is missing, no synthetic target is supplied.
- **No learning-curve guarantee:** the supervised head can discover systematic feasibility structure only if qualification labels are meaningful and related to graph/knobs/clock. If the prior 926 failures represent missing HLS latency/report evidence rather than invalid optimization choices, first solve the root source/HLS problem.

## Sprint LR-4 — fair prospective evaluation

- Evaluate both a deterministic frozen actor (`frozen`) and a separately **stochastic frozen actor** (`frozen_stochastic`, categorical/Beta samples, **no gradient updates**) alongside `online_adapt` (same initial checkpoint and stochastic sampling protocol, real-gradient updates) on each held-out test task. The stochastic frozen/adaptive arms reset their Torch random seed at episode start to align their first action distribution; after adaptive updates, trajectories may diverge naturally.
- A **never-trained GNN** initialized with the same source-defined random seed (no gradient updates) is evaluated stochastically, to isolate the effect of 100-episode training from the inductive bias of the GNN architecture. Random, greedy, evolutionary/novelty, and Parzen-like references have distinct RNG streams and different feedback-based fallback algorithms even when no qualified incumbent exists. These are **reference heuristics**, not fully tuned Optuna/TPE competitors.
- Reports include **feasibility on every test task including censored zero-feasible cases**; produce paired latency effects only where both methods have qualified results under equal budgets, never fabricate speedups from no-feasible rows.
- Current cohort still has 1–2 training operators and 4 heldout test operators; **one test task/seed cannot establish transfer**. A later preregistered multi-seed, multiple untouched test-task campaign is necessary for a scientific superiority claim.

## Next experiment — explicitly chosen 100 online train × 10 HLS calls

- **100 train episodes**, **1 frozen validation**, **8 test/control arms** (trained deterministic frozen, trained stochastic frozen, untrained stochastic GNN, trained stochastic online adaptive, random, greedy, evolutionary, Parzen reference) on one untouched 4-layer test network = **109 total episodes**, **up to 1,090 original native Vitis attempts**. This is one-seed *pilot*, not independent replication.
- Proposed fresh output: `build/gym_100_train_10_hls_seed101_learning_v3`. Never reuse original `build/gym_100_train_30_hls_seed101_paper_v2`. Run source revision on clean checkout, unit tests, source-only plan, **small original-native smoke** (2 train +1 validation +8 test =11 episodes ×2 calls = **22** native invocations) and independent audit first.
- Every original `case_####`, `native_steps.jsonl`, `completed_episodes.jsonl`, `campaign.json`, checkpoints and source hashes are immutable with read-only audit and paper export.
- Metrics: train feasibility vs episode *and* calls, first-feasible evaluations/task, best-qualified latency within a task (not cross-task raw latency), fraction of tasks with any feasible, actor/critic/entropy loss, **supervised-feasibility loss**, **pre-evaluation probability/Brier score** and balanced trained-vs-untrained and frozen-vs-adaptive heldout comparisons. Present confidence/seed restrictions honestly.

## Commands (native not automatically authorized by this document)

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q rl_environment tests
python -m pytest -q tests/test_gym_graph_hls_contract.py
python -m pytest -q
# The old original case folders were deleted; this command must not
# be run against the old campaign without restoring its original native files.
# Estimate free space and report artifacts that still exist:
df -h .
du -sh build/* 2>/dev/null | sort -h | tail -20

python -m rl_environment.gym_study --train-episodes 100 --validation-episodes 1 \
  --test-tasks 1 --max-vitis-compiles-per-episode 10
# Must report 109 episodes / 1090 maximum native invocations and NO Vitis calls.

# Only after reviewing the qualification funnel and CPU tests:
python -m rl_environment.gym_study \
  --train-episodes 2 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 2 --seed 101 \
  --output build/gym_learning_smoke_2train_seed101_v3 \
  --execute-native --max-total-vitis-calls 22 --vitis-executable vitis_hls
python -m rl_environment.gym_audit --study build/gym_learning_smoke_2train_seed101_v3
# New small cases contain original gate receipts; classify them read-only.
python -m rl_environment.qualification_funnel \
  --study build/gym_learning_smoke_2train_seed101_v3
# Check smoke size before extrapolating the 1,090-case campaign;
# case sizes vary by network depth, so this is NOT a guaranteed estimate.
du -sh build/gym_learning_smoke_2train_seed101_v3
df -h .

# Only after independent native smoke audit, failure diagnosis and storage gate pass:
python -m rl_environment.gym_study \
  --train-episodes 100 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 10 --checkpoint-every-train-episodes 10 \
  --seed 101 --output build/gym_100_train_10_hls_seed101_learning_v3 \
  --execute-native --max-total-vitis-calls 1090 --vitis-executable vitis_hls \
  --timeout-seconds 7200

python -m rl_environment.gym_audit \
  --study build/gym_100_train_10_hls_seed101_learning_v3
python -m rl_environment.paper_evidence \
  --study build/gym_100_train_10_hls_seed101_learning_v3 \
  --output build/gym_100_train_10_hls_seed101_learning_v3_export
python -m rl_environment.gym_report \
  --study build/gym_100_train_10_hls_seed101_learning_v3 \
  --output build/gym_100_train_10_hls_seed101_learning_v3_report
```

**Verification status:** changes are committed to GitHub but Vitis and complete CPU tests are NOT executed by the assistant. Source-only dry run, local CPU suite, qualification funnel, and verified real 22-call smoke gate MUST precede the costly native study. There is no global campaign wall-clock stopping rule; per-invocation tool timeout remains separate.
