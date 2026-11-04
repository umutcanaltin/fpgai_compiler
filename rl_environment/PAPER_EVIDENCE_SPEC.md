# FPGAI Gym: publication evidence and reproducible learning-curve contract

**Applies only to the restarted, source-pinned native Gym study.** This file does **not** change the single latency-only reward or redefine the optimization target. The user explicitly agreed to stop the running study and start a fresh source-pinned experiment to collect complete policy diagnostics. Keep the stopped experiment directory immutable; do not retrospectively assign it policy losses that were not recorded.

## Immutable source and experimental configuration

Before the first native Vitis call, `campaign.json` captures `paper_evidence` with:
- exact clean Git SHA and SHA256 of `reward.py`, `gym_agent.py`, `gym_env.py`, `gym_study.py`, `clock_actions.py`, `post_hls_env.py` and `gym_audit.py`;
- the *actual* implemented reward formula, evaluation/infeasibility penalties, latency units, HLS timing, accuracy and resource qualification criteria;
- from-scratch shared GNN/architecture-and-clock actor description, optimizer, learning rate, discount, actor/critic/entropy terms, gradient clipping and update schedule;
- network cohorts/seeds, architecture/clock rules, per-episode 30-Vitis-attempt hard cap, total maximum original calls, original tool version per case, and the explicit test baseline/control arms;
- Python/OS, NumPy, PyTorch, Gymnasium and matplotlib versions when installed, together with decoded per-operator FPGAI architecture knob values (precision, integer bits, parallelism, PEs, SIMD, reuse/serialization, tiling, pipeline II) and per-layer input/output tensor dimensions.

### Actual reward (not paper approximation)

For each source-pinned original Vitis HLS CSim+CSynth result:
```
L_ns = original HLS whole-top worst-case cycles × policy-requested period in ns
qualified = numerical CSim error <= task tolerance
         AND reported resource occupancy <= each usable cap
         AND estimated clock <= (reported requested target − reported uncertainty)
         AND full original native source/clock/report authentication

if NOT qualified:                 R = -0.05
else if no previous feasible best: R = -0.005
else:                              R = ln(previous_best_ns / new_best_ns) - 0.005
```
A candidate that does not improve the best latency receives `-0.005`, regardless of its own absolute feasible latency. Only actual original Vitis-qualified latency enters this reward. LUT/DSP/BRAM usage is a **hard feasibility constraint**, not a minimization term. For exact reproducibility cite `source_snapshots/reward.py`, its SHA256, and the original experiment Git SHA.

### Raw private experiment evidence

- `native_steps.jsonl`: **every completed original HLS step**, source-attested native reward, requested float clock, cycle count, feasibility, qualified latency, running best, timing margin and original report/invocation hashes, plus `trial_evidence` (complete architecture indices **and decoded knob values**, original graph layer types and shapes, max CSim absolute error, all five HLS resource usages and limits, numerical tolerance, timing estimate, original tool duration). The journal also now records `cumulative_episode_reward`.
- `training_metrics` per `online_train`/`online_adapt` step: observed on-policy actor, critic, entropy and total losses, entropy estimate, joint action log probability, predicted value, discounted bootstrap target, advantage, gradient norm before clipping and update sequence. **Frozen/heuristic controls have null training metrics**, not fictitious zero loss.
- `completed_episodes.jsonl`: durable per-episode rewards, best qualified ns, original call count, task digest, partition, policy-update count and unique episode path. All completed train, frozen validation, frozen test, random, greedy, evolutionary, Parzen-reference and online-adaptive episodes are included.
- `seed_*/training_checkpoints/after_*_episodes/` and final `seed_*/checkpoint/`: private hash-bound shared GNN parameter snapshots + optimizer state (stored locally, not committed into Git).
- Each `case_####/`: original generated source, action/requested clock Tcl, manifest, numerical CSim reference, vendor logs, invocation metadata and original original-Vitis CSynth XML. A failed/interrupted case can have staged files but no completed reward; inventory it as attempted/unknown, **never assign a surrogate reward**.

## Paper export (separate, create-only, no original files changed)

```bash
# Provisional descriptive plots during an incomplete/stopped campaign:
python -m rl_environment.paper_evidence \
  --study build/gym_100_train_30_hls_seed101_paper_v2 \
  --output build/gym_paper_preview_v1 --allow-incomplete

# Final export: ONLY when summary.json exists AND gym_audit independently passes:
python -m rl_environment.gym_audit \
  --study build/gym_100_train_30_hls_seed101_paper_v2

python -m rl_environment.paper_evidence \
  --study build/gym_100_train_30_hls_seed101_paper_v2 \
  --output build/gym_paper_audited_v1
```

The exporter **never runs Vitis/Vivado** and never modifies any source study folder. Export paths must be new and cannot be overwritten.

Output:
- `evidence_manifest.json`, `campaign_snapshot.json`, `source_snapshots/{reward.py,gym_agent.py,gym_study.py,clock_actions.py}`;
- `native_trial_table.csv`: **all completed attempts**, feasible or infeasible, with per-trial configuration (both knob indices and decoded values), network dimensions, clock, cycles, latency, reward, CSim error, resources, HLS timing and source provenance;
- `native_attempt_inventory.csv`: staged original cases, including failed/interrupted invocations with no corresponding completed step journal;
- `episode_learning_curve.csv`: per-episode reward, feasible fraction, call count, first/best qualified latency, normalized within-task improvement, tool duration and training updates;
- `policy_update_curve.csv` and `reward_curve.csv`: per-update losses/entropy/advantage/gradients and reward trajectories (missing values remain blank);
- optional figures: training reward by episode, feasibility rate by episode, within-task improvement, actor/critic/entropy total loss vs update count.

**Strict labels:** incomplete exports are `PROVISIONAL_INCOMPLETE_NOT_INDEPENDENTLY_AUDITED`; completed outputs are issued only after a full independent replay with original HLS evidence. Missing outcomes remain null, not fabricated. A one-seed 100-episode study is an **engineering pilot, not sufficient statistical evidence** for improved transfer or generalization. Cross-network absolute HLS latency curves are not directly comparable without task normalization; the episode reward/feasibility figures are exploratory, not proof of held-out improvement. For publication, later run several seeds, multiple untouched test tasks and matched-budget original Vitis controls; report paired effects and uncertainty.

## Restart safety

Stop the current foreground experiment before pulling new source code. Do **not** delete or reuse its output folder; source-pinning requires an exact clean Git revision for each experiment. All new evidence fields appear only in a **fresh campaign** launched from the revised SHA. The existing experiment cannot legitimately acquire missing past training metrics by retrospective editing.

See [EXPERIMENT_100_EPISODES.md](EXPERIMENT_100_EPISODES.md) for the selected 100-training-episode/30-original-Vitis-calls setup and a separate small audited native smoke.
