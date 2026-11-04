# Recovery of the 1 November 2026 100x10 Gym learning experiment

## Verified incident and preservation

Original user's foreground terminal log contains **1,009 successful original Vitis invocations (exit=0)** and then a Python exception during frozen validation:

```
ValueError: Repeated architecture-clock pair is not a fresh native case
```

The traceback points to `gym_study.py` line 280, the **validation_frozen** episode. Therefore all 100 training episodes (1,000 original calls) and nine validation calls completed before a duplicate deterministic architecture-clock proposal; none of the eight test controls ran. The original study's `summary.json` is absent, so **the original full 1,090-call single-revision audit was NOT completed**. Do not claim it was fully independently audited.

**DO NOT delete, modify, overwrite, or resume in place**:

```
build/gym_100_train_10_hls_seed101_learning_v3
```

It contains the exact source-pinned 1,000-step training journal and private checkpoint at `seed_101/checkpoint`. Its original source commit is captured in its `campaign.json`. It is essential to preserve these even when disk space is limited.

## Fixed source bug

`gym_study._choose_unused_hls_action` now checks the original backend's **already executed** `(architecture, canonical requested clock)` pairs before invoking Vitis. A frozen deterministic policy's repeated exact Beta-mean clock uses the nearest unused one-picosecond requested period for that same architecture. A baseline reference uses the same deterministic duplicate-clock tie-breaker; a stochastic policy may resample using its genuine action distribution and retain the final selected action's gradient tape. This is an explicit *duplicate-candidate constraint*, not synthetic HLS reward or repeated licensed invocation.

The fix is at a NEW Git commit and may not be retroactively represented as the old original experiment's code. It also canonicalizes raw binary floating periods to the exact 1-ps HLS grid **before** detecting duplicates, so `7.2010000000000005` correctly compares as the already tried `7.201`. This is a no-Vitis pre-action validation correction.

## Step 1 — CPU verification and provisional training curves, zero licensed calls

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q rl_environment tests
python -m pytest -q tests/test_gym_graph_hls_contract.py tests/test_gym_eval_recovery.py
python -m pytest -q
# Independently recheck 1000 completed training and 9 partial validation cases.
# This explicitly returns full_campaign_completed=false and invokes NO Vitis.
python -m rl_environment.gym_partial_audit \
  --study build/gym_100_train_10_hls_seed101_learning_v3
python -m rl_environment.paper_evidence \
  --study build/gym_100_train_10_hls_seed101_learning_v3 \
  --output build/gym_learning_v3_training_provisional \
  --allow-incomplete
python -m rl_environment.gym_learning_diagnostics \
  --export build/gym_learning_v3_training_provisional
```

The training-learning diagnostic prints comparable first/last 25 and 50 episode feasibility fractions, mean per-task episode returns, original-HLS feasibility prediction Brier scores, and a constant-infeasible prediction reference. These are **descriptive metrics, not heldout evidence that GNN training helped**. It uses the existing exported CSV files and makes no native calls.

The read-only partial-case audit reconstructs every original source-pinned native case and qualified HLS reward while explicitly reporting that the full 1,090-case campaign is incomplete. The provisional export is **not a full study audit**: it extracts recorded source-pinned native HLS trial rows, 100 training episodes and the 9 recorded validation calls, reproduces learning curves including auxiliary feasibility losses/Brier scores, and retains the distinction between complete training and incomplete validation/test.

## Step 1b — a real learning diagnostic without more Vitis calls

The old reward curve alone cannot demonstrate improvement: the study produced only 84 qualified / 1000 training cases, with 8 episodes registering a within-task latency improvement, and first-half versus second-half feasibility was equal (42/500 each). The auxiliary feasibility head used positive-weight=8 binary cross-entropy; raw sigmoid probabilities are not calibrated probabilities, so raw Brier around 0.192 should be interpreted with caution. Analytical correction is `p=q/(8-7*q)`, where q is the model's weighted sigmoid. This correction assumes the ideal weighted-loss relationship; it is **not heldout calibration** and can still be wrong if the model is misspecified. Also plot **ROC AUC** to test rank discrimination, which is invariant to this monotonic correction.

```bash
python -m rl_environment.gym_learning_diagnostics \
  --export build/gym_learning_v3_training_provisional \
  --output build/gym_learning_v3_calibrated_plots_v1
# Diagnose source-attributed rejection stages over the STILL-EXISTING
# current 1009-case study, not previously deleted old experiments.
python -m rl_environment.qualification_funnel \
  --study build/gym_100_train_10_hls_seed101_learning_v3
```

**Fixed-future, no-Vitis checkpoint probe:** Hold the same 100 original architecture-clock-HLS feasibility outcomes from training episodes 91–100 fixed. Evaluate only the untrained shared GNN and checkpoints after episodes 10,20,...,90. None of those checkpoints trained on the fixed 100 cases. Evaluate learned feasibility *ranking* (ROC AUC) and analytical-class-weight-corrected Brier. **This is an offline source-measured observational classifier test, not a policy-level HLS latency speedup or unseen untouched test-cohort search;** later outcomes were chosen by a later policy, so selection bias remains. All checkpoint probes use an intentionally uniform empty-history context, not reconstructed original adaptive histories.

```bash
python -m rl_environment.gym_checkpoint_probe \
  --study build/gym_100_train_10_hls_seed101_learning_v3 \
  --output build/gym_learning_v3_fixed_future_probe_v1
xdg-open build/gym_learning_v3_fixed_future_probe_v1/checkpoint_feasibility_roc_auc.png
xdg-open build/gym_learning_v3_fixed_future_probe_v1/checkpoint_feasibility_corrected_brier.png
```

If learning is genuine, the classifier should discriminate future original feasible vs infeasible cases better than an untrained GNN (ROC AUC above 0.5, ideally improving over checkpoints), and corrected Brier should beat a constant feasibility reference. **Do not claim a positive trend when metrics do not show one.** The separate source-pinned trained-vs-untrained 90-call heldout HLS search is still needed to test whether learning improves the actual search policy.

## Step 2 — standalone heldout evaluation dry-run, no licensed calls

```bash
python -m rl_environment.gym_eval_recovery \
  --training-study build/gym_100_train_10_hls_seed101_learning_v3
```

Requires all 100 training episodes, 1,000 native training journal entries, and the unchanged original source-identified, SHA256-verified trained GNN/optimizer checkpoint. Expected maximum **90 new** original HLS calls = ten validation and 80 test/control calls; it will **not rerun 1,000 training steps**.

## Step 3 — NEW source-pinned heldout evaluation, only after CPU tests/dry-run

```bash
python -m rl_environment.gym_eval_recovery \
  --training-study build/gym_100_train_10_hls_seed101_learning_v3 \
  --output build/gym_learning_v3_eval_recovered_v1 \
  --execute-native --max-total-vitis-calls 90 \
  --vitis-executable vitis_hls --timeout-seconds 7200
```

This loads the **original trained private policy** and fresh optimizer state for online adaptation; frozen stochastic/online adaptive and untrained stochastic/method baselines run against the *same* original heldout cohort. It records both the original training source SHA and the NEW evaluation source SHA and the exact checkpoint SHA256. It writes its own `campaign.json`, `native_steps.jsonl`, `completed_episodes.jsonl` and `summary.json`, separate from and without rewriting the interrupted old study.

## Step 4 — independently audit new evaluation, produce comparison

```bash
python -m rl_environment.gym_eval_recovery \
  --study build/gym_learning_v3_eval_recovered_v1 --audit
python -m rl_environment.gym_eval_recovery \
  --study build/gym_learning_v3_eval_recovered_v1 \
  --output build/gym_learning_v3_eval_recovered_report_v1 --report
```

The new evaluation audit independently reconstructs native C++ source using exact compiler knobs, verifies original Vitis case receipts and strict numerical/resource/HLS-clock constraints, recomputes scientific latency rewards, checks case and episode journals against each other, verifies original checkpoint SHA256 and asserts zero native tool calls during audit. It **does not retroactively claim a full independent replay of the old 1,009-case interrupted experiment**; an original-revision source-pinned replay of that incomplete study would be a separate investigation.

The report `heldout_comparison.csv` includes feasibility, HLS ns, 10-call budget, reward and policy updates per method. Compare it with the **provisional** original-training learning curves and state clearly that the two phases came from **different Git commits** because the original frozen-policy duplicate bug was fixed.

### Scientific interpretation

This is a recovery study, not an exactly equivalent 1,090-call uninterrupted experiment. The original 1,000 training calls count as historical evidence only; the newly changed duplicate-handling rule must be clearly declared as a protocol deviation for deterministic validation evaluation. A single seed and heldout task cannot establish population-level superiority or generalization. Plot reward trends and pre-HLS Brier/feasibility calibration honestly; do not guarantee rising rewards. Do not reuse or delete prior evidence until extraction, audit and backup.
