> **Historical 30-call planning document.** The already completed independently audited **100-train × 10-call** experiment was source-pinned to `eb4a7991a5ec870f2b543412dc60dde2da237c41`; its native cases and paper evidence remain immutable. A **new** RL/history/feasibility learning revision with an explicit 100-train × 10-call budget and an additional stochastic-frozen test control (109 total episodes / max 1,090 calls) is defined in [SPRINT_RL_LEARNING_REDESIGN.md](SPRINT_RL_LEARNING_REDESIGN.md). Use **that document, not the historical 30-call commands below**, when planning the new run. New source still requires local CPU tests, old-case qualification-funnel diagnosis and a small native replay-audited smoke. No learning improvement has been demonstrated for the prospective code.\n\n# Native FPGAI Gym research: 100 training episodes (30 HLS calls each)

**Status:** A prior revision was already started on the user's workstation; **preserve its original output folder** when stopping. The replacement paper-instrumented run is **not executed or verified yet**. Earlier 31/31 focused and 658/658 repository CPU tests passed at `0cc2a22` before the episode-budget, diagnostics and exporter changes. A fresh CPU gate plus a small original-Vitis native smoke and independent audit are mandatory before launching the replacement 100-episode campaign.

## Required restart order for the paper-instrumented run

1. Stop the **previous native command** with `Ctrl+C` in its own terminal. The runner terminates its own Vitis subprocess group on interruption. **Keep the old `build/` directory**; it is incomplete/old-source evidence. Never overwrite or manually retrofit missing metrics.
2. Only **after it exits**, `git pull --ff-only origin main` into a clean source tree. The new source records reward definition, exact package/software versions, architecture knob values, policy losses/entropy and a durable trial journal.
3. Run `compileall`, all focused tests and full `pytest`. The **previous** tested revision had 31 focused/658 full passing; these additional paper-logging edits are not yet validated on the workstation.
4. Run the small 18-compile original native pilot below and verify `gym_audit` passes. If it fails, preserve its original cases and do not start 3,210-call study.
5. Launch the fresh 100-training-episode study under the **new `..._paper_v2` directory**. There is no time-based stop, and no change to the one latency-only reward.
6. Export source-pinned trial/reward/learning curves to a separate new folder after completion and independent audit; incomplete snapshots require `--allow-incomplete` and remain provisional.

## Scope and exact experiment counts

- **100 online training episodes** per seed, one distinct immutable network task each. The policy is the same randomly initialized GNN with learned architecture-conditioned floating-point HLS clock actions across those 100 episodes; training is from scratch, not imported/pretrained.
- **1 frozen validation episode** and **1 frozen test episode + 1 adaptive test episode + 4 test baseline episodes** (random, greedy, evolutionary and Parzen-reference) on one held-out test task.
- **107 total episodes**; **at most 30 original Vitis HLS CSim+CSynth attempts per episode**, including unsuccessful attempts, so **3,210 authorized maximum original Vitis invocations**. Real total may be lower if episodes exhaust legal actions or terminate early.
- No 8-hour campaign clock, time-based stopping condition, Vivado, post-route or Fmax claim. Each individual Vitis subprocess has a separate **7,200-second safety timeout**; its receipt/logs must survive timeouts.
- Periodic private checkpoint **every 10 completed training episodes**, plus a final training checkpoint; append-only `native_steps.jsonl` and `completed_episodes.jsonl` track source-attributed progress. Interrupted campaigns remain *incomplete and immutable*; **automatic resume is not yet supported**.
- HLS numerical, resource and **effective clock budget** (requested period minus original Vitis-reported uncertainty) must qualify before an estimated-latency observation is accepted.

## Calibrating runtime, not imposing a duration

An earlier **different, fixed-clock cycles-only pilot** recorded 450.353 seconds across 16 original Vitis calls, giving **28.147 seconds per call** (rounded forecast input: 28.15 seconds). For 3,210 calls this produces a rough **25.1 hours of original Vitis process time** alone. Different networks, HLS timing targets and device loading can materially change average compile time. Data preparation, GNN updates, disk writes and control overhead add time. This estimate **never** stops the experiment.

Use the value observed from a new small *native* preflight as `--estimated-seconds-per-compile` for a better forecast. This is purely advisory.

## Preflight — zero new vendor invocations

From a clean `main` checkout with the intended Vitis-capable Python virtualenv:

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q rl_environment tests
python -m pytest -q tests/test_gym_graph_hls_contract.py
python -m pytest -q

python -m rl_environment.gym_study \
  --train-episodes 100 \
  --validation-episodes 1 \
  --test-tasks 1 \
  --max-vitis-compiles-per-episode 30 \
  --estimated-seconds-per-compile 28.15 \
  --checkpoint-every-train-episodes 10
```

The last command must say `DRY_RUN_ZERO_VENDOR_CALLS`, `total_episodes_planned=107`, `maximum_original_vitis_invocations=3210`, `hard_cap_vitis_compiles_per_episode=30`, and `wall_clock_limit_seconds=null`. The planning step neither stages cases nor calls any vendor tool.

## Explicit foreground native experiment — ONLY after the CPU preflight succeeds

Choose a brand-new output folder; do not reuse a prior campaign directory or change source during the run.

```bash
cd ~/Desktop/github_projects/fpgai_research
python -m rl_environment.gym_study \
  --train-episodes 100 \
  --validation-episodes 1 \
  --test-tasks 1 \
  --max-vitis-compiles-per-episode 30 \
  --estimated-seconds-per-compile 28.15 \
  --checkpoint-every-train-episodes 10 \
  --seed 101 \
  --output build/gym_100_train_30_hls_seed101_paper_v2 \
  --execute-native \
  --max-total-vitis-calls 3210 \
  --vitis-executable vitis_hls \
  --timeout-seconds 7200
```

Here `--execute-native` is **explicit authorization** for up to 3,210 HLS subprocess invocations (no Vivado). `--timeout-seconds` applies to **each individual compile only**, not the whole campaign. Without `--execute-native` the same command only produces a plan.

**Do not start this large run before an independently audited 1–2-episode native smoke using a new output directory.** A CPU-green suite validates code paths, not original Vitis invocation, compilation feasibility, report attestation or throughput.

### First new native source-pinned smoke (2 training episodes)

This is a distinct, smaller foreground study with **2 training + 1 validation + 6 test/control episodes = 9 total**, at **2 calls per episode maximum = 18 Vitis invocations**:

```bash
python -m rl_environment.gym_study \
  --train-episodes 2 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 2 \
  --seed 101 \
  --output build/gym_float_smoke_2_train_seed101_v1 \
  --execute-native --max-total-vitis-calls 18 \
  --vitis-executable vitis_hls --timeout-seconds 7200

python -m rl_environment.gym_audit \
  --study build/gym_float_smoke_2_train_seed101_v1
```

These commands perform **real original Vitis HLS execution** only when explicitly issued on your workstation. If compilation, native source attestation or independent replay fails, preserve the smoke directory and resolve that issue before authorizing the 100-training-episode study. Each experiment uses a fresh output directory.

## Audit only a fully completed source-pinned study

See [PAPER_EVIDENCE_SPEC.md](PAPER_EVIDENCE_SPEC.md) for the full source-pinned per-step experiment-values contract: reward function/source hashes, complete architecture and numerical/resource/timing receipts, online actor/critic/entropy/value/advantage/gradient diagnostics, completed episodes, learning-curve CSVs and figures. Never claim missing historical diagnostics are present.

A paper-ready, create-only reward/learning curve export is made **after** the full independent audit passes:

```bash
python -m rl_environment.paper_evidence \\
  --study build/gym_100_train_30_hls_seed101_paper_v2 \\
  --output build/gym_100_train_30_hls_paper_export_v2
```

To inspect a stopped/incomplete campaign **without calling Vitis** or pretending it was independently audited, use a distinct output folder and add `--allow-incomplete`.


```bash
python -m rl_environment.gym_audit \
  --study build/gym_100_train_30_hls_seed101_paper_v2

python -m rl_environment.gym_report \
  --study build/gym_100_train_30_hls_seed101_paper_v2 \
  --output build/gym_100_train_30_hls_seed101_report_paper_v2
```

For incomplete/interrupted studies, inspect the immutable `campaign.json`, completed cases, `native_steps.jsonl`, `completed_episodes.jsonl` and per-episode checkpoints. Never claim that an incomplete or failed run passed the independent full-study audit.

**Important:** This campaign uses 100 *training* episodes and seven additional validation/test/control episodes. If the scientific protocol later requires exactly 100 episodes **including controls**, the input must be 93 training episodes; do not silently change the comparison allocation.
