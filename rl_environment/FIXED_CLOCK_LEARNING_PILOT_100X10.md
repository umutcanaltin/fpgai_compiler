> **Status:** The user completed the original **24 ns 22-call smoke** with all Vitis processes exiting successfully; native HLS qualification and audit are still not reported. For the NEW easier pilot (fixed **32 ns** + locked **16-bit/6-integer** format), use [FIXED_CLOCK_NUMERIC_LOCKED_PILOT_100X10.md](FIXED_CLOCK_NUMERIC_LOCKED_PILOT_100X10.md). This older 24 ns protocol remains historical. The previous 1009-case campaign showed 850 timing XML parser exceptions; clock relaxation cannot fix XML evidence by itself. Do not rerun or overwrite the completed 24 ns smoke.\n\n# First measurable RL-learning milestone: fixed-clock architecture-only 100×10 pilot

**1 November 2026 — prospective, deliberately easier study. No new Vitis run is claimed.**

## Research question and scientific boundary

Can **one shared from-scratch GNN policy** learn, across 100 changing one- and two-operator MatMul/Gemm networks and only 10 original Vitis HLS calls per episode, to choose compiler-legal hardware architecture candidates that improve **strictly qualified whole-network HLS latency** compared with its never-trained initialization and equal-budget search references?

This deliberately tests only architecture optimization. It **does NOT** claim the agent has learned the requested clock period, transfer to previously unseen operator families, or routed/physical Fmax. The original harder joint-clock experiment remains preserved.

### Reduce action complexity without manufacturing reward

- Use the EXISTING explicit `gym_study.py --fixed-clock-ablation-ns T`. No new HLS-mode implementation is required.
- All arms (100 train, frozen, untrained, adaptive and random/greedy/evolutionary/Parzen references) use the **same one source-recorded HLS target clock**. The sampled clock Beta policy is inactive in this ablation. The action is a full compiler-legal architecture from the standard menu.
- Objective remains **minimize** `worst-case original Vitis HLS top cycles × T` subject to exact original CSim numerical, LUT/FF/DSP/BRAM/URAM, source and HLS schedule timing qualification. Because `T` is fixed, minimizing latency within this experiment is exactly minimizing qualified HLS cycles.
- **Scientific reward unchanged**: -0.05 for unqualified; -0.005 for first feasible/no improvement; `log(old_best/new_best)-0.005` for strictly improving feasible latency. No synthetic reward, HLS proxy, altered failure status, selected best-looking seed or smoothing presented as measurement.
- Search still covers all currently configured architecture knobs for the one- and two-operator train networks; it is NOT equivalent to exhaustive search. A fixed clock may improve timing qualification but cannot repair missing original CSim/cycles/resource evidence.
- Training cohort is **not simplified** at the same time. Hold topology and numeric/resource constraints constant vs the existing experiment, so the clock ablation is interpretable.

## Choosing the fixed clock without looking at heldout performance

**Provisional engineering choice: T=24.000 ns** (single conservative, relatively relaxed HLS requested clock; *not* a known qualified Fmax). This is a controlled experimental setting, not a claim the part runs at any particular speed.

First inspect the **existing** independently replayed current campaign's original rejection gates:

```bash
python -m rl_environment.qualification_funnel \
  --study build/gym_100_train_10_hls_seed101_learning_v3
```

If the dominant hard gate is **missing CSim source, missing worst-case HLS cycles or missing LUT/FF/DSP/BRAM/URAM evidence**, abort the fixed-clock study and repair the compiler/report pipeline first. Choosing an easier clock cannot produce missing HLS reports. If verified HLS scheduling timing infeasibility dominates, a fixed relaxed target is worth testing. Do not choose T after inspecting final heldout outcomes.

## CPU-only preflight: zero Vitis calls

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m pytest -q
python -m rl_environment.gym_study \
  --train-episodes 100 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 10 \
  --fixed-clock-ablation-ns 24.000 --seed 111
```

Plan should indicate `fixed_clock`, one fixed period, **109 episodes and at most 1090 original HLS calls**. If the source scripts or study contract fails a focused test, do not use Vitis.

## New fixed-clock 22-original-call smoke

The only allowed source of measurements is actual foreground Vitis HLS, not a synthetic Gym simulator:

```bash
python -m rl_environment.gym_study \
  --train-episodes 2 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 2 \
  --fixed-clock-ablation-ns 24.000 --seed 111 \
  --output build/gym_fixed24_arch_smoke_seed111_v1 \
  --execute-native --max-total-vitis-calls 22 \
  --vitis-executable vitis_hls --timeout-seconds 7200
python -m rl_environment.gym_audit \
  --study build/gym_fixed24_arch_smoke_seed111_v1
python -m rl_environment.qualification_funnel \
  --study build/gym_fixed24_arch_smoke_seed111_v1
```

**Do not move to the expensive study** if the smoke fails unit/native audit, cannot produce the expected real HLS cycle/timing evidence, or indicates a dominant source/CSim/reporting problem unrelated to T. Verify 10 *distinct* legal configurations per training episode are reachable at fixed T; the environment's fixed-clock action mask must prevent duplicate architecture-clock trials.

## ONLY after verified native smoke: prospective 100×10 fixed-clock study

```bash
python -m rl_environment.gym_study \
  --train-episodes 100 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 10 \
  --checkpoint-every-train-episodes 10 \
  --fixed-clock-ablation-ns 24.000 --seed 111 \
  --output build/gym_fixed24_arch_100x10_seed111_v1 \
  --execute-native --max-total-vitis-calls 1090 \
  --vitis-executable vitis_hls --timeout-seconds 7200

python -m rl_environment.gym_audit --study build/gym_fixed24_arch_100x10_seed111_v1
python -m rl_environment.paper_evidence \
  --study build/gym_fixed24_arch_100x10_seed111_v1 \
  --output build/gym_fixed24_arch_100x10_seed111_paper_v1
python -m rl_environment.gym_report \
  --study build/gym_fixed24_arch_100x10_seed111_v1 \
  --output build/gym_fixed24_arch_100x10_seed111_comparison_v1
```

### How to claim genuine improvement, never just a prettier plot

Primary: trained-vs-untrained, *same heldout model*, *same fixed T*, *same original 10-HLS-call budget*, distinct trajectories, verify a) at least one feasible result, b) best feasible HLS ns, and c) source-attested numerical/timing/resource constraints. Compare against independent random and reference heuristics. A single seed/test task remains a *pilot*; no population-level superiority claim. The deterministic frozen and stochastic frozen arms answer different evaluation questions and must be labeled distinctly.

Secondary: first/last 50 feasibility percentages, episode-level reward on different tasks, median first-feasible HLS attempt, best-within-episode latency improvement, per-episode source-gate rejection cause, auxiliary feasibility discrimination measured on a fixed future set. **No guarantee of upward reward trend;** a flat curve is a real finding. Never renormalize or cherry-pick posthoc to depict learning.

Follow-up once architecture-only learning is demonstrated: (i) train/test on unseen same-depth topologies and widths, (ii) unfreeze clock within a narrow, pre-registered range, (iii) finally return to full 2–32ns joint optimization and harder 3/4-operator topology generalization. These are *different preregistered experiments*, not results to mix into the fixed-clock paper claim.

### Existing evidence remains immutable

Do not delete or rewrite `build/gym_100_train_10_hls_seed101_learning_v3` (1000 original training cases +9 interrupted validation cases, independently audited as INCOMPLETE), its original GNN checkpoint, or the provisional CSVs. The source-pinned recovery script for 90 heldout evaluation calls belongs to the earlier joint-clock experiment, not this new study.
