# Preregistered beginner RL milestone — one MatMul, four knobs, 32 ns, 16/6

**2026-11-01. Source implementation committed. This is a prospective experiment, NOT a claimed successful learning result.**

## 1. Exact experiment and what it can prove

The standard FPGAI research compiler remains intact: System/Model/Math/Circuit/Physical and all numeric, timing, architecture search features are not removed.

This optional `--beginner-pilot` **only** changes the first small research experiment:

| Component | Beginner setting |
|---|---|
| Original source-of-truth | AMD Vitis HLS **original CSim + CSynth** |
| Training episode | **1** actual Functional Model IR MatMul (matrix row × small constant matrix) |
| Training widths | **4→4, 4→8, 8→4, 8→8**, repeated with independently seeded weights across 100 tasks |
| Validation widths | **4→6, 6→4** (unseen training shape combinations) |
| Test widths | **6→8, 8→6** (unseen training shape combinations) |
| Requested clock | **32.000 ns**, fixed target (not proven physical Fmax) |
| Numeric representation | **ap_fixed<16,6>**, all layers, locked |
| Architecture knobs agent can choose | **parallelism, PEs, SIMD, reuse/serialization** |
| Other hardware knobs | Untiled and no explicit pipeline-II setting (baseline locked) |
| Agent | One from-scratch shared GNN, no separate policy per task |
| Original training calls | **100 episodes × at most 10 new original Vitis calls** = ≤1000 |
| Heldout controls | 1 frozen validation + 8 test arms each at most 10 original Vitis calls |
| Whole study cap | **109 episodes / ≤1090 original Vitis invocations** |
| Architecture correctness | Real compiler resolve/refinement/emitter, source CSim, HLS cap and timing gates |
| Physical route claims | **None**. HLS estimated target-frequency and timing only |

This isolates whether a global GNN can learn a **conditional policy for designing small hardware MatMul accelerators**, not whether it generalizes to arbitrary AI networks or discovers an optimal system-level clock.

## 2. Scientific reward — deliberately new and denser

Historical full joint-clock experiments remain source-pinned with the original best-improvement-only reward.

For this **explicit opt-in beginner research experiment** only:

```
if not strictly HLS + CSim + numeric + resources qualified:
    R = -0.05
else:
    exact_MACs = input_width * output_width
    R = exact_MACs / (exact_MACs + original_Vitis_worst_case_cycles)
```

All qualified HLS cases deliver positive, differentiable-between-actions scalar feedback; for a fixed network this strictly rewards lower actual original HLS latency, because requested period is fixed at 32 ns. MACs come from Functional Model IR, **not a learned performance proxy**. No area or power is rewarded. The original best-latency per episode is still measured for heldout comparisons.

This reward measures the value of each **feasible candidate**, and is **NOT mathematically the same objective as optimizing only the best candidate among 10 trials**. That distinction is explicit in the campaign's scientific metadata and independent auditor. Never put the old and new rewards on the same axis as if their numerical scales were comparable.

## 3. Critical HLS timing source-evidence correction

The previous independently replayed 1009-case study had **850 original HLS timing parser failures**, 65 CSim/source/numeric failures and only nine verified HLS scheduling failures. The user's completed 24 ns smoke had 22 Vitis exit=0 cases, but **no final qualification/audit result has yet been shared**. Vitis subprocess success does not imply timing or numerical qualification.

AMD Vitis HLS 2023.2 supports `set_clock_uncertainty` and defines effective scheduling period as requested period minus uncertainty. The beginner pilot explicitly inserts **`set_clock_uncertainty 8.640`** into the generated **Vitis HLS Tcl**, after `create_clock -period 32`, equivalent to 27% uncertainty and **23.36 ns effective HLS scheduling budget**. This is a documented tool setting, not a physically proven board frequency. See AMD UG1399 2023.2 `set_clock_uncertainty`.

Only if **the original fresh HLS XML contains target and estimated timing but omits the uncertainty**, the audited parser may use the **explicit original generated and executed Tcl setting**, verified with:
- original immutable native manifest's `explicit_clock_uncertainty_ns`;
- exactly one `set_clock_uncertainty 8.640` command in the original Tcl;
- native invocation's `script_sha256`, before/after executed Tcl hashes, successful/stable tool execution;
- fresh original csynth XML and matching `fresh_report_sha256` plus strict `summary.hls_report_sha256`;
- matching requested target and **no contradiction** if the XML also reports uncertainty.

**Fail closed:** missing source attestation, contradicted XML uncertainty, missing target/estimated period, invalid CSim, missing resources, tool failure or a false original HLS schedule still cannot be qualified. Never assume uncertainty zero or fabricate HLS cycles. This new source rule does **not** retroactively change original studies.

Before any large run, inspect the earlier study (read-only, no Vitis) using:

```bash
python -m rl_environment.qualification_funnel \
  --study build/gym_100_train_10_hls_seed101_learning_v3
```

It now emits `timing_exception_category_counts`; if the dominant cause is NOT missing XML uncertainty, the source-level remedy above may not help. Diagnose that actual cause before 100 episodes.

## 4. CPU-only plan, zero vendor calls

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q rl_environment src tests
python -m pytest -q
python -m rl_environment.gym_study \
  --train-episodes 100 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 10 \
  --fixed-clock-ablation-ns 32.000 \
  --freeze-numeric-format --beginner-pilot \
  --seed 111
```

Verify `beginner_pilot=true`, `beginner_pilot_schema=single_matmul_four_knobs_v1`, four `searchable_architecture_fields`, source-fixed numeric format, and 109 episodes / maximum 1090 native calls. **Zero Vitis invocations** in plan and tests. CPU tests must pass before real native smoke.

## 5. Native 22-call smoke, source-pinned create-only

```bash
python -m rl_environment.gym_study \
  --train-episodes 2 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 2 \
  --fixed-clock-ablation-ns 32.000 \
  --freeze-numeric-format --beginner-pilot \
  --seed 111 \
  --output build/gym_beginner_matmul_32ns_smoke_seed111_v1 \
  --execute-native --max-total-vitis-calls 22 \
  --vitis-executable vitis_hls --timeout-seconds 7200

python -m rl_environment.gym_audit \
  --study build/gym_beginner_matmul_32ns_smoke_seed111_v1

python -m rl_environment.qualification_funnel \
  --study build/gym_beginner_matmul_32ns_smoke_seed111_v1
```

**Pre-registered smoke stop-gate:**
- Independent read-only HLS case audit passes.
- Every model has one MatMul, and every architecture action locks 16/6, tiling and pipeline II.
- Every native script includes the source-attested `set_clock_uncertainty 8.640` and original XML contains valid target+estimate; none of the new cases fail `ORIGINAL_HLS_TIMING_REPORT_INVALID`.
- At least **14/22 original cases** strictly qualify numerical+resource+HLS timing, including some qualified training cases, before committing to a 100-episode campaign. This is an engineering feasibility preflight threshold, **not an eventual published learning result or a statistical inference**.
- If the gate fails, diagnose/repair; do not quietly loosen it, sweep after observing test performance or overwrite original case files.

## 6. Fresh 100×10 source-pinned experiment — only after smoke gate

```bash
python -m rl_environment.gym_study \
  --train-episodes 100 --validation-episodes 1 --test-tasks 1 \
  --max-vitis-compiles-per-episode 10 \
  --fixed-clock-ablation-ns 32.000 \
  --freeze-numeric-format --beginner-pilot \
  --checkpoint-every-train-episodes 10 --seed 111 \
  --output build/gym_beginner_matmul_32ns_100x10_seed111_v1 \
  --execute-native --max-total-vitis-calls 1090 \
  --vitis-executable vitis_hls --timeout-seconds 7200

python -m rl_environment.gym_audit \
  --study build/gym_beginner_matmul_32ns_100x10_seed111_v1
python -m rl_environment.paper_evidence \
  --study build/gym_beginner_matmul_32ns_100x10_seed111_v1 \
  --output build/gym_beginner_matmul_32ns_100x10_paper_v1
python -m rl_environment.gym_report \
  --study build/gym_beginner_matmul_32ns_100x10_seed111_v1 \
  --output build/gym_beginner_matmul_32ns_100x10_report_v1
```

## 7. Evidence needed to claim learning

**Primary:** trained GNN vs never-trained GNN and equal-budget independent random/reference arms on the **same untouched small heldout MatMul** at 32 ns, 16/6, using original qualified HLS latency. Show number of valid cases, best qualified HLS ns and first-feasible calls. **One seed / one heldout test task is a demonstration pilot, not generalization proof.**

**Secondary:** proportion of qualified trials, source-error breakdown, real-reward and 20-episode rolling trend, mean qualified cycle efficiency, original policy entropy/loss and positive-label feasibility discrimination. A curve can remain noisy even if heldout policy improves; never force an increasing curve by plotting or cherry-picking. After a positive pilot, expand to multiple unseen width combinations and random seeds before making population claims.

**Progression:** beginner one MatMul / four knobs / fixed clock and numeric => 1–2 operators and more knobs => narrow learnable clock => full joint-clock search and deeper heterogeneous networks. The complete FPGAI/Morfics five-layer compiler product goal is unchanged.
