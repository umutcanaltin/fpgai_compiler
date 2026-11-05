> **HW-RL-05 update:** The new `hardware_compare.py` can independently
> re-audit two matching complete frozen heldout original-HLS campaigns
> before creating censored-aware paired comparison files. The separate
> `hardware_checkpoint.py` adds tensor-only Adam/CPU RNG snapshots
> at completed episode boundaries. It **does not** implement automatic
> native campaign resume. Both modules and their CPU tests have not
> yet been verified on the user's PC.

> **5 November 2026 — HW-RL-05 online algorithms + bounded campaign
> runner source committed (workstation tests pending).** The user's
> HW-RL-04 workstation passed 40 focused tests, then 33 focused tests;
> one original Vitis 6.55 ns mixed-CNN action produced one independently
> audited negative reward (-0.05) and one genuine GNN optimizer update.
> Rejection was **only HLS scheduling**: 5.935 ns estimate versus
> 4.78 ns effective budget (−1.155 ns margin). CPU GNN operation was
> verified, not multi-network learning or physical speedup.
>
> New `hardware_algorithms.py` adds episodic REINFORCE-with-value-baseline,
> clipped PPO and an independent random control alongside the original
> one-step actor-critic. `hardware_campaign.py` introduces explicit
> source-only plans, source-pinned/opt-in multi-network original-Vitis
> episodes, task splits, persistent evaluation-only checkpoints,
> journaled native outcomes and read-only independent replay. **All
> new HW-RL-05 source and tests still require local verification.**
> See [prospective paper protocol](HARDWARE_RL_PAPER_PROTOCOL.md).

> **5 November 2026 — HW-RL-04 GNN hardware action integration added.**
> HW-RL-03 is now truly native validated on the user's PC: 2,032
> tests passed (17 skipped), and a qualified 15 ns single Gym step
> produced 9,690 ns HLS-estimated latency and −0.005 first-qualified
> reward. A new mixed Conv/MatMul/Gemm shared GNN reuses the original
> directed MessagePass, adds per-operator target embeddings and attention
> over up to 30 original audited trials. A new source-only preview CLI
> and explicit one-Vitis/one-gradient path are on the feature branch.
> **This newly added GNN/agent has not been workstation tested,
> trained, or proven to generalize.** Read
> [HW-RL-04 specification](HARDWARE_ACTION_SPACE.md).

> **HW-RL-03 native Gym source added, testing pending (4 November 2026).**
> The user completed a source-attested, independently audited 14 ns
> CNN Vitis HLS case: 646 cycles × 14 ns = 9,044 ns HLS-estimated
> latency, with +0.007 ns estimated scheduling margin. A new separate
> `GymHardwareNativeEnv` exposes typed graph/action masks and
> one fresh original Vitis call per explicitly authorized step,
> with the existing real-latency reward. The source and CPU-only
> tests now need workstation verification; a true Gym-native step
> has not been run yet. No agent learning or physical speedup claimed.
> See [HW-RL-03](HARDWARE_ACTION_SPACE.md).

> **HW-RL-02 now staged on the RL feature branch (2026-11-04).**
> The user confirmed `fpgai-rl-hardware-space` runs from `/tmp` and
> inspects real Conv/MatMul/Gemm source operators. A new optional
> `fpgai-rl-hardware-native` command reuses original Vitis HLS
> synthesis, numerical CSim and source/clock-timing evidence to
> evaluate one immutable hardware action state. Default is read-only;
> native execution is explicit and still **untested on the workstation**.
> No learning, physical Fmax or achieved hardware improvement is claimed.
> See [HW-RL-02 qualification plan](HARDWARE_ACTION_SPACE.md).

> **4 November 2026 CLI install correction.**
> HW-RL-01 passed 30 focused and 1,982 full CPU tests (21 skipped),
> but the installed `fpgai-rl-hardware-space` entrypoint initially
> could not import the top-level `rl_environment` package.
> Package discovery now explicitly includes both the root RL package
> and the `src/` compiler. Install editable again and run
> `tests/test_rl_installed_cli_packaging.py`, including subprocess
> checks outside the repository. This is a packaging change only;
> no native HLS execution or trained reward is claimed.

> **4 November 2026 — Active RL hardware action-space sprint (separate branch).**
> [HW-RL-01](HARDWARE_ACTION_SPACE.md) exposes the current compiler's
> eight real architecture knobs plus continuous requested-clock actions
> for mixed Conv/MatMul/Gemm graphs, with typed immutable states, source
> compilation, finite action masks and graph observations. The original
> Gym/agent/training evidence is unchanged. This new interface is
> **source-only, untested on the workstation and not yet a native
> reward-producing Gym environment.** Native HW-RL-02 integration and
> original HLS testing are separate acceptance gates.

**LATEST structured real-HLS Gym benchmark:** [GYM_CURRICULUM_L0_L6.md](GYM_CURRICULUM_L0_L6.md) defines L0..L6 immutable increasing complexity environments with source-attested native Vitis, four-to-eight architecture knobs, lock/unlock precision and clock, original CSim/HLS qualification, same-from-scratch GNN, per-stage original-reward-compatible method provenance (original non-curriculum reward unchanged), source-measured MAC-normalized dense HLS-ns reward only for stage runs. `gym_curriculum --list-stages`/`--plan-all` are source-only, `--stage L0` opt-in native requires explicit output and budget. `clock_mismatch_probe` shows actual target/requested XML pairs behind the 850 old `different_reported_target` failures. `gym_curriculum_report` independently audits stage results and pairs trained/untrained real qualified HLS latency within the same task. **No new stage CPU/native tests claimed passed**; run full tests and 22-case L0 audit before stage L0 100×10. Preserve old case folders.\n\n**New smallest real-Vitis GNN learning pilot:** [BEGINNER_MATMUL_100X10.md](BEGINNER_MATMUL_100X10.md). The explicit beginner mode uses one 4/8-wide MatMul, heldout small dimensions, fixed 32 ns/16-bit/6 integer bits, 4 architecture knobs, trained from scratch with ≤1000 original HLS training calls. New beginner reward gives each **strictly qualified** real-HLS cycle result a positive monotonic within-task latency learning signal; normal 5-layer compiler and old full-search reward are preserved. HLS 2023.2 Tcl sets exact 8.640ns margin and auditor accepts missing *XML uncertainty* only from source SHA-attested executed Tcl; report source/hashes/target/estimate still required. New native smoke and independent audit not executed on user workstation yet. Do not confuse previously successful Vitis subprocess exit with real qualified case; old 850 failures must be categorized and initial 22-case smoke must have >=14 qualified and no report-invalid cases before full 1090-call test.\n\n**1 Nov 2026: finished training interrupted during validation — DO NOT RESTART 1000 HLS calls.** Actual `gym_100_train_10_hls_seed101_learning_v3` log confirms 1000 training and nine frozen validation original Vitis invocations, then duplicate architecture/clock exception before summary. Preserve its original source-pinned journal/checkpoint unchanged. Read [INTERRUPTED_1000_CALL_RECOVERY.md](INTERRUPTED_1000_CALL_RECOVERY.md) for independent incomplete-campaign `gym_partial_audit`, provisional learning figures, and separate create-only **90-native-call** checkpoint-based validation/test continuation. Recovery evaluation Git SHA differs from training SHA, so do not claim a continuous 1090-case original-source audit.

**New 100×10 Gym RL learning redesign:** [SPRINT_RL_LEARNING_REDESIGN.md](SPRINT_RL_LEARNING_REDESIGN.md). Source-only changes add 30-step original-HLS history observations, compiler-validated global/local offers (no forced first baseline), native-feasibility auxiliary actor signal, explicit source/CSim/worst-case-cycle/resources/timing failure gates, stochastic frozen-policy evaluation and baseline no-incumbent fallback differentiation. Earlier 1,070-case pilot at `eb4a799` remains immutable and demonstrated **no reliable transfer-learning trend**. Use `python -m rl_environment.qualification_funnel --study build/gym_100_train_30_hls_seed101_paper_v2` to diagnose the original 926 missing timing/qualification cases without a Vitis call. The new campaign is **109 episodes including 100 training, at 10 native steps each = 1090 maximum Vitis calls**, pending CPU smoke and original-native audit. Do not infer success before running tests.

## HW-RL-04: install GNN runtime dependencies on the workstation

The installed GNN CLI requires **PyTorch**, **Gymnasium** and the ONNX
frontend. The default editable install with `--no-deps` intentionally
installs none of these optional extras; success of the CPU test suite
does not guarantee GNN readiness, because tests requiring PyTorch may
skip. Install into the **same activated virtualenv**:

```bash
python -m pip install -e '.[rl,onnx]'
python -c "import torch, gymnasium, onnx; print('torch', torch.__version__)"
python -m pytest -q tests/test_rl_hardware_optional_dependencies.py tests/test_rl_hardware_agent.py tests/test_rl_hardware_gym.py
```

CPU-only PyTorch is sufficient for the graph policy on a Linux CPU.
An alternative wheel source is
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
followed by `python -m pip install 'gymnasium>=1.0,<2'` if the
other dependencies are already present. Re-run the existing
`fpgai-rl-hardware-agent` command **without** `--execute-native`
first; this creates no original Vitis cases or gradient updates.
The CLI reports an explicit install instruction before attempting to
load the model when PyTorch cannot be imported. A corrupted PyTorch
installation must be repaired, not bypassed.

# Gym Graph RL — single active neural-network HLS optimization research track

**AUTHORITATIVE:** [RESEARCH_QUESTION.md](RESEARCH_QUESTION.md). The Gym/GNN/clock-aware source passed **31 focused and 658 full tests** on the user's workstation at `0cc2a22`. An actual Vitis HLS 2023.2 CSynth XML was parsed read-only, revealing that the old raw-target comparison was too permissive; the effective clock-margin gate was corrected and subsequently passed CPU tests. Later 100-episode campaign setup (max 30 HLS calls per episode) and checkpoint/audit updates need a fresh CPU retest. **No new source-pinned original Vitis timing qualification has run.**

## Selected experiment configuration

See [EXPERIMENT_100_EPISODES.md](EXPERIMENT_100_EPISODES.md). **100 training episodes**, plus **seven validation/test/control episodes**, max **30 original Vitis HLS attempts per episode** or **3,210 planned maximum total**; no campaign wall-clock stop. Optional average-compile-time forecasts are planning aids only. Private checkpoints every 10 training episodes, and append-only per-step/per-episode journals. CPU gate and small audited native smoke must precede the large explicit native run.

## What we are building

A Gymnasium-compatible environment with one neural network per episode. Every episode supplies a complete typed network graph, usable FPGA LUT/FF/DSP/BRAM/URAM capacities and a numerical error constraint. The environment outputs a graph observation, and a **single trainable GNN and shared DRL policy initialized from scratch** interacts with it, accumulating experience across networks.

The agent chooses a **complete per-layer architecture plus global requested HLS clock** at each step. The requested clock is *not* supplied as task input. The normal Gym study takes **no clock period or clock-list argument**; the shared policy now trains an architecture-conditioned **Beta distribution** to generate floating-point requested periods within an internal **2–32 ns** safety envelope. Values are canonicalized to **0.001 ns** for stable Vitis Tcl/report hashes. This is not a preselected clock grid or a claim of physical Fmax. Only original HLS synthesis qualifies timing feasibility. Original Vitis HLS CSim+CSynth provides fresh source-attested feedback. The only objective is the lowest **HLS-timing-qualified estimated network latency in ns**. The agent may use all usable FPGA resources and any precision that passes numerical correctness. No area/power/energy reward, no Vivado, no physically measured Fmax.

The environment **does not train itself**; a separate online RL trainer calls `step()`, updates GNN/policy weights after verified native feedback, and persists in-project training checkpoints.

## Paper evidence and learning curves

For the **fresh** source-pinned experiment (after stopping the earlier running campaign) the original `native_steps.jsonl` journal captures every completed architecture + floating-clock action, CSim error, original estimated resources, HLS timing and feasibility, the exact reward and cumulative reward, and per-update actor/critic/entropy/value/advantage/gradient diagnostics. Campaign metadata preserves exact source SHA256 hashes and the implemented reward/optimizer formula. `completed_episodes.jsonl` includes frozen test and adaptive arms. `python -m rl_environment.paper_evidence --study <path> --output <new folder>` is **read-only on the source study** and produces trial/reward/policy-update CSVs, optional learning-curve figures, and exact source snapshots after independent audit. For incomplete runs only, use `--allow-incomplete`, labeling results provisional. Refer to [PAPER_EVIDENCE_SPEC.md](PAPER_EVIDENCE_SPEC.md) and [EXPERIMENT_100_EPISODES.md](EXPERIMENT_100_EPISODES.md); never invent missing historical training diagnostics or commit large private native artifacts to Git.

## Current files and truthfulness

| File | Purpose | Status |
|---|---|---|
| `RESEARCH_QUESTION.md` | Canonical hypothesis, graph/clock/accuracy/reward and experiment definitions | Specification locked in conversation |
| `END_PRODUCT.md` | Acceptance criteria | Planned |
| `ROADMAP.md` | Engineering dependencies and milestones | Planned / tracked |
| `SPRINTS.md` | Current code/evidence versus future tasks | Current |
| `HANDOFF.md` | Safe next-chat continuation and branch/source pins | Current |
| `tasks.py`, `cohort.py`, `clock_actions.py` | Frozen model/tasks, strict 1 ps floating-clock actions, cohort | Learned clock with internal 2–32 ns safety bounds; restricted 1–4-layer source |
| `post_hls_env.py`, `audit.py` | Source-attested CSim/HLS step; legacy read-only cycle audit | Backend modified for clock-aware ns; legacy audit applies only to source-pinned archived pilot |
| `agent.py` | Retained legacy actor–critic with handcrafted graph features | Retired for new Gym research |
| `study.py`, `report.py` | Old report driver / dry-run only | Legacy native study disabled |

The original old-version real Vitis smoke passed an independent **16/16-case** source/numeric/resource audit and had a heldout **43-versus-43 HLS-cycle tie**. It is useful engineering evidence only, not evidence for the revised objective.

## Non-native verification

```bash
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_online_post_hls_env.py \
  tests/test_online_post_hls_audit.py \
  tests/test_online_post_hls_report.py
```

No implicit Vitis/Vivado calls. Use an installed research venv, not assumed system Python. **Do not run the old 100-network cycles-only command as the new paper study.**

## New modules and verification gates

`gym_env.py` implements the currently bounded graph Gym interface; `gym_agent.py` implements the shared trainable GNN and online trainer; `timing.py` and `reward.py` implement the ns feasibility target; `gym_study.py` provides the bounded dry-run/native campaign; `gym_audit.py` replays original native cases read-only, and `gym_report.py` produces only independently audited HLS-estimated results. `baselines.py` supplies random, local-greedy, evolutionary and Parzen-reference search heuristics. No real new study has been run.

## Next implementation gates

Focused Gym contract tests passed **31/31**, with **658/658 full tests** at `0cc2a22`. The actual original Vitis 2023.2 report then parsed successfully; recorded target **10 ns**, estimate **7.664 ns** and uncertainty **2.7 ns**. Effective HLS budget is **7.3 ns**; the report **misses by 0.364 ns**, and the corrected timing parser now marks it unqualified; those corrections passed on workstation. Later source changes for the 100-training-episode campaign and 30-call cap still require a fresh workstation test run. Inspect a completed *existing* original Vitis HLS case with the read-only probe below, then run a bounded new native smoke and independently replay every case.

```bash
# Inspection only: no Vitis invocation and no new science qualification
python -m rl_environment.timing --case /absolute/path/to/existing/HLS/case_0000
```

Use a REAL existing case folder containing `manifest.json` and `fpgai_hls_project/solution1/syn/report/fpgai_model_csynth.xml`. The probe checks field compatibility with the installed Vitis report and requested clock in the manifest. **Timing feasibility must additionally satisfy EstimatedClockPeriod ≤ TargetClockPeriod − ClockUncertainty; a missing margin is not assumed zero.** When an older parser rejects actual Vitis 2023.2 field placement, inspect the safe clock-only schema first:

```bash
python -m rl_environment.timing --case /path/to/real/case_0000 --inspect-schema
python -m rl_environment.timing --case /path/to/real/case_0000
```

Expected Vitis layout to check: user clock target under `UserAssignments/TargetClockPeriod`, estimated HLS clock under `PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod`. The new parser refuses ambiguous, duplicated, conflicting and unsupported evidence. The inspector deliberately exposes no arbitrary XML contents or private model data. **It cannot qualify a new Gym run, numerical correctness, source freshness, FPGA resource fit or physical Fmax.** Source ablation flags `--fixed-clock-ablation-ns 10` and `--no-message-passing` are available but unverified. Add a true tuned Bayesian/TPE baseline and complete matched native runs before the full multi-seed paper experiments. Add tests and new small source-pinned native audit before any large multi-network experiments.

Keep source-pinned original HLS receipts private; no public GitHub release until IP/license/history checks.
