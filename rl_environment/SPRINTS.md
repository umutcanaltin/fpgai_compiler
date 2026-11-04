# HW-RL-04 shared hardware GNN actor–critic — 2026-11-05

The user verified the original HW-RL-03 Gym-native flow: **2,032 CPU
tests passed**, **17 skipped**, 160 source-legal actions and one qualified
Vitis case at 15 ns with 646 cycles, 9,690 ns HLS-estimated latency,
+0.105 ns scheduling margin and −0.005 first-qualified reward.

The isolated HW-RL-04 implementation reuses the historical directed
message-passing primitive but adapts it to the new typed tensor/operator
graph. Exact action target indices and up to 30 previous real audited
trial rows are now in the Gym observation. The shared actor has a
compiler-masked categorical edit head, conditional Beta clock output,
critic and feasibility auxiliary; gradient updates require the exact
live Gym case/reward and fresh independent source-attested audit.
Default CLI is source-only. All new CPU tests, installed CLI smoke and
single original native-agent step remain **pending workstation run**.

[Research question, evidence gates and commands](HARDWARE_ACTION_SPACE.md)

---

# HW-RL-03 native Gymnasium environment — 2026-11-04

The user's native HW-RL-02 CNN baseline at 14 ns has **qualified**
original-HLS estimated latency of 9,044 ns (646 cycles). Its 7 ps
effective-scheduling margin does not prove robust physical timing.
The independent read-only audit reproduced the result; the earlier
10 ns run was genuinely infeasible. Both original cases are preserved.

New isolated HW-RL-03 source introduces a bounded variable-size graph
Gym with typed architecture-choice masks, model-global numeric actions,
continuous requested-clock head, one immutable new original Vitis case
per authorized native step, exact source/sample/evidence replay and
the existing latency-only reward. Default preview is read-only.
**New tests and original Gym-native smoke remain pending**. No
existing GNN weights, old paper cases or native reward definitions
were rewritten.

[HW-RL-03 scope and commands](HARDWARE_ACTION_SPACE.md)

---

# HW-RL-02 source-bound native HLS evaluation — 2026-11-04

The user's installed CLI now successfully runs outside the repository
and inspected an eight-operator Conv→Pool→Conv→Dense ONNX model with
zero native calls. The new optional
`rl_environment.hardware_native` backend and
`fpgai-rl-hardware-native` CLI reuse the existing Vitis HLS package,
vendor runner, numerical CSim, source/evidence parser and strict
effective-clock timing checks for this exact operator-level RL design
state. Tests include negative tampering, fake vendor receipts, missing
XML, unknown metrics, missing samples, original-case preservation, and
opt-in native execution.

**Status: new HW-RL-02 code is not yet tested on the workstation; no
new real Vitis cases have been launched.** Native original call requires
explicit opt-in, verified resource budgets, fixed float32 sample cohort,
and fresh immutable output. See
[hardware action space and one-shot smoke plan](HARDWARE_ACTION_SPACE.md).

---

# RL interface development resumed — 2026-11-04 (new feature branch)

User explicitly authorized work on RL readiness and full **implemented**
hardware knob classes. Work only on
`feature/rl-hardware-knobs-environment-20261104` while source changes are
reviewed and tested. The **2026-11-01 freeze below is historical**; it
continues to protect prior original native case folders, old agents,
unbroken experiment methods, policy weights and evidence, not this new
source-only hardware-action adapter.

[New source-backed hardware-action sprint](HARDWARE_ACTION_SPACE.md):
HW-RL-01 covers eight existing compiler `ArchitectureRequest` fields,
model-global numeric type, per-Conv/MatMul/Gemm mapping, bounded candidate
enumeration/source masks, variable-size weight-private graph observations
and continuous requested-clock actions. Source-only code/tests are
implemented but **not yet workstation-verified**; no Vitis, Vivado or
training initiated. HW-RL-02 must bind the exact action state to original,
qualified, independently replayable Vitis HLS evaluation before a
reward-returning Gym `step` or online-learning campaign is allowed.

---

# RL development freeze — 2026-11-01

**Status: FROZEN for further RL development and new RL experiments.** Primary research priority is now the [new FPGAI compiler-IR paper](../paper/compiler_ir/README.md) and its [implementation sprints](../paper/compiler_ir/SPRINTS.md). Do **not** implement PPO/A2C/REINFORCE/SAC, broaden stages, alter Gym reward/agents/auditor, or start additional RL native campaigns during this freeze. The existing RL sprint history below remains **backlog only**, not an active commitment.

**Exception for the already-started L0 study:** the user's L0 graph-actor-critic smoke at source `5221094af585bd501e842deca94d19c2289bb628` completed 22/22 qualified cases with a passing independent audit; the user's already-running full L0 study (if still in progress) may finish unchanged and receive its read-only independent audit and evidence export. Do **not** stop it, change its source, pull remote `main` in that running workstation checkout, relabel earlier runs, or initiate any new RL experiment. Results of this old source-pinned L0 run are **not** results for the new compiler-IR paper. This administrative freeze changes no executable code.

---

**2026-11-01 guarded first-native experiment launcher:** Added `rl_environment/paper_campaign.py` with source-only default planning, explicit original-Vitis 22-call L0 smoke, strict independent original-case audit and read-only qualification funnel, and an automatically enforced same-source/seed/stage/algorithm smoke gate (exactly 22 original calls, >=14 qualified, zero timing parse exceptions) before optional 1,090-call L0 full campaign. New `tests/test_paper_campaign.py` covers dry-run, invalid IDs, smoke threshold, exact native budget and create-only output. Exact PC commands in [paper/rl_experiments/RUNBOOK.md](../paper/rl_experiments/RUNBOOK.md). The user's 757 CPU tests were for the **previous** `ca0b338` revision; this new launcher/tests are committed but NOT tested on their PC and no licensed HLS experiment has been run. Do not mix scientific algorithm changes into source-pinned campaign once started; PPO etc still pending real implementation and algorithm-aware independent-audit support.\n\n# Paper DRL comparison implementation sprints — 2026-11-01

Scope: implement and validate on source/CPU first; run **original licensed Vitis HLS only later on the user's PC**. Never mark an agent implemented solely for documentation or a registry name. Historical experiments/checkpoints are immutable.

| Sprint | Deliverable | Acceptance gate | Status |
| --- | --- | --- | --- |
| PAPER-01 | Audit baseline source binding, L0–L6 curriculum plan, agent registry and existing paper report path | Local source contract, 747 CPU tests (user reported), no Vitis during inspection | Existing source verified on user's PC |
| PAPER-02 | Protocol, fixed comparison matrix, immutable experiment IDs, matched seeds/tasks, native compile budget and manifest validation | CPU tests for duplicate IDs/seeds, invalid stages/agents/budgets; zero vendor calls | Source manifest planner and tests committed; workstation tests pending |
| PAPER-03 | Structured candidate-agent developer space, stable policy API, real paper template | Candidate agents not advertised as runnable | Documentation implemented |
| AGENT-01 | Graph PPO: joint masked architecture + Beta clock actor, true clipped loss, critic, entropy, checkpoint serialization | CPU gradient/checkpoint/action-mask tests, frozen evaluation, smoke with original HLS | Pending implementation |
| AGENT-02 | Graph A2C: shared GNN, policy/value losses and source-only verification | Same acceptance gates | Pending implementation |
| AGENT-03 | Graph REINFORCE: baseline and online learning without future reward leakage | Same acceptance gates | Pending implementation |
| AGENT-04 | Hybrid Graph SAC feasibility study, discrete/continuous masked policy and replay legality; only implement if justified by interaction cost | Explicit off-policy protocol and CPU tests before registration | Research design pending |
| ENV-01 | Finish stable agents/environments/experiments/infrastructure/evaluation module migration; compatibility wrappers | Full import regression, untouched source-pin/audit APIs | Pending |
| ENV-02 | New operator integration contract and full reference/CSim/HLS tests before inclusion in Gym tasks | Fail-closed unsupported operators, consistent old task SHA | Pending |
| PAPER-04 | Independent per-agent source-audit, paired heldout tables, confidence intervals, qualification funnels, reproducible figure export | No placeholders interpreted as results; no cherry-picking failures | Pending |
| PAPER-05 | Stage-by-stage L0–L6 source-only regression and original-native 22-call L0 smoke | Fully audited qualifying smoke and threshold gate | Guarded L0 launcher and CPU-only gate tests committed; workstation verification and original HLS pending |
| PAPER-06 | Full source-pinned comparisons: >=100 train episodes × exactly 10 real HLS calls **per stage/algorithm/seed** + matched heldout | Immutable original receipts, independent audit, negative-result disclosure | Pending native PC |
| PAPER-07 | Ablations message passing, trial history, classifier, fixed vs learned clock and L6 depth transfer | Each has separately versioned protocol and stable splits | Pending |
| PAPER-08 | Freeze paper methods/results and IP/security release checklist | All claims sourced to audited real HLS, no post-route Fmax claims | Pending |

Implementation order: PAPER-02/03 → AGENT-01 → AGENT-02/03 → ENV-01/02 → PAPER-04 → PAPER-05/06 on licensed workstation → PAPER-07/08. A full combinatorial L0–L6 × algorithms × multiseed campaign can require tens of thousands of licensed syntheses; never start automatically or claim feasible scheduling without user authorization.

Read [paper experiment protocol](../paper/rl_experiments/README.md) and [experiment matrix](../paper/rl_experiments/EXPERIMENT_MATRIX.md). Use `python -m rl_environment.paper_comparison --help` for CPU-only matrix manifest generation once implemented. Historical L0–L6 contracts and 100×10 source audit remain authoritative.

---

**2026-11-01 installed-vs-checkout native-Gym import remediation:** L0 100×10 plan and 734 CPU suite passed at prior revision, but first live L0 smoke failed with `write_vitis_package unexpected keyword clock_uncertainty_ns` despite source function already supporting it. Pytest injects source checkout with `pythonpath=["src","."]`, whereas `python -m` may import stale site-packages. New pre-native `verify_local_compiler_package()` ensures source-bound writer/Tcl and throws actionable editable-install guidance BEFORE creating an immutable study. New no-Vitis actual-package integration regression and source-binding tests. Required `python -m pip install --no-deps -e .`; verification `python -m rl_environment.runtime_contract`; rerun CPU tests and fresh L0 22-case source-audited smoke `v3`. Keep `v2` partial as evidence; no changes to source-recorded old cases.\n\n**2026-11-01 100-minimum-every-Gym-stage research gate:** User confirmed 716 CPU tests passed on c078203, all plans had 100×10. Requested all seven L0..L6 complete at least 100 episodes even when no learning. Enforced >=100×10 in plan, actual native progress/journal audit and final stage reporter, not just defaults. Preserved separate explicitly labeled 2×2/22-call `--smoke`. Fixed prospective L5/L6 0.01ns action clock quantization and 0.01ns explicit uncertainty Tcl to match Vitis original XML target/uncertainty precision; 911 original clock mismatch discrepancies (most only millips) are read-only historical evidence and are NOT retroactively edited. Added per-stage actual original HLS reward and feasibility CSV/PNG curves, all-seven completeness flag and three scientifically cautious learning-evidence statuses. Full local tests and L0 original 22-case smoke/audit pending after latest source revision. Seven complete stages require 7000 source HLS train calls plus up to 630 heldout controls.\n\n**2026-11-01 structured research curriculum sprint implemented in source (workstation verification pending):** Fix duplicate comma SyntaxError in `gym_study.py`. Add immutable stages L0..L6, from single small MatMul/four knobs, variable widths, 2-op, tiling/II, numeric choice, continuous clock and deep 3/4-op transfer; per-stage source hashed task cohort, allowable knob mask, stage-specific real-HLS dense nanosecond reward and audited native studies. Source-only `--list-stages`, `--plan-all`; no automatic native run or pretrained weights between stages. Add exact read-only requested/XML clock target diagnosis for 850 old mismatches. Add independent stage scorecards of all heldout feasibility and paired trained-vs-untrained HLS latency (missing qualifying pairs = no speedup). Full science/control details [GYM_CURRICULUM_L0_L6.md](GYM_CURRICULUM_L0_L6.md). Pending: local tests, inspect real mismatch magnitudes, fresh 22-call L0 native smoke+audit, then L0 100×10; never weaken time/source/numeric gates or claim proven learning.\n\n**2026-11-01 beginner evidence sprint:** User chose easier original-Vitis 100×10. Completed SOURCE IMPLEMENTATION (workstation validation pending): opt-in one MatMul with small repeated widths and never-seen heldout widths; lock 32ns/16,6 numeric, search only PE/SIMD/parallelism/reuse; source-attested explicit 8.640ns Tcl uncertainty (new cases only), XML target/estimate hash validated, fail closed on mismatch; beginner-only real-HLS qualified dense reward MACs/(MACs+HLS cycles), normal full-study best-improvement reward unchanged; independent paper metadata and replay audit plus source-only regression tests. New [BEGINNER_MATMUL_100X10.md](BEGINNER_MATMUL_100X10.md) hard gates: full CPU suite, timing root-cause category diagnosis, 22 original calls and strict independent audit, no invalid timing XML, >=14/22 qualified, then and only then 100 train ×10. Old incomplete and 24ns smoke preserved. Cannot promise 100-episode positive learning curve; judge trained vs untrained on heldout native HLS measurements.\n\n**2026-11-01 LR-easy sprint:** Implemented optional global numeric lock `--freeze-numeric-format` fixing 16 total and 6 integer bits, using user-requested conservative clock `--fixed-clock-ablation-ns 32.000`; catalog-based global precision/integer mutation disabled without changing full product/default 5-layer search or HLS compiler domain. Campaign source Task SHA opt-in, independent auditor and read-only gate funnel record/reconstruct exact configuration. Dedicated source-pinned experiment [FIXED_CLOCK_NUMERIC_LOCKED_PILOT_100X10.md](FIXED_CLOCK_NUMERIC_LOCKED_PILOT_100X10.md) is prospective, with first-gate 22 actual HLS calls and full 100×10 only after audit; 850/1009 old cases failed strict XML timing parser, so source evidence must be investigated. Existing 24ns smoke 22/22 Vitis exit=0 is NOT qualification.\n\n**2026-11-01 Learning-evidence sprint:** Diagnosed 100×10 HLS: 84 feasible/1000, identical first and second-half feasibility, 8 positive improvement episodes, and poor RAW Brier score explained partly by 8× positive-weighted BCE sigmoid. Implemented source-only ideal-weight correction/ROC-AUC and honest rolling paper visuals; plus checkpoint fixed-future 100-original-HLS-action observational feasibility probe (checkpoint 0..90, original cases 91..100, no Vitis). Full `gym_learning_diagnostics` and `gym_checkpoint_probe` CPU tests pending workstation verification. **Do not substitute a smoothed curve for demonstrated improvement**; use current study `qualification_funnel` to identify primary rejection gates; do not redesign latency-only reward before checking true source failures. Full causal comparison remains separate trained/untrained 90 native heldout calls from interrupted original checkpoint.\n\n**1 Nov 2026 urgent incident recovery sprint:** Complete training was successful for all 1000 original calls, but frozen validation crashed after 9 original calls from repeated architecture + canonical clock. Added duplicate-free action selection without altering latency-only reward; source-pinned incomplete-campaign 1009-case audit, create-only provisional paper curves, checkpoint-verified **90-call** separate heldout evaluation and independent audit/report. Source and method revision must be explicitly distinguished. Full instructions: [INTERRUPTED_1000_CALL_RECOVERY.md](INTERRUPTED_1000_CALL_RECOVERY.md). CPU/native recovery test suite on user's workstation is still pending; **do not rerun full training**.

**New sprint LR-1 to LR-4, source implementation committed (unverified):** fine-grained source/CSim/worst-cycles/resource/timing rejection diagnostics; read-only diagnostic on the NEW smoke (old source directories deleted); GNN policy attention over 30 source-grounded original trial steps; initial compiler-legal global and local offers; supervised original-HLS feasibility auxiliary (BCE weight .2, positive weight 8, action-prior weight .5); independent baseline RNG/fallback; stochastic frozen vs stochastic online-adaptive heldout comparison; paper CSV/curve updates, source-hash audit and tests. **Pilot remains immutable**, new 100 training+9 validation/test controls×10 =1090 maximum native calls. Full details [SPRINT_RL_LEARNING_REDESIGN.md](SPRINT_RL_LEARNING_REDESIGN.md). Must run all workstation CPU tests and disk budget, audit 22-call native smoke and use its fresh rejection receipts, then launch new source-pinned campaign. No assistant native tests have run.

# Sprint ledger — FPGAI Gym graph RL only

**Latest user confirmation:** 31 focused and 658 full CPU tests passed at `0cc2a22`, followed by explicit start of an original native 100-training-episode campaign. The user agreed to **stop and restart** so the paper study can capture new per-update diagnostics and source-pinned reward/optimizer metadata. New `research_metadata.py`, `paper_evidence.py`, native per-step metrics, frozen/adaptive completed-episode journal fix and extended independent audit are source-committed but **NOT YET workstation-tested or native-audited**. Preserve the first source-pinned run untouched; restart into `build/gym_100_train_30_hls_seed101_paper_v2` only after tests and a fresh small native smoke. See [PAPER_EVIDENCE_SPEC.md](PAPER_EVIDENCE_SPEC.md).


Last reconciled: **31 October 2026** on **main**. **User workstation has now passed 31/31 focused and 658/658 full tests at `0cc2a22`, including the working floating-clock dry-run and Vitis 2023.2 timing schema. Selected run is 100 training episodes + seven validation/test/control episodes, 30 original HLS attempts maximum per episode, 3,210 total planned maximum, and NO global duration stop. New plan-limit, private checkpoint/journal and audit changes are not yet workstation tested.** **Newest workstation gate at `b59af61`: 30/31 focused pass, 657/658 full pass, syntax compile and dry-run plan succeeded. Exactly one failing test sampled a padded architecture slot without an action mask. The test now uses Gymnasium masked Tuple sampling (mask applies to architecture, not the floating clock), and checks fail-closed behavior for padded slots. Changes are committed but pending workstation verification; native Vitis Gym study remains unexecuted.** **After the trainable floating-clock refactor, the user's workstation reported 18/31 focused pass and 645/658 full pass (13 failures), traced to one `gym_env.py` indentation typo and a stale case-sensitive test expectation. Both are corrected on main; additional architecture-menu test assertions were repaired. Latest changes have not passed a subsequent workstation pytest run. A CPU-only GitHub Actions workflow was added but failed before a runner/steps were allocated, so it cannot be counted as test evidence.** The user's workstation verified **24/24 focused and 651/651 full tests passing** at `dde0ba6`. The read-only original Vitis HLS **2023.2** CSynth probe then successfully parsed target 10 ns, estimate 7.664 ns and uncertainty 2.7 ns. AMD UG1399 defines effective scheduling period = 10 − 2.7 = **7.3 ns**, so the reported estimate **misses timing by 0.364 ns**: the previous parser's `feasible=true` was overly permissive. The effective-budget code, receipt/replay auditing and three tests then passed on the workstation; the newer 0.001-ns learned-clock actor, Gym action interface and associated tests are **NOT YET WORKSTATION TESTED**. No new native Gym HLS study has executed.

| Sprint | Current status | Remaining work |
|---|---|---|
| GYM-00 Specifications/cleanup | **CPU gate passed 651/651** at `dde0ba6`; Phi oracle and import guard restored | Rerun after strict HLS-margin patch |
| GYM-01 Gymnasium task/observation | 30/31 focused tests passed at `b59af61`; remaining masked-slot sampling test corrected, **not yet rerun** | Rerun 31 focused and 658 full, then Gym checker and native HLS boundary |
| GYM-02 Shared GNN | Source implemented in `gym_agent.py`; **unverified** | Gradient and heldout-size regression tests |
| GYM-03 Action + clock | RL clock output-only; source now includes learned architecture-conditioned **floating Beta clock policy**, 0.001 ns precision, 2–32 ns internal safety bounds, no clock grid; **new source/tests unverified** | Rerun focused/full tests; verify actual joint-gradient and original floating-clock Tcl/CSynth audit |
| GYM-04 HLS timing feasibility | **Actual Vitis 2023.2 XML layout verified** on workstation, but effective scheduling-budget correction was necessary (10−2.7=7.3 ns < 7.664 ns estimated). Strict parser and audit changes added, unverified | Rerun focused/full CPU tests, repeat real-case probe (now must return feasibility false), then original source-pinned native HLS |
| GYM-05 HLS-ns latency reward | Source implemented, fixture tests added; **unverified** | Test latency reversals and re-audit actual native cases |
| GYM-06 Online from-scratch trainer | Shared online actor CPU-verified at `0cc2a22`; latest 100-episode hard cap and every-10-episode private checkpoint/journal changes **unverified** | Retest CPU suite, run 1–2 episode original Vitis smoke, then 100 train + 7 evaluation/control episodes |
| GYM-07 New original Vitis smoke | **NOT EXECUTED** under new contract | New source-pinned, timed, clock-varying, separately audited Vitis cases |
| GYM-08 Generalization and adaptation | **NOT EXECUTED** | Holdout tests, frozen versus from-same-checkpoint online adaptation |
| GYM-09 Baselines and ablations | Random, greedy, evolutionary and *Parzen-reference* code added, **not yet executed** | Full independently tuned Bayesian/TPE remains; fixed-clock and no-message-passing source ablations were added but have not been tested or executed |
| GYM-10 Paper/evidence | New full step/action/clock/CSim/resource/timing/reward/loss/entropy/gradient source-pinned logging, create-only evidence exporter and graph outputs **implemented in source but unverified** | CPU gate, new native smoke and independent audit, pilot learning curves; multi-seed heldout comparisons required for paper claims |
| GYM-11 Public release | **BLOCKED** | Institutional/IP rights, license, history secrets/weights audit, authorization |

## Verified historical native engineering evidence — archived, not current hypothesis

- Original source SHA `d8bfdcc12a6e5bf9a1b6412c5cba6afbe87d2194`; independent audit SHA `623a0dc30fd9b762524644f90fbc9988ed1d4a59`.
- User's workstation at that audit revision: **10 targeted tests passed**; **2,156 full tests passed**.
- **16/16** original Vitis HLS cases source-, CSim-, numeric-, resource-qualified under old **fixed-clock, cycles-only** feasibility; zero numerical exceedances; 8 qualified baselines.
- **450.353 seconds** original vendor process time and 2 shared-policy updates.
- Heldout DRL **43 cycles** vs random **43 cycles** on one network: no DRL superiority established.
- The read-only original independent audit launched **zero** vendor programs and modified **zero** original files.
- This does **not** verify a message-passing GNN, new Gymnasium API, clock action, estimated HLS timing or latency-ns reward. No 100-network run under the new contract.

## Next actionable sprint

**Active experiment profile:** [100 training episodes / 30 Vitis HLS attempts each](EXPERIMENT_100_EPISODES.md). Dry-run forecasts only; no wall-clock stop. Total episode/call upper bound: **107 / 3,210**. First finish workstation CPU and a source-pinned native smoke, then run the full explicitly authorized campaign.


**Now:** Run the CPU fixture + full test suite in a clean main checkout; repair regressions; verify original Vitis XML timing fields against installed tool; execute one explicit small source-pinned native smoke and independently replay with `python -m rl_environment.gym_audit --study <path>`. Only after that expand budgets and experiments.

## Stop conditions

Never claim passed new main-branch tests without a workstation run. No native run is warranted merely because prior cycles-only experiments qualified. Do not use Vivado or surrogate latency as a fallback. The old v4–v9 experiments are archived Git history, not future research milestones.

## Branch cleanup completed 31 October 2026

The original cleanup removed 410 paths and retained 46 core test modules. One essential Phi-oracle test was accidentally removed and has been restored from history, giving a net 409 old-path deletions. Including the new Gym and import-integrity tests there are now **49 test modules**. After the assertion fix the workstation passed **15/15 focused and 642/642 full** tests at `f83e5ab`; after adding timing-probe tests, the workstation passed **17/17 focused and 644/644 full** at `a0f5220`. The installed Vitis 2023.2 target/estimate/uncertainty XML was successfully parsed and the raw fields verified on a real case at `01c7f87`, but true HLS scheduling margin is **negative**. Later strict timing corrections and three CPU fixture tests must be rerun before original native HLS. Historical cases remain immutable. The historical 2,156-pass count belongs to a different revision.
