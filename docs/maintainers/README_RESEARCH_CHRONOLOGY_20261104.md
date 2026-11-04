<!-- Historical preservation copy, not current community guidance. -->

> **Latest C1/C2 compiler-paper evidence (4 November 2026):** R3g **133/133 independent finite event-oracle comparisons**, R3h **48 source action catalogs with no extra cached selection compilation**, **1,751 full regression tests passed** in the author's original worktree at `3309076`. These observations are source-only, not trained-agent superiority or routed FPGA QoR. New code for R3j (216 planned held-out traces), R3k (192 planned matched source-limit action checks), and R3l (read-only publication table release) is in `main` but **has not yet been workstation-tested**. [Compiler paper research status](paper/compiler_ir/C1_C2_COMPILER_IR_FINAL_SCOPE_AND_ACCEPTANCE_GATES_20261103.md) · [active manuscript](paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md). External independent novelty comparison is still required.

> **Current compiler IR research direction (4 November 2026):** Two contributions **C1 bounded typed operator/Phi source event ownership** and **C2 learning-compatible, policy-neutral source state/action/mask interface**; trained RL policy results are deferred to a second paper. Workstation original results: **1,705 tests passed**, R3d E0 **3/2/1**, R3e **5/5** finite oracle + **25/25** selected mutants, R3f **21/24/3 of 48**. See [active compiler IR manuscript](paper/compiler_ir/C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md) and [results checkpoint](paper/compiler_ir/R3D_R3E_R3F_ORIGINAL_C1_C2_WORKSTATION_RESULTS_20261104.md). New R3g/R3h code remains untested on the author's PC; external prior-art parity and generalization are open; **not submission-ready**. Older checkpoints below are historical.

> **Latest compiler-paper original results (3 November 2026):** 1,649/1,649 full Python tests passed; original mixed CNN/Dense R3a completed **6/6 original Vitis attempts, 6/6 E2 CSim, 4/6 strict E3 CSynth**, and **2/3 original matched E3 pairs**. Both independent-Phi variants in E3-qualified pairs used more DSP/LUT/FF and worst cycles, with HLS BRAM18K ties. Both strided-CNN arms (`case_002/003`) passed E2 but failed the 9 ns effective clock gate (9.181 ns reported), and their HLS resource/latency pair stays NULL. **No new routed E4, novelty superiority or measured power claim.** The [new R3c full-cohort export](paper/compiler_ir/R3C_COMPLETE_CNN_HLS_PUBLICATION_RESULTS_20261103.md) is source-only and not yet workstation-tested. **Scientific paper status: MAJOR REVISION.** Older checkpoints below describe earlier stages and are superseded.

> **Latest compiler-paper original results (3 Nov 2026):** 1,633/1,633 pytest passed, 12/12 original mixed CNN/Dense GCC source checks, **three** original R3a Vitis cases attempted: two E3-qualified, third passed E2 but failed strict effective-clock E3; one qualified paired HLS comparison (BRAM Δ0, DSP +48, FF +1796, LUT +9146, worst cycles +287). Cases 003–005 remain unattempted in the supplied log. New opt-in R3b continuation for these cases preserves the failed original; not yet workstation tested. [Protocol](paper/compiler_ir/R3B_PUBLICATION_NATIVE_COMPLETION_AND_EVIDENCE_20261103.md). **Paper MAJOR REVISION; no routed E4 claim.**

> **Compiler paper checkpoint, 3 Nov 2026:** Source revision `f665b62` passed 1,626 full tests and 12/12 original GCC mixed CNN/Dense checks. R3a `case_000` then passed original Vitis E2/E3 with **1 HLS-estimated BRAM18K**; other five original native cases/three matched pairs remain incomplete and physical E4 is absent. See [R3b native completion and publication evidence](paper/compiler_ir/R3B_PUBLICATION_NATIVE_COMPLETION_AND_EVIDENCE_20261103.md). Paper status: **MAJOR REVISION**, not published. The R3b implementation and tests have not yet been rerun on the workstation.

## November 3, 2026: mixed CNN/MLP compiler source path and strict paper gate

**New source functionality (committed, not yet workstation verified):** `from fpgai_research.compiler import compile_neural_chain, compile_onnx_neural_chain` supports strict static CNN→pool/activation→Flatten→multiple fully connected Gemm/MatMul neural chains through the EXISTING Model IR/resolver/Conv/Dense HLS emitters. The new R2z original source campaign is **3 network models × 2 independent operator-mapping policies = 6 source packages**, with **12 planned actual GCC per-stage numerical executions**; the opt-in R3a native campaign uses those exact cases for 3 matched scalar/Phi Vitis HLS E2/E3 comparisons. **NO R2z/R3a original results or native resource improvements have been observed yet.** Run the one-command **source-only** gate from repository root after checking Git state: `bash scripts/run_compiler_ir_publication_source_gate.sh`; this never calls Vitis/Vivado. Vendor E2/E3 is a separate opt-in step after source qualification. See [R2z](paper/compiler_ir/R2Z_MIXED_CONV_DENSE_NEURAL_NETWORK_PROTOCOL_20261103.md), [R3a](paper/compiler_ir/R3A_CNN_DENSE_ORIGINAL_HLS_NATIVE_PROTOCOL_20261103.md) and [publication decision gate](paper/compiler_ir/SUBMISSION_DECISION_GATE_20261103.md).

**Prior original evidence remains frozen:** R2x source graph 32/32 cases and 64/64 contrasts, R2v 57 E3 qualified +7 unknown matched HLS factor pairs; first R2y Vivado `case_000` routed with disclosed DRC warnings and is **NOT clean E4** (8/9 original physical cases unattempted). No physical BRAM speedup, trained RL improvement, new vendor runs, GitHub Actions or accepted scientific paper is claimed. The compiler manuscript remains **MAJOR REVISION**, with closely overlapping Allo, StreamTensor, Filament, CODO, SODA and hls4ml literature. See [concrete November 2026 prior-art matrix](paper/compiler_ir/RELATED_WORK_COMPILER_IR_NOV2026_VERIFIED_MATRIX.md).

## Current research priority (2026-11-01): FPGAI compiler-IR paper

**New paper:** [FPGAI: An Architecture-Semantic Intermediate Representation for FPGA Neural-Network Compilation](paper/compiler_ir/README.md). The proposed central research contribution is typed operator-local space–time architectural mappings **composed across graph edges** and refined into real generated HLS. See the [compiler-first sprint backlog](paper/compiler_ir/SPRINTS.md), [source capability audit](paper/compiler_ir/SOURCE_CAPABILITY_AUDIT.md), [preregistered fast experimental plan](paper/compiler_ir/EXPERIMENT_PLAN.md), and [compiler-only workstation runbook](paper/compiler_ir/RUNBOOK.md). A full frozen-source backup branch is retained at [`backup/rl-environment-frozen-2026-11-01`](https://github.com/umutcanaltin/fpgai_research/tree/backup/rl-environment-frozen-2026-11-01). The paper evaluates compiler expressiveness, graph composition, lowering conformance, a restricted-IR ablation and a small source-attributed native HLS sample. All claims are prospective until implemented and independently validated. The IR is *designed to offer* a graph-structured, compiler-legal interface suitable for **future RL/search**; no learning benefit or RL superiority is claimed.

**RL track frozen for new work:** RL agent/environment development and additional RL experiments are on hold. A previously started L0 original-Vitis full campaign may finish and be audited on its existing source-pinned local revision (`5221094af585bd501e842deca94d19c2289bb628`). **Do not pull remote `main` into that active local checkout** or modify tracked code until it finishes and the original case evidence has been independently audited. GitHub planning docs have been committed directly to remote `main` as requested; they do not update the running workstation's local checkout. See [RL sprint freeze](rl_environment/SPRINTS.md).

---

**Contributing research extensions:** [Contribution guide](CONTRIBUTING.md) · [research module architecture](docs/RESEARCH_EXTENSION_ARCHITECTURE.md) · [new compiler operator/layer guide](docs/COMPILER_OPERATOR_CONTRIBUTION.md). The repository separates source-pinned DRL agents (`rl_environment/agents/`), Gym environment registration (`environments/`), reproducible experimental protocols (`experiments/`), evidence/infrastructure and typed compiler passes (`src/fpgai_research/`). Existing old imports and original Vitis evidence remain intact. If `python -m` imports a stale sibling checkout, the package-level import bootstrap selects the current repository's `src/` and the native source guard verifies it before any vendor call; new tests simulate this mismatch in a separate Python process.\n\n**Progressive source-attested RL Gym benchmark (1 Nov 2026):** [Seven-stage original-Vitis HLS Gym L0–L6 curriculum](rl_environment/GYM_CURRICULUM_L0_L6.md), from a tiny single MatMul with four hardware knobs to precision, continuous clocks and deeper networks. Run `python -m rl_environment.gym_curriculum --list-stages` or `--plan-all` with **zero vendor calls**; every optional native stage requires a new source-pinned output and explicit Vitis budget, 100 online training episodes ×10 calls and equal-budget trained/untrained/heuristic heldout controls. Stages are separate fresh GNN experiments, NOT pretrained progressive transfer in v1. New `clock_mismatch_probe` diagnoses the earlier **850 Vitis XML requested-target mismatches** without changing original case evidence; explicitly setting uncertainty is not enough to fix target mismatches. New per-stage read-only `gym_curriculum_report` builds true qualified HLS scorecards. Source implemented, full CPU tests/new Vitis stages still to verify on workstation; L0 must pass 22-case original-native smoke audit before costly 100×10. Previous original experiments unchanged.\n\n**Beginner FPGAI RL milestone (1 November 2026):** a deliberately easier first **real original-Vitis** learning task, opt-in `--beginner-pilot --freeze-numeric-format --fixed-clock-ablation-ns 32.000`: only one small 4–8-width MatMul, unseen heldout width combinations, fixed 16/6 numeric precision and 32ns target; search only parallelism/PEs/SIMD/reuse. The NEW beginner-only reward is dense and grounded in each strictly qualified original HLS worst-cycle report, not fabricated latency; the default/full research reward remains unchanged. A source-hash-attested executed Vitis Tcl `set_clock_uncertainty 8.640` may resolve **absent XML uncertainty only**, retaining fresh XML target/estimate and strict timing/CSim/resources gates. See [beginner 100×10 protocol](rl_environment/BEGINNER_MATMUL_100X10.md). Not yet workstation/native validated; 22-case smoke audit and >=14 strictly qualified cases are mandatory before any 1090-case campaign. Earlier 1000-training and 24ns smoke original studies stay immutable.\n\n**RL 100×10 architecture-only learning pilot (1 Nov 2026):** The user requested a simpler first experiment with a single conservative **32 ns HLS requested target** and fixed **16-bit/6-integer numeric format**. Source supports opt-in `--fixed-clock-ablation-ns 32.000 --freeze-numeric-format`: GNN explores nonnumeric hardware architecture, strict real Vitis qualifications/latency-only reward unchanged. See [fixed-clock locked-numeric experiment and stop gates](rl_environment/FIXED_CLOCK_NUMERIC_LOCKED_PILOT_100X10.md). **The clock is not proven physically safe; prior 1009-case data showed 850 XML timing parser invalidations.** Original 24ns smoke completed 22 vendor exits, but its independent original-revision replay audit is still missing; do not run new expensive experiment until new locked-numeric smoke and report evidence pass.\n\n**New Gym learning-remediation sprint (31 Oct 2026):** [RL learning redesign and reproducible 100×10 validation plan](rl_environment/SPRINT_RL_LEARNING_REDESIGN.md). The earlier 1,070-case original-Vitis pilot is independently audited, but only 71/1,000 training actions qualified; this does not demonstrate learned superiority. The new prospective protocol adds original-HLS qualification-stage diagnostics, a bounded historical-observation GNN, full compiler-legal initial candidate alternatives, auxiliary real-HLS feasibility classification, sampled frozen-policy matched comparison, independent reference-search random streams and censored heldout feasibility reporting. The latency-only scientific reward and source-pinning remain unchanged. **New revision not yet workstation/native validated.**

# FPGAI Research — Gymnasium Graph Reinforcement Learning for Vitis HLS

**Current active research track (31 October 2026).** The main branch studies **one shared GNN+DRL policy trained from scratch across changing neural networks**, using **real, original Vitis HLS CSim+CSynth feedback**. The goal is to minimize *HLS-timing-qualified estimated whole-network latency in nanoseconds* for resource- and numerical-feasible designs, not to minimize cycles alone, FPGA area, or power.

**Design specification:** [Research question](rl_environment/RESEARCH_QUESTION.md) is authoritative. The user's workstation passed **31/31 focused and 658/658 full-suite tests** at `0cc2a22` and the float-clock plan used zero vendor calls. A real read-only original Vitis HLS 2023.2 report then revealed its 7.664 ns estimate exceeds the **effective** 7.3 ns scheduling budget (10 ns target − 2.7 ns reported uncertainty). Strict HLS clock-margin and learned floating-clock CPU tests passed on workstation; the later 100-episode campaign controls, 30-invocation hard cap, progress and checkpoint audit changes are pending retest. There are no newly qualified original Vitis HLS cases. The previous cycles-only MatMul/Gemm environment remains engineering history, not evidence for the new hypotheses.

## Research at a glance

- **Episode:** one frozen Functional Model IR neural network, its structure and dimensions, a target FPGA with usable LUT/FF/DSP/BRAM/URAM limits, a maximum numerical error and finite HLS-call budget.
- **Observation:** full typed variable-length network graph (operation type, tensor dimensions, connections), FPGA resources and numerical-error requirement; dynamic HLS reports and remaining budget. No initial raw weight/activation statistics in GNN features.
- **Agent:** the **same learnable message-passing GNN and shared DRL policy across all episodes**, randomly initialized and trained online *in our Gymnasium environment*. Retain both per-node and graph-level embeddings.
- **Action:** a complete compiler-legal hardware design: global format and **agent-selected requested clock period**, plus layer-local parallelism, PEs, SIMD, tiling, reuse/serialization and II. The problem/task contains no default requested clock or list of clock values; only immutable engineering safety bounds on a policy-generated floating-clock action.
- **Evaluation:** exact FPGAI Model IR → original Vitis HLS CSim+CSynth, source/clock/part/testbench-attested audit. Zero vendor programs from reset/report/tests.
- **Objective:** minimize HLS worst-case cycles × **requested, HLS-estimated-timing-feasible** clock period in ns. Feasible only when CSim error, resource caps and reliable HLS clock feasibility pass. No reward for conserving resources; using all *usable* device resources is allowed.
- **Training:** scratch initialization, online feedback updates, checkpoints from that training, frozen unseen-network evaluation, separate adaptation experiment; random/local/evolutionary/Bayesian baselines.
- **Scope:** **no Vivado, post-route/physical Fmax, board latency, energy or surrogate performance rewards** in this Gym study.

## Active documentation

- [Research question and full prospective experiment specification](rl_environment/RESEARCH_QUESTION.md)
- [End-product acceptance contract](rl_environment/END_PRODUCT.md)
- [Engineering roadmap](rl_environment/ROADMAP.md)
- [Sprint and blocker tracker](rl_environment/SPRINTS.md)
- [Source-pinned next-chat handoff](rl_environment/HANDOFF.md)
- [Current Gym research package and native prototype](rl_environment/README.md)
- [Research changelog](rl_environment/CHANGELOG.md)
- [Paper-ready reward, original-HLS and policy-training evidence contract](rl_environment/PAPER_EVIDENCE_SPEC.md)
- [Selected 100-training-episode native experiment (max 30 HLS calls each)](rl_environment/EXPERIMENT_100_EPISODES.md)
- [Open-source release blockers](docs/OPEN_SOURCE_RELEASE_CHECKLIST.md)

## New Gym source implementation (floating-clock CPU gate passed; 100-episode campaign setup pending retest)

- `rl_environment/gym_env.py`: masked Gymnasium observations and full candidate+clock actions, currently bounded to the compiler-supported 1–4 MatMul/Gemm graph scope.
- `rl_environment/gym_agent.py`: shared learnable message-passing graph actor–critic, online updates and private checkpoints.
- `rl_environment/timing.py` and `reward.py`: strict original HLS clock feasibility and single ns objective.
- `rl_environment/gym_study.py`: dry-run plan by default; native work requires explicit authorized foreground budget.
- `rl_environment/gym_audit.py`: independent read-only case replay of original HLS, source/clock, reward, trial values and policy-update diagnostics; no vendor calls. `gym_report.py` generates benchmark reports only after replay passes.
- `rl_environment/research_metadata.py` and `paper_evidence.py`: source-pin exact reward/optimizer setup and export a complete native trial table, training episode reward/feasibility curves, per-update actor/critic/entropy/gradient diagnostics, native attempt inventory, source snapshots and optional figures. Incomplete exports are provisional, not audited.
- `rl_environment/baselines.py`: random, greedy, evolutionary and Parzen-reference controls, not a fully tuned Optuna TPE. Optional `--fixed-clock-ablation-ns 10` fixed-clock and `--no-message-passing` ablation modes are source-implemented.

Plan with zero native processes:

`python -m rl_environment.gym_study --train-episodes 100 --validation-episodes 1 --test-tasks 1 --max-vitis-compiles-per-episode 30 --estimated-seconds-per-compile 28.15`

The selected study is **100 training episodes plus seven validation/test/baseline episodes**, up to **3,210 Vitis HLS calls** (30 per episode), with **no 8-hour stop**. The forecast is informational only. Checkpoints every 10 completed training episodes and append-only progress were added; the original Vitis invocation path remains a separate, explicitly authorized foreground step.

CPU tests (**31 focused and 658 full passed at `0cc2a22`**; 100-episode campaign controls pending fresh test run):

`python -m pytest -q tests/test_gym_graph_hls_contract.py`

`python -m pytest -q`

## New Gym validation and experimental entrypoints

The CPU-only source passed 31 focused/658 full workstation tests at `0cc2a22` and real Vitis 2023.2 XML field layout and strict clock-margin probe are verified. **Later campaign-control tests still need a workstation run**, and neither an old report nor a read-only probe qualifies a new native Gym experiment.
Run this from a clean checkout of `main` with the intended Vitis-capable Python venv:

```bash
git pull --ff-only origin main
python -m pip install -e '.[test-rl]'
python -m pytest -q tests/test_gym_graph_hls_contract.py tests/test_online_post_hls_env.py
python -m pytest -q
python -m rl_environment.gym_study --train-networks 2 --validation-networks 1 \
  --test-networks 1 --calls-per-network 3
```

The normal study does **not** accept user clock periods or a clock grid. The shared policy uses a learned architecture-conditioned **Beta clock distribution** to output real-valued HLS period proposals within an internal **2–32 ns** safety interval; proposals are canonicalized to **0.001 ns precision** for reproducible Tcl/hash/audit. Clock targets are not a preselected discrete list. This is high-resolution floating search, not unbounded physical Fmax discovery. A feasible HLS clock boundary is unknown until real synthesis. The last command is a **plan-only dry run**, requiring zero Vitis tools.
Check a previously completed Vitis case first, **read-only**, without running synthesis or claiming qualification (the former parser failed on a real Vitis HLS 2023.2 report, so a strict alternate XML target-clock location and privacy-limited inspector were added; **not yet workstation-verified**):

```bash
python -m rl_environment.timing --case /real/path/to/HLS/case_0000 --inspect-schema
python -m rl_environment.timing --case /real/path/to/HLS/case_0000
```

The case must contain `manifest.json` and `fpgai_hls_project/solution1/syn/report/fpgai_model_csynth.xml`. This only checks timing schema and manifest clock agreement; it does not qualify the old case for the new paper. Only after the new preflight's tests pass and real local Vitis XML timing grammar is compatible may the owner explicitly run a bounded original native pilot:

```bash
python -m rl_environment.gym_study --train-networks 2 --validation-networks 1 \
  --test-networks 1 --calls-per-network 3 \
  --output build/gym_native_smoke_v1 --execute-native \
  --max-total-vitis-calls 27 --vitis-executable vitis_hls
python -m rl_environment.gym_audit --study build/gym_native_smoke_v1
python -m rl_environment.gym_report --study build/gym_native_smoke_v1 \
  --output build/gym_native_smoke_report_v1
```

Every native output location must be new and is immutable. A failed/ambiguous
Vitis timing report is **not** qualified. If tests or a native timing gate fail,
stop and fix the issue before scaling. For controls use the same call budgets
and seeds in separate new folders: `--fixed-clock-ablation-ns 10` for an intentionally fixed-clock and
`--no-message-passing` for graph-message-passing ablation. The Parzen search
method is a reference heuristic, not an independently tuned Bayesian/TPE
method. No automatic vendor CI or full scientific experiment has run.

## Existing runnable engineering prototype (NOT the new full research environment)

The original source-pinned cycles-only, fixed-clock prototype completed **16 original Vitis calls**: 16/16 independently source/numerically/resource-qualified, 2 online updates, 450.353 seconds total vendor process time, and one held-out tie: RL 43 vs random 43 HLS cycles. The user ran 2,156 full tests at the audit revision. **This does not establish trained GNN generalization, clock-period optimization, HLS timing feasibility or latency-ns improvements.** Original source commit: `d8bfdcc12a6e5bf9a1b6412c5cba6afbe87d2194`. Independent audit revision: `623a0dc30fd9b762524644f90fbc9988ed1d4a59`. Preserve original evidence privately.

The retained legacy `rl_environment/` modules and their three targeted test modules still support source-only/non-native verification; the retired `study.py` now rejects native campaigns. New experiments must use the Gym entrypoint and new audit. Future Gym development must keep original audit provenance. Do **not** execute an old 100-network **cycles-only** run and claim it validates the new research question.

### Source-only test entrypoint

```bash
PYTHONPATH="$PWD/src" python -m pytest -q tests/test_online_post_hls_env.py tests/test_online_post_hls_audit.py tests/test_online_post_hls_report.py
```

The user must verify the correct venv Python. Native Vitis execution requires a fresh, clean source-pinned checkout, an explicit bounded foreground opt-in and immutable case folders.

## Source layout

- `rl_environment/`: **only active RL research** (new Gym wrapper, shared trainable GNN/trainer, original-safe HLS adapter and separately labeled legacy pilot).
- `src/fpgai_research/`: reusable research compiler, operator semantics, architecture resolver, emitters, original vendor runners/report parsers and any transitive compatibility helpers. Do not confuse retained compiler support libraries with active RL experiments.
- `tests/`: retained current-environment and core compiler regression tests.
- Earlier search games, physical/Vivado RL and old manuscript/experiment entrypoints have been removed from this **branch**, not Git history. The old branches and exact historical commits remain the archive.

## Publication status

Repository remains **PRIVATE**, without an approved license. No release until institutional/IP authorization, dependency and full-history secret/weight checks and explicit owner approval. Do not share original Vitis case files or trained weights by default.

**Compiler IR scientific novelty (prospective):** [Compositional Space–Time Model IR](paper/compiler_ir/COMPOSITIONAL_SPACETIME_IR_NOVELTY.md) now specifies the candidate first-paper contribution beyond known multilevel IR or per-layer hardware knobs: an operator-local iteration-to-space/time mapping coupled to tensor-element production, availability, physical storage/layout and downstream consumption, with checked HLS lowering. This has **not been implemented or validated as a new graph-level mechanism yet**. Recent Allo, HIDA, StreamTensor and Stream-HLS prior art means mere type-safe composition or streaming layout is insufficient novelty. No RL source code is modified.


**Compiler source implementation update (2026-11-01):** Opt-in \`model_ir/composition.py\` couples operator-local Phi provenance to tensor exchange and actual alternate HLS matrix addressing; \`model_ir/row_lifetime.py\` and \`codegen/hls/row_interleaved.py\` implement a separate narrow 2×MatMul schedule in which each producer row is completed and consumed before reusing a scratch row. [CST novelty review](paper/compiler_ir/PRIOR_ART_NOVELTY_GATE.md) and [compiler-only test runbook](paper/compiler_ir/RUNBOOK.md). **All new tests and native measurements are pending; these prototypes are not yet an established novel IR or physical timing/BRAM result.** The default source-emission path and frozen RL agents/environments remain intact.


**CST compiler experiment driver (prospective):** \`compiler/cst_pipeline.py\` offers opt-in whole-tensor, column-major and row-lifetime Model IR compilation. \`validation/cpp_row_trace.py\` and new mutation tests independently check generated-host-C++ temporal/numeric behavior. \`experiments/cst_native_campaign.py\` prepares matched original-HLS source/testbench cases with **zero vendor calls** by default and requires explicit bounded \`--execute-native\` for Vitis. [Commands](paper/compiler_ir/RUNBOOK.md). All new scientific numerical/HLS results remain **unmeasured** until the tests/studies run; no RL code changed.


**Compiler research manuscript and readiness (2026-11-02):** [CST-Model IR full paper draft](paper/compiler_ir/MANUSCRIPT_DRAFT.md), [publication acceptance gates](paper/compiler_ir/PUBLICATION_READINESS.md), [evidence ledger](paper/compiler_ir/REPRODUCIBILITY_LEDGER.md). Source-pinned CI: **819 passed, 2 skipped** full CPU regression and **16/16** selected GCC source numerical checks over three CST experiment families. Native Vitis performance remains **unmeasured**; a separate manual self-hosted [original-HLS workflow](.github/workflows/compiler-ir-native-vitis.yml) is available, but is not executable without a user-configured Vitis-equipped runner. 2026 ST-Flow related work prevents any generic first-of-kind spatial/temporal or reuse claim. Frozen RL sources remain unchanged.


**New CST nonlinear research checkpoint (2026-11-02):** the opt-in compiler can now preserve and lower sequential **Flatten/Gemm(bias + transB)/Relu/Gemm** graphs to real B-row source schedules. At commit `b563ba9c`, [full CI](https://github.com/umutcanaltin/fpgai_research/actions/runs/41369433057) passed **893 tests, five skipped**; [research CI](https://github.com/umutcanaltin/fpgai_research/actions/runs/41369433106) passed **128 focused tests** and a new 16-case synthetic nonlinear source study (6/6 selected host numeric, 3/3 row traces); [ONNX CI](https://github.com/umutcanaltin/fpgai_research/actions/runs/41369394773) independently passed three synthetic ONNX Runtime/NumPy/GCC cases. Actual trained-model source-provenance and Vitis preparation commands are in [runbook](paper/compiler_ir/RUNBOOK.md); original trained assets, native vendor execution and published-compiler comparator experiments remain outstanding. RL sources preserved at frozen backup.


## One-terminal compiler-paper experiment series (2026-11-02)

The [compiler-IR automatic research suite](experiments/compiler_ir/README.md) now runs the source-pinned E0/E1 experiments, independent oracle, mutation controls, scalability, nonlinear CST, matched vendor-source preparation, and optional actual-trained-asset/explicit original-Vitis phases in **one terminal command** from a separate clean checkout:

```bash
bash experiments/compiler_ir/run_all.sh --bootstrap --profile paper
```

Every run creates a new `build/compiler_ir/series/<UTC>_<SHA>/` with a manifest, consolidated summary, raw experiments and checksummed stage logs. Native HLS is **never** invoked without `--with-native --confirm-isolated-native`; trained MNIST sources require their actual checksum-verified private asset folder. Original RL/Gym code and running L0 experiments are unaffected. External ST-Flow/Allo/StreamTensor comparisons, Conv/branching CST generality and physical FPGA claims remain distinct unresolved scientific gates.

## Research compiler: policy-independent, learning-ready decision IR

New `src/fpgai_research/model_ir/decision_interface.py` exposes immutable complete architecture states, typed edits, actual-compiler-revalidated acceptance/rejection, finite legal-action masks and weight-private variable-size observations. It works without RL/Gym and does not change the original frozen RL environment. A new create-only `cst_decision_campaign` runs restricted independent action controls plus random/heuristic proposals through the *same* interface, included in the [one-command compiler experiment series](experiments/compiler_ir/README.md). Scientific semantics and the L0–L9 sprint gates: [CERTIFIED_DECISION_IR_SPRINTS.md](paper/compiler_ir/CERTIFIED_DECISION_IR_SPRINTS.md). The v1 row-atomic validator is **not yet** Phi-conditioned per-element, multi-reader or physical-HLS proof.
