## 2026-11-05 — HW-RL-05 multi-network auditable algorithms

- Added one shared GNN policy interface for genuine one-step actor-critic,
  original-HLS-audited episodic REINFORCE with value baseline and
  clipped PPO with multiple real on-policy optimizer epochs, plus
  independent-RNG random baseline. No fake vendor feedback.
- Added create-only `hardware_campaign` experiment driver with explicit
  complete budget, disjoint model-fingerprint train/val/test manifest,
  per-episode original-Vitis execution, per-trial and per-episode raw
  journals, evaluation-only tensor checkpoints, read-only source/task/
  report/reward replay and no physical-Fmax claims.
- Added focused no-Vitis unit tests for budgets, disjoint network
  fingerprints, no-pretrained-eval gate, algorithm audit guard and
  synthetic-only campaign plumbing.
- Registered `fpgai-rl-hardware-campaign` installed CLI. Default is
  source-only; native requires a fresh `--output-root`,
  `--execute-native` and exact `--max-total-vitis-calls`.
- Documented a falsifiable, prospective second-paper protocol with
  prior-work differentiation and explicit missing gates. **All new
  HW-RL-05 code and tests still need workstation verification.**
  No real multi-network learning results are claimed.

## 2026-11-05 — HW-RL-04 native smoke and diagnostic visibility

- User's first real shared-GNN/Vitis one-step smoke returned one audited
  native call and gradient update, but the sampled 6.55 ns / operator-5
  tile (1,4,27) design was infeasible and earned -0.05. Exact original
  rejection gates must be read from the immutable local episode record.
- Forward source-attested gate failures, CSim/numeric values, raw cycles,
  HLS timing, resource consumption and caps in future agent CLI outputs.
  Qualified latency stays null for rejected cases; no fake Fmax or
  secondary native execution is introduced.
- Report `private_weights_saved` accurately. The smoke did not specify
  a checkpoint, so its one gradient update did not persist.
- Added CPU regression for passthrough of an infeasible source audit.
  The new GitHub patch has not yet been workstation-tested.

## 2026-11-05 — HW-RL-04 optional PyTorch installation diagnosis

- Workstation confirmed 2,033 passing / 18 skipped full CPU tests, but
  the GNN CLI stopped because `pip install --no-deps -e .` had not
  installed the `rl` extra's PyTorch requirement.
- Fail before model loading or native execution with the precise
  `python -m pip install -e '.[rl,onnx]'` remediation and preserve
  the import failure when a torch wheel is broken.
- Added dependency-path regressions that do not skip when PyTorch is
  absent, and documented workstation install and source-only preview.
  No vendor qualification or GNN learning is claimed from this patch.

## 2026-11-05 — HW-RL-04 shared GNN bridge for full compiler hardware Gym

- User verified HW-RL-03: 75 focused passing, one skipped; full CPU
  2,032 passing, 17 skipped; source preview 160 legal actions.
  Original Vitis 15 ns clock-only Gym step qualified at 646 worst cycles,
  9,690 ns HLS latency, +0.105 ns scheduling margin and −0.005 reward.
  Native case and prior 14 ns/10 ns observations remain immutable.
- Added exact operator-target node indices plus a 30-step, masked
  genuinely observed native-feedback history to `hardware_gym`.
- Added one shared graph policy using existing `gym_agent.MessagePass`
  with context-conditioned discrete compiler action mask, continuous
  Beta clock, trial-history attention, critic and feasibility auxiliary.
- Added policy update gate: original matching live Gym state/action/reward
  plus a fresh read-only audit of the original case. Missing or
  forged vendor evidence cannot earn updates via an ordinary dict.
- Added `fpgai-rl-hardware-agent` read-only untrained action preview
  and opt-in **single** original-Vitis source-audited gradient step.
  No large native campaign or pretrained checkpoint is created.
- Added CPU source/mask/gradient/metadata/checkpoint and installed
  entrypoint tests. **New HW-RL-04 code and tests not yet executed on
  the user workstation**; no trained transfer or speedup is claimed.

## 2026-11-04 — HW-RL-03 actual native-qualification Gym interface

- User confirmed 14.00 ns mixed CNN original Vitis HLS case qualified:
  CSim passed, 646 worst cycles, 9,044 ns estimated HLS latency, 10.213 ns
  estimated target path vs 10.220 ns effective budget, 0.007 ns margin.
  Independent audit reproduced E3 and provenance without a new Vitis call.
  The earlier 10 ns case remains unqualified; nothing was overwritten.
- Introduced a separate `GymHardwareNativeEnv` with bounded
  tensor/operator graph nodes and edges, explicit source-mask and
  action-field features, and a learned floating clock output quantized
  only for original-Vitis report compatibility (0.01 ns).
- An authorized native Gym step invokes the existing one-shot
  `execute_native_case`, independently re-audits its original
  source/numeric/resource/timing receipts and uses the established
  latency-only reward. All original vendor source cases remain separate
  from episode-level step records.
- Added a read-only preview and one-step opt-in
  `fpgai-rl-hardware-gym` CLI and CPU-only mock-lifecycle tests.
  **The new HW-RL-03 code and first native Gym step have not yet
  been tested on the user's workstation.**
- No existing historical Gym/agent/reward/curriculum code, old Vitis
  cases or GitHub Actions changed; no trained policy or physical Fmax
  claim is made.

## 2026-11-04 — fix strict native receipt comparison after JSON round-trip

- User ran HW-RL-02 locally: the focused suite reported **3 failed, 54 passed, 1 skipped**; the full suite **3 failed, 2,000 passed, 22 skipped**. All failures were the same strict action receipt comparison, before any original Vitis qualification.
- Root cause: `dataclasses.asdict(ArchitectureRequest)` preserves `tile` as a Python tuple, while the saved `hardware_action.json` represents it as a JSON array. Comparing JSON-loaded data directly to an unnormalized Python plan rejected legitimate immutable source cases.
- `planned_case` now canonicalizes its exact full content to JSON-native values before either saving or auditing. The audit still compares the **complete** action record and separately verifies source, testbench, part, clock, numerical tolerance, CSim, E3 and native evidence.
- Added regressions for baseline and non-empty Conv tiling round-trips, plus tampering with the nested architecture tile value. No vendor execution or old case mutation.
- This revision is committed separately on the RL feature branch and **requires new workstation pytest verification** before any original Vitis case.

## 2026-11-04 — HW-RL-02 original native HLS evaluation adapter

- The user's installed `fpgai-rl-hardware-space --help` succeeded from
  `/tmp`, and source-only CLI returned a real typed graph for eight
  ONNX operators with two Conv nodes, MatMul, Gemm, and supported scalars.
- Added a separate `compile_state` reuse of the original compiler's
  resolver, Model IR refinement and HLS emitter; source-only receipts
  and historical RL APIs preserve semantics.
- Added `rl_environment.hardware_native` with bounded source-only
  preparation, full-state and numerical-cohort source digests,
  explicit one-case original Vitis HLS execution, immutable case
  folders and independently replayed E2/CSim and E3/CSynth qualification.
- Explicit uncertainty and a 0.01-ns native period requirement protect
  existing Vitis 2023.2 XML target attribution; source clock remains
  continuous to 0.001 ns.
- Unknown/missing measurements are never imputed. No new rewards,
  training policy, native physical timing or board evidence claimed.
- Added source-only fail-closed tests and installed CLI subprocess smoke;
  new tests and first licensed Vitis run must execute on the user's PC.

## 2026-11-04 — repair installed RL CLI package discovery

- The user verified HW-RL-01 source functionality: **30 focused tests passed**
  and **1,982 full passed / 21 skipped** on the local workstation.
- Their installed `fpgai-rl-hardware-space` raised
  `ModuleNotFoundError: No module named 'rl_environment'` because
  setuptools only discovered packages under `src/`. Fixed by explicitly
  mapping both package roots and finding `fpgai_research*` and
  `rl_environment*` with proper subpackages.
- Added regression tests for setuptools package discovery and real
  console/Python imports **from outside the checkout with no
  PYTHONPATH**. No source copy, script-level sys.path hack, vendor calls,
  changes to native RL tasks, agent weights, rewards or experiments.
- The new package correction and tests are pending reinstall and
  workstation validation.

## 2026-11-04 — hardware knob source interface on separate RL feature branch

- Explicitly reopened RL interface development after the earlier
  compiler-first freeze, without restarting any frozen original native study.
- Added `rl_environment.hardware_space`: all eight *currently implemented*
  `ArchitectureRequest` categories, globally synchronized numeric format,
  per-operator Conv/MatMul/Gemm mapping and tiling, continuous 2–32 ns
  requested clock, immutable full candidate SHA and original compiler E0/E1
  source checks. The policy cannot select fake memory/AXI/CDC/physical knobs.
- Added deterministic bounded candidate menu and source action mask,
  variable-size weight-private graph observation, and an explicit source-only
  planning environment. No HLS latency, physical feasibility or training
  reward is invented; Gym `step` stays disabled pending audited real Vitis.
- Added source-only unit/integration tests and `fpgai-rl-hardware-space`
  read-only ONNX CLI. **Workstation tests and native HW-RL-02 are pending**.

## 2026-11-01 — native L0 imported compiler source-binding guard

- Diagnosed discrepancy: CPU test suite 734 passed, but first `gym_curriculum --stage L0 --smoke` stopped before native Vitis with `write_vitis_package() unexpected keyword clock_uncertainty_ns` even though checked-out `src/fpgai_research/toolchains/vitis_hls/package.py` already supports that parameter. Likely non-editable/stale installed package shadowing source during `python -m`; CPU pytest inserts repo src into its own imports.
- Added `rl_environment.runtime_contract` to fail **before create-only HLS study allocation** if imported compiler module path, writer/Tcl signatures or backend binding do not match the current checkout. Tests now call real package writer with explicit uncertainty and reject relocated/old source bindings without vendor calls.
- Documented `python -m pip install --no-deps -e .`, fail-fast origin verification, and new `v3` L0 smoke after CPU tests; preserve original partial `v2`. No native validation has been executed by assistant.

## 2026-11-01 — structured L0–L6 from-scratch original-HLS Gym experiments

- Fixed a collection-blocking duplicate comma SyntaxError in `gym_study.py` CLI.
- Added frozen source-defined `curriculum.py` stage contracts; `gym_curriculum.py --list-stages/--plan-all/--stage Lx` uses source-only stage plans, create-only native studies and no silent vendor execution. L0 small 1-op MatMul/four knobs, L1 variable-width 1-op, L2 2-op, L3 tile+II, L4 numeric format, L5 continuous clock, L6 harder 3/4-op topology heldout. New task fingerprints, Functional Model IR cohorts, native offer checks, exact trained-from-scratch GNN per stage.
- Added curriculum-only real HLS-nanosecond normalized dense feedback, using original worst-case top cycles, requested period and typed Model IR MACs; every feasible case helps train. Strict source/accuracy/resources/timing gate intact, historical Gym reward unchanged. Independent `gym_audit` recomputes stage reward, model and action knob contract.
- Added read-only `clock_mismatch_probe.py` because user logs showed 850/1009 earlier native rejections from `different_reported_target`, not missing clock uncertainty. Exact original clock differences must be inspected before any report/parser policy modification.
- Added `gym_curriculum_report.py` original-HLS audited same-task trained vs untrained comparisons and per-stage feasibility without misleading cross-task absolute latency averages. Added source fixture regression tests and protocol in [GYM_CURRICULUM_L0_L6.md](GYM_CURRICULUM_L0_L6.md). **No new local CPU test execution or original Vitis calls by assistant; user workstation verification and staged smoke required.**

## 2026-11-01 — true beginner original HLS learning configuration

- Added opt-in `--beginner-pilot --freeze-numeric-format --fixed-clock-ablation-ns 32.000` through Task SHA, Functional Model IR cohort, Gym candidate authorization, study and independent audit. Training contains one MatMul per model, small 4/8 widths, distinct same-depth heldout widths; only parallelism, PEs, SIMD, serialization vary. Does not alter full 5-layer product numeric/device/clock search.
- Added original Vitis 2023.2 Tcl explicit `set_clock_uncertainty 8.640` for new beginner cases. If XML lacks uncertainty, original strict HLS parser can use ONLY verified executed Tcl (matching immutable manifest, native script digest, before/after tool hashes) alongside fresh original XML target and estimate. Conflicts and timing failures remain fail-closed. Zero need to modify original 1009 prior cases or prior incomplete studies.
- Added separately documented beginner-only dense from-scratch reward from every verified real HLS cycle observation `mac_count/(mac_count+worst_cycles)` and existing -0.05 for rejected. Historical best-only reward untouched, science metadata and independent auditor record the distinction.
- Added source-level tests and pre-registered 22-case native gate and 100×10 experiment under [BEGINNER_MATMUL_100X10.md](BEGINNER_MATMUL_100X10.md). **Workstation CPU tests and new native smoke have not been run or claimed as passed.**

## 2026-11-01 — 1,009-call incomplete Gym campaign recovery

- Actual first redesigned native experiment completed 1000 training calls, then 9 frozen validation calls before repeated deterministic architecture-clock proposal caused Python `ValueError`. No full `summary.json`/full original campaign audit exists; original case folders and checkpoint remain immutable.
- Fixed **pre-native** duplicate-action rejection: deterministic frozen/reference methods choose nearest unused 1-ps clock for the same architecture, stochastic policies resample with the final action's gradient tape. No fabricated QoR or re-run of previously executed case.
- Added independent `gym_partial_audit.py` (replays original source-pinned 1009 cases; explicitly marks full campaign incomplete), and `gym_eval_recovery.py` (verifies original trained checkpoint, launches **new 90-case** heldout evaluation on a distinct revision, independently audits, exports method table). The new study MUST disclose both source SHAs and cannot be presented as one uninterrupted campaign.
- Read [INTERRUPTED_1000_CALL_RECOVERY.md](INTERRUPTED_1000_CALL_RECOVERY.md); source committed but user's CPU/native tests of new recovery are pending. Never overwrite original training evidence or rerun 1000 calls unnecessarily.

# Current Gym Graph RL research changelog

## 2026-10-31 — RL learning remediation after independently audited 1,070-case pilot

- Verified historical original source `eb4a799` remains unchanged; pilot had only 71/1000 training feasible results and 5 positive rewards. The first action was forced to baseline; frozen/adaptive and four reference baselines were not independently comparable.
- New source-only Vitis gate diagnosis exposes source/native, CSim, numerical, cycles, resource and HLS timing failure categories, and read-only `qualification_funnel.py` for old native cases without invoking tools or modifying source studies. No relaxation of accuracy, clock or resource qualification.
- Added full bounded trial-history tensor and attention to shared GNN; legal candidate menu mixes global proposals and local refinements and does not force baseline on first HLS call.
- Added original Vitis labelled feasibility auxiliary cross-entropy loss and feasibility-guided categorical action prior, while keeping the original latency-only reward exactly unchanged, 2–32 ns floating requested-clock action, and one shared from-scratch policy.
- Added matched sampled frozen arm and independent RNG/feedback-based fallback reference heuristics. Updated audit, results and paper exports with real gate and training diagnostics. Pilot 100 train ×10 +1 validation ×10 +8 test arms ×10 =109 total episodes, maximum 1090 original Vitis calls.
- New contract and commands: [SPRINT_RL_LEARNING_REDESIGN.md](SPRINT_RL_LEARNING_REDESIGN.md). **All new revisions pending workstation CPU and original native smoke testing; no improved learning claim.**


## 2026-10-31 — Restart authorized to capture reproducible paper learning curves

- User explicitly agreed to stop the already started original Vitis study and restart; **preserve prior output directory immutable**. Do not mutate old source-pinned cases or claim newly recorded policy metrics exist in old results.
- Introduced `research_metadata.py` to record exact source commit, SHA256 of active reward/agent/clock/HLS/audit modules, numeric reward definition (-0.05 infeasible, -0.005 first feasible/evaluation penalty, log improvement), actor-critic hyperparameters and original HLS-only objective in `campaign.json` before executing vendor tools.
- Per-step `native_steps.jsonl` now includes complete architecture and clock, CSim error/tolerance, original reported LUT/FF/DSP/BRAM/URAM and limits, effective HLS clock/timing margin, native report hashes, raw and cumulative episode rewards, and actual actor/critic/entropy/value/advantage/gradient diagnostics when training. Frozen and reference-search arms contain null training diagnostics. **No reward or policy optimization math changed**, only optional metric returns and output logging.
- Fixed previously missed frozen-test/adaptive-test completed-episode journal rows. Source-pinned `gym_audit.py` now validates paper metadata, original trial summaries, cumulative per-episode rewards, policy-update loss components and training/adaptation update sequence.
- New create-only `paper_evidence.py`: read-only original campaign, original-commit reward/agent/study/clock source snapshots, per-trial CSV, per-episode/reward/loss CSV and optional exploratory figures; separates unlogged/failed native cases, labels incomplete runs **provisional**, and requires independent replay for completed reports. New CPU-only negative/contract tests added; **not yet run on workstation**. See [PAPER_EVIDENCE_SPEC.md](PAPER_EVIDENCE_SPEC.md).


## 2026-10-31 — Select 100-training-episode campaign; hard cap 30 native HLS calls per episode

- User workstation verified **31/31 focused** and **658/658 full** CPU tests at `0cc2a22`; floating-clock dry-run showed zero Vitis calls. No new source-pinned native Gym HLS case has been run.
- User chose **100 training episodes** (instead of 3,000 or 200); preserve 1 frozen validation and 6 frozen/adaptive/baseline test episodes, totaling **107 episodes**, maximum **3,210 vendor calls**. Limit per new Gym episode is **30 original Vitis HLS attempts**, counting failures. Native run requires explicit `--max-total-vitis-calls 3210` authorization. This cap does not impose a reward penalty for resource usage.
- No 8-hour or other **campaign wall-clock time limit**. Added optional `--estimated-seconds-per-compile` solely for a maximum-call runtime forecast. Prior fixed-clock pilot's ~28.15 seconds/call would imply ~25.1 hours for 3,210 calls, NOT a validated estimate of the new float-clock HLS campaign.
- Added create-only private checkpoints every 10 completed training episodes and durable append-only `completed_episodes.jsonl`, with independent final-audit episode accounting; no automatic campaign resume. Detailed zero-vendor preflight and explicit native launch commands in [EXPERIMENT_100_EPISODES.md](EXPERIMENT_100_EPISODES.md).
- **Latest source edits need workstation CPU rerun and a small independently audited original Vitis smoke before attempting the full campaign.** Historical cases unchanged.

## 2026-10-31 — Correct masked Gymnasium Tuple sampling (last remaining CPU test failure)

- User's workstation at `b59af61`: **30/31 focused tests passed, 657/658 full tests passed**, and the zero-vendor-call floating-clock study plan succeeded. All Python syntax checks passed.
- The sole failure was in `test_automatic_clock_action_choices_really_are_agent_outputs`: it called unmasked `env.action_space.sample()`, which legitimately returned a padded architecture slot outside `env.offers`. In the Gym contract, `action_space` is a fixed-size `Tuple(Discrete(max_actions), Box(clock_low, clock_high))`; `observation["action_mask"]` specifies which compiler-legal architecture slots may be used. The environment correctly raised `ValueError` for a masked slot; no vendor tools were invoked.
- Regressed the correct standard Gymnasium API: `env.action_space.sample(mask=(obs["action_mask"].astype(np.int8), None))` samples **only valid architecture slots**, while leaving the floating clock unmasked. Exercised this 20 times without a Vitis call and added explicit rejection of padded slots.
- Only the test changed; native receipts, compiler sources, learned clock policy, 0.001 ns canonicalization, timing and audit gates are unchanged. This latest change still needs a fresh workstation run, and no new source-pinned native Gym case is yet qualified.

## 2026-10-31 — Fix uploaded 13-failure float-clock regression log

- Workstation at `6a73594`: **31 focused executed, 18 passed / 13 failed**; full suite **645 passed / 13 failed**. The 12 cascading failures originated from malformed indentation at `rl_environment/gym_env.py:152`; the remaining failure was an outdated case-sensitive `legacy clock` test assertion against the actual `Legacy clock` error. The dry-run CLI failed because that same import syntax error prevented module loading. No Vitis tool was invoked.
- Corrected the Gym action-mask indentation and the assertion's case; also corrected two test assertions that erroneously assumed exactly one compiler-legal architecture offer whenever the clock is continuous. The continuous floating clock and 0.001 ns precision contracts remain unchanged.
- Added a GitHub Actions CPU-only regression workflow that performs `compileall` before imports, installs Python RL/test dependencies, runs focused and full pytest suites, and checks zero-native-call dry-run behavior. Its first run was marked **failure before any steps or runner were allocated** (runner_id 0; no logs); treat this as an **automation infrastructure blocker, not a test result**. Workstation rerun is required.
- **Current source fixes remain unverified by an executed Python test suite**. Native Gym Vitis CSim/CSynth and source-pinned independent replay remain unexecuted.

## 2026-10-31 — Trainable floating-point clock actor replaces internal clock list

- User correctly rejected short integer/half-nanosecond clock lists. Added `clock_actions.py` with immutable **2–32 ns** internal engineering safety bounds and **0.001 ns** reproducible clock resolution. The normal task accepts **no requested clock or clock list**; the policy generates float targets from a learned architecture-conditioned **Beta distribution**. Values such as `3.141 ns`, `7.125 ns` are valid original Vitis Tcl actions and independently attributable from their hashes.
- Gym now accepts a joint `(architecture_slot, floating_clock_ns)` action and exposes a real-valued clock action space; `PostHLSEnv` checks the full pair for novelty and strict clock canonicalization. Fixed-clock ablations remain explicit and their already-tested candidate pairs are masked. Source-bound receipts and `gym_audit` validate the exact proposed float; baselines search the same numeric envelope and use only actual HLS response labels.
- Added/updated focused regression tests for learned float clock sampling and gradient updates, strict 1 ps period validation, exact Tcl emission, no vendor calls during dry-run, and explicit fixed-clock arms. **Latest refactor is NOT YET workstation-tested; no original new native Vitis run has occurred.**

## 2026-10-31 — Stop asking the experimenter for a clock grid

- User's workstation subsequently verified **24/24 focused and 651/651 full tests** at `dde0ba6`; real Vitis HLS 2023.2 XML timing now correctly fails the effective 7.3 ns scheduling budget when estimated period is 7.664 ns. The new research dry-run indicated eight maximum vendor calls, all **unexecuted**.
- Removed normal `--clock-action-domain-ns`/`--clocks-ns` input from the Gym study CLI. Standard Gym tasks and study plans now generate an internal, bounded **discrete** clock action domain with no user clock value or list. The policy still outputs its chosen requested period alongside the complete architecture. The only CLI clock argument retained is intentionally named `--fixed-clock-ablation-ns` for fixed-clock control experiments.
- Updated original-case audit to require clock-action domain provenance and agreement with the source-defined envelope. Added three CPU regression tests for clock-free task generation, a no-clock CLI dry-run and actual agent-selectable clock offers. **New source changes are not yet workstation-verified**. The current clock actor still selects from finite internal candidates; continuous clock generation is not yet claimed.

## 2026-10-31 — Verify Vitis 2023.2 original XML, correct effective clock-margin gate

- User workstation confirmed **21/21 focused** and **648/648 full** CPU tests at `01c7f87` (2.03s and 11.57s). Actual original Vitis 2023.2 CSynth XML report was parsed read-only: target **10 ns**, estimate **7.664 ns**, uncertainty **2.7 ns**, source report digest `9602767353935998154709342854cfe02a919c3634775d5dd95b350e15e802a2`; zero vendor processes were launched and no old results modified.
- **Corrected previous timing-feasible interpretation:** AMD UG1399 Vitis 2023.2 subtracts the reported clock uncertainty from the target to set its effective HLS synthesis scheduling budget. This report has a **7.3 ns effective budget** and **−0.364 ns scheduling margin**: it is **not HLS-timing qualified**, despite estimated 7.664 ns being below raw target 10 ns. A requested 10 ns remains the period used to compute latency if a candidate otherwise qualifies.
- `timing.py` now requires explicit original uncertainty and enforces `EstimatedClockPeriod ≤ TargetClockPeriod − ClockUncertainty`, returns effective budget and signed margin. Audited backend receipts, native study records and read-only Gym replay audit all propagate/check these values; three new CPU fixture tests cover the exact observed values and missing uncertainty.
- **Current patch has not been executed on the workstation.** The old read-only case is not qualified for the new Gym science; no original new native Gym CSim/CSynth or physical evidence has been produced.

## 2026-10-31 — Correct Vitis HLS clock XML layout and preserve strict qualification

- A real original Vitis HLS 2023.2 case was probed from the private workstation. Unlike the previous placeholder attempt, it found the manifest and reached the real CSynth XML, but the existing parser failed closed with `Missing or malformed HLS target/estimated period`. There is still **no timing-qualified new native Gym result**.
- Found a known CSynth XML grammar in hls4ml: `UserAssignments/TargetClockPeriod` for the requested clock, and `PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod` for the tool estimate. Updated the gate to accept this precise target-clock location or the earlier timing-summary location, rejecting duplicate, malformed, contradictory or manifest-mismatched evidence.
- Added a **privacy-limited, read-only** `python -m rl_environment.timing --case <existing-case> --inspect-schema` diagnostic reporting only the recognized clock field locations/values and report SHA. It invokes zero vendor tools, never qualifies old cases for the new study, and does not reveal arbitrary XML/private weight content.
- Added **four CPU fixture tests** for Vitis-style XML, conflicts, read-only existing-case probing and diagnostic privacy. These latest code/tests were **not executed on the user's workstation yet**; verify tests and the original report before authorizing any fresh Vitis HLS campaign.

## 2026-10-31 — CPU gate 644/644 and Vitis 2023.2 identified

- At `a0f5220`, user workstation confirmed **17/17 focused tests and 644/644 full tests passing** (2.02s and 11.62s). This establishes the current CPU-only gate, not original new native timing qualification.
- Installed vendor tool identified as **Vitis HLS 2023.2**. Previous original CSynth XML files are present in local private `build/` folders, but the first read-only preflight was given the literal instructional placeholder `/absolute/path/to/existing/case_0000`, so it predictably failed before reading any report. No XML schema compatibility is claimed yet.
- Next action: use an actual case folder with `manifest.json` plus native `fpgai_model_csynth.xml`; run `python -m rl_environment.timing --case <actual-case-directory>`, preserve old original evidence and do not start a new scientific campaign until report compatibility is established.

## 2026-10-31 — Green CPU suite and read-only original HLS XML timing preflight

- User confirmed **15/15 focused tests and 642/642 full tests passed** on the research workstation at `f83e5ab` (1.95s, 11.60s); this resolves the cleanup/test assertion regression at that source revision. No original new Vitis work has run.
- Added `python -m rl_environment.timing --case <existing-original-HLS-case-directory>`: checks actual existing Vitis CSynth XML timing fields against the case manifest clock without invoking Vitis, writing to the case or claiming full source/numeric/resource qualification. Reports `READ_ONLY_XML_SCHEMA_COMPATIBLE_NOT_NATIVE_QUALIFICATION` if grammar checks pass.
- Added two CPU fixture tests covering read-only state, expected timing feasibility and strict malformed/missing/mismatched report rejection; **these latest additions have not yet been workstation-tested**. Full independently source-pinned native timing still requires original Vitis CSim+CSynth and separate replay audit.

## 2026-10-31 — Workstation 641/642 regression and minimal test fix

- After restoring the Phi-oracle, full pytest successfully collected/executed 642 tests: **641 passed, one failed**, and focused Gym tests were **14 passed, one failed**. Both failures were the same assertion in `test_tasks_keep_legal_clock_grid_immutable`, expecting outdated text `clock grid` instead of the new `legacy clock` explanation.
- Changed only the test's expected error-message regex to `legacy clock`, retaining the actual validation: only legacy fixed-clock tasks must contain their legacy period in any supplied domain. Gym RL tasks remain clock-free, with target clock selected exclusively by the policy action.
- Source fix is committed, but the corrected full suite has **not been rerun on the workstation**. No original Vitis calls were launched or implied by this change.

## 2026-10-31 — Initial workstation tests, Phi-oracle restoration, RL-selected clock correction

- User's first focused new Gym test run passed **12/12** (3.09s). Full test collection failed due to a missing `test_hls_loop_phi_oracle.py` used by surviving C++ Phi trace tests; the exact original file was restored from Git history and an import-integrity regression added. The full suite must be rerun.
- Clarified the clock is **solely an RL output**, not a desired/known input: new Gym tasks set `clock_ns=None` and policy observations omit any default requested clock. The only clock input is a bounded *action search domain*, not a timing claim; a new native synthesis evaluates each architecture-clock action.
- Removed the fixed 10ns membership rule from the clock-action search domain. Default multi-clock exploration is now 3, 4, 5, 6, 8, 10, 12 and 16ns; supported choices must be tested against real HLS report timing before any claim of feasible Fmax. Unit tests added but latest changes **not executed**.
- These are source/repository changes only; no Vitis experiment has run under the new reward and no physical Fmax is claimed.

## 2026-10-31 — Gym source implementation on main (NOT yet verified)

- Merged the canonical Gym cleanup/specification into main (PR #4). Updated engineering and handoff instructions for direct main development; historical source-pinned cases remain immutable.
- Added `gym_env.py`: masked graph/architecture observations, Gymnasium reset/step, per-candidate global requested clock, and explicit foreground-only original Vitis calls.
- Added `gym_agent.py`: shared differentiable two-layer directed message-passing GNN, graph/node embeddings, policy/value heads, from-scratch immediate online updates, create-only private checkpoint with optimizer.
- Extended `tasks.py` and `post_hls_env.py` for per-episode legal clock grid, clock-bound original source/receipt, strict HLS-timing-qualified latency in ns and no area/power reward.
- Added `timing.py` for fail-closed, source-attested Vitis CSynth estimated period / target clock checks, and `reward.py` for independent single-objective calculation.
- Added `gym_study.py`: default dry-run, clean SHA, exact full native-call cap, 1–4 MatMul/Gemm, frozen validation, frozen/random/greedy/evolutionary/Parzen-reference/online-adaptive heldout treatment, fixed-clock and no-message-passing ablation modes. The Parzen reference is **not** yet an independently tuned Bayesian/TPE baseline.
- Added `gym_audit.py` (read-only independent regeneration/replay) and `gym_report.py` (paired comparisons, uncertainty and estimated-ns vs original HLS-call plots); plus focused CPU-only fixture tests.
- Retired `study.py` from native execution (dry-run only) to avoid mixing old cycles-only rewards with current objectives.
- **Not yet verified:** no workstation pytest, no fresh Vitis report schema confirmation, no native launch and no independent native replay. These are hard gates before starting scientific experiments or asserting speedup.
- New source is implementation work, **not** a scientific result or proven full end product.

## 2026-10-31 — New active research direction

- Established `rl_environment/RESEARCH_QUESTION.md` as canonical prospective research contract.
- Corrected the training design: **one shared GNN and DRL policy train from scratch ONLINE in our Gym environment**, no pretrained models.
- Locked network graph inputs: layer identity, dimensions, kernel properties and graph connectivity; plus usable five-resource FPGA capacities and numerical error limit. Fixed-dimensional global embedding from one shared, trainable GNN.
- Locked complete hardware architecture + global requested HLS clock action.
- Locked **single objective**: lowest *HLS-timing-qualified estimated latency in ns*; resources/numerical error/timing are constraints, not resource-saving rewards; no Vivado.
- Simplified documentation, active sprint tracker, engineering rules and handoff. Old experiments are archived in Git history, not active Gym experiments.
- The new Gym/GNN/clock/timing/latency features are **NOT YET IMPLEMENTED or workstation-verified**. Do not misrepresent old cycles-only experiments as new results.

## Historical native evidence retained in Git history (old contract)

- Pinned old study `d8bfdcc12a6e5bf9a1b6412c5cba6afbe87d2194`; independent audit at `623a0dc30fd9b762524644f90fbc9988ed1d4a59`.
- 16 original Vitis calls; 16/16 fixed-clock cycles/resource/numerical-qualified; 2 online updates; 450.353 vendor seconds; heldout 43 versus 43 HLS cycles.
- Prior audit run passed 2,156 original repository tests. No scientific GNN/clock/latency-ns generalization claim.

## 2026-10-31 — Branch cleanup and current rules

- Removed 410 old experiment/paper/reports/docs/scripts/obsolete tests and 12 old RL implementation modules in a dedicated new Gym branch, retaining 46 selected current/core tests. Nothing in main or the historical commits was deleted.
- Canonical engineering rules and research questions specify **training from scratch online inside our own Gym** (not pretraining).
- Remaining compatibility libraries under `src/fpgai_research/experiments/` must be dependency-audited before any further pruning. No new-branch workstation test result has been received.
