# FPGAI original-Vitis Gym curriculum: L0 → L6

**Version 1 / 1 November 2026.** This is a series of **seven separately source-pinned REAL Vitis HLS experiments**, not an automatically trained simulator or fabricated learning curve. All stages train the SAME GNN architecture **from scratch** under different hardware-search tasks. Older experiments remain immutable; the default compiler and standard Gym are unaffected.

## Mandatory minimum training in EVERY environment

**L0–L6 each require at least 100 complete online training episodes, with exactly 10 original source-attested Vitis HLS evaluations in EVERY training episode.** This is enforced in the planner, native runner, independent auditor and scorecard, not just a suggested default. More than 100 episodes are allowed. A failed, truncated or partly sourced study is **incomplete**, not evidence of non-learning.

The optional explicit `--smoke` mode is a *different* short source-qualification campaign: 2 train episodes ×2 steps, 1 validation and 8 test/control episodes ×2 steps (22 original HLS calls maximum). Its cases never count toward the 100-episode minimum.

Seven full experiments at 100 episodes per stage require **7,000 original HLS training evaluations**, plus up to 630 matched validation/test-control evaluations, for **up to 7,630 calls total**. Each stage must report actual per-episode reward and feasibility, independently audited heldout trained versus untrained latency, plus inconclusive evidence rather than inventing learning or failure-to-learn.

## Research objective

Test the hypothesis that a shared GNN can learn increasingly general hardware architecture configurations with **100 training episodes × 10 original Vitis HLS evaluations**, beginning with very small MatMul kernels and ending with heterogeneous network widths, precision choices and learnable requested clocks. Demonstrate progress with authenticated heldout trained-vs-untrained original HLS latency, not visual smoothing of a reward curve.

| Stage | Models trained | Frozen parameters | Searchable parameters | Heldout topology |
|---|---|---|---|---|
| **L0** | 1 MatMul, 4→4, 4→8, 8→4, 8→8 | 32 ns, 16/6, tile, pipeline II | parallelism, PE, SIMD, reuse | new single-layer widths |
| **L1** | 1 MatMul, several additional fixed widths | 32 ns, 16/6, tile, pipeline II | same 4 knobs | unseen widths of same depth |
| **L2** | 2 MatMul/Gemm operators | 32 ns, 16/6, tile, pipeline II | 4 knobs per operator | unseen 2-op widths |
| **L3** | 1–2 MatMul/Gemm operators | 32 ns and 16/6 | above + tile + pipeline II | 1–2-op holdout |
| **L4** | 1–2 MatMul/Gemm operators | 32 ns | all 8 catalog knobs, including globally synchronized 16/6 alternatives | unseen 1–2-op models |
| **L5** | 1–2 MatMul/Gemm operators | no fixed clock | all 8 knobs + continuous requested clock 2–32 ns | unseen 1–2-op models |
| **L6** | 1–2 MatMul/Gemm operators | none | all 8 knobs + continuous requested clock | **3-op validation and 4-op test transfer** |

Precisions in L4–L6 are constrained by real compiler legality; no unsupported mixed-precision operations are admitted. Each stage is independently seeded from scratch; this v1 benchmark does **not** claim learning-transfer via pretrained checkpoint warm starts.

Each observation retains the existing typed Model IR-derived node/edge mask, hardware resource caps, numeric-error limit, bounded true-Vitis trial history, clock bounds, and action menu. The GNN is shared across all **tasks within a stage**, never a separate policy for each task. Only original Vitis CSim+CSynth source-pinned case receipts can supply numeric, resource, clock and latency feedback.

## Truthful dense reward and scientific comparisons

At **each curriculum stage only**, accepted source-attested native HLS cases receive the same monotone within-task latency reward:

```
MACs = sum(exact Functional Model IR MatMul/Gemm multiply-accumulates)
reference_ns = 32.0 * MACs
actual_HLS_ns = original_HLS_worst_case_cycles * original_requested_clock_ns
R = reference_ns / (reference_ns + actual_HLS_ns)     # strict feasible only
R = -0.05                                              # all other cases
```

- This is a dimension-normalized scalar reward from **real HLS cycles and requested clock**, never a fake latency prediction. Area/resources and accuracy remain **hard gates**, not rewarded objectives.
- The full product's and previous general Gym study's `latency_only_reward` remain unchanged. **No equivalence to best-of-10 optimization is claimed**; compare actual best feasible HLS ns in matched heldout episodes instead.
- Training reward scales are **not quantitatively comparable across different networks/stages**. Plot stage-specific reward/feasibility curves, trained-versus-untrained original HLS latency and regret where valid. A genuine learning result can fail to emerge; disclose that rather than manufacture a trend.

## The unresolved evidence bug is a hard stop-gate

In the previously independently replayed 1,009 original HLS cases, **850 timing validation failures were classified as `different_reported_target`** (requested vs XML TargetClockPeriod mismatch), NOT missing clock uncertainty. The later original XML probe also observed **911/1,009 different XML target periods**, including some cases rejected by other qualification gates. Sample differences were 0.001–0.004 ns: the original requested clock had three decimals, while Vitis XML printed two. **New staged experiments canonicalize requested period and explicit uncertainty to 0.01 ns**, preserving the original HLS XML target/estimate source evidence and all strict qualification rules. The unmodified old original study remains source-pinned. This prospective fix must still pass genuine native 22-call smoke and independent audit. Merely relaxing the clock or inserting uncertainty Tcl does not establish validity.

Before any further expensive native run, use the read-only XML diagnosis:

```bash
python -m rl_environment.clock_mismatch_probe \
  --study build/gym_100_train_10_hls_seed101_learning_v3
```

It shows the actual requested/XML target clock values and differences in a sample of the original native cases. Original HLS XML and immutable case files are not modified. Do **not** overwrite reported XML targets or silently widen the request-match tolerance to improve success. Investigate actual Vitis Tcl/solution clock behavior and correct source generation only when verified by authentic new native runs.

For the curriculum, a newly created native case uses the explicit `set_clock_uncertainty` command in the original executed and hash-attested Tcl, with 27% of the requested target rounded to 1 ps. If XML omits uncertainty, the parser may use only this **verified executed setting**, and still requires fresh XML target, estimated period, numerical, resource and timing qualification. A target mismatch is NOT fixed by this uncertainty policy.

## Environment setup and compiler source-binding gate — REQUIRED

**2026-11-01 L0 smoke failure diagnosed:** the workstation passed 734 CPU tests, then `python -m rl_environment.gym_curriculum --stage L0 --smoke --execute-native` failed **before any Vitis invocation** at `write_vitis_package(..., clock_uncertainty_ns=...)` with `TypeError: unexpected keyword argument`. The source checkout **already declares that exact keyword**. Root cause is most likely an older installed `fpgai_research` imported by the native CLI, while `pytest` silently prepends `src` via `pyproject.toml`. This has not yet been verified on the workstation by module-origin inspection.

Install the current checkout **editable, without redownloading dependencies**, inside the intended venv; then verify exactly which physical module Python imports, without launching Vitis:

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m pip install --no-deps -e .
python -m rl_environment.runtime_contract
python -m compileall -q rl_environment src tests
python -m pytest -q
```

`runtime_contract` must print `LOCAL_CURRENT_SOURCE_VITIS_PACKAGING_BINDING_VERIFIED`, the checked-out `src/fpgai_research/toolchains/vitis_hls/package.py` file location and `original_vitis_calls=0`. If it reports a different path, **do not run native smoke**. Every new native `gym_study.run` now repeats this verification *before* allocating its create-only output directory, so an outdated installed library cannot silently consume an HLS budget or strand a new study.

The failed initial `build/gym_curriculum_L0_smoke_seed111_v2` directory is partial and **must not be relabeled as a completed, audited source study**. Leave it untouched for provenance until inspecting its contents; create a fresh `v3` directory for a rerun.

```bash
python -m rl_environment.gym_curriculum --stage L0 --smoke \
  --seed 111 --output build/gym_curriculum_L0_smoke_seed111_v3 \
  --execute-native --max-total-vitis-calls 22 --vitis-executable vitis_hls

python -m rl_environment.gym_audit \
  --study build/gym_curriculum_L0_smoke_seed111_v3
python -m rl_environment.qualification_funnel \
  --study build/gym_curriculum_L0_smoke_seed111_v3
```

Do not begin 100×10 full L0 until this new smoke passes its strict source, CSim, resource and original Vitis timing audit gates. The native packaging interface now has dedicated no-Vitis regression tests calling the actual `write_vitis_package` with explicit uncertainty rather than testing only Tcl generation.

## A. Plan all seven stages with ZERO Vitis calls

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q rl_environment src tests
python -m pytest -q
python -m rl_environment.gym_curriculum --list-stages
python -m rl_environment.gym_curriculum --plan-all \
  --train-episodes 100 --max-vitis-compiles-per-episode 10
```

Seven stage plans each show 100 online-train + 1 validation + 8 equal-budget heldout/control arms = **109 total episodes / ≤1,090 Vitis calls per stage**. **Do not run all seven on the first day:** that would be up to 7,630 original HLS calls. Only L0 should begin after its gates; later stages follow individually.

## B. L0 short native smoke, only after requested-vs-reported clock diagnosis

```bash
python -m rl_environment.gym_curriculum \
  --stage L0 --smoke \
  --validation-episodes 1 --test-tasks 1 --seed 111 \
  --output build/gym_curriculum_L0_smoke_seed111_v1 \
  --execute-native --max-total-vitis-calls 22 \
  --vitis-executable vitis_hls

python -m rl_environment.gym_audit \
  --study build/gym_curriculum_L0_smoke_seed111_v1
python -m rl_environment.qualification_funnel \
  --study build/gym_curriculum_L0_smoke_seed111_v1
python -m rl_environment.clock_mismatch_probe \
  --study build/gym_curriculum_L0_smoke_seed111_v1
```

Smoke acceptance is **predefined**: all source case replay checks pass; no source/CSim/report validation failure, at least 14/22 strict qualified original cases, with genuine distinct architecture choices under fixed 32 ns and 16/6. This is an engineering debugging threshold, not proof of RL learning. If it fails, fix the actual root cause first and use a **new** output name; do not rewrite the original case.

## C. L0 full run only after passing its original-HLS smoke

```bash
python -m rl_environment.gym_curriculum \
  --stage L0 --train-episodes 100 --validation-episodes 1 \
  --test-tasks 1 --max-vitis-compiles-per-episode 10 \
  --seed 111 --checkpoint-every-train-episodes 10 \
  --output build/gym_curriculum_L0_100x10_seed111_v1 \
  --execute-native --max-total-vitis-calls 1090 --vitis-executable vitis_hls

python -m rl_environment.gym_audit \
  --study build/gym_curriculum_L0_100x10_seed111_v1
python -m rl_environment.paper_evidence \
  --study build/gym_curriculum_L0_100x10_seed111_v1 \
  --output build/gym_curriculum_L0_100x10_paper_v1
python -m rl_environment.gym_report \
  --study build/gym_curriculum_L0_100x10_seed111_v1 \
  --output build/gym_curriculum_L0_100x10_report_v1
python -m rl_environment.gym_curriculum_report \
  --study L0=build/gym_curriculum_L0_100x10_seed111_v1 \
  --output build/gym_curriculum_L0_scorecard_v1
```

To advance to L1, inspect **matched actual heldout** trained vs never-trained GNN, feasibility, baseline searches, independent original-HLS audit and task coverage. If no feasible pair exists, the correct answer is *latency improvement not identified*. A single L0 seed can show a pilot trend but cannot establish statistical generalization.

## D. Subsequent environments

For **each** further stage L1..L6, first run its own fresh 22-call smoke, independent audit and clock/provenance diagnosis; only then start the **same 100×10** training/heldout command with a stage-specific create-only output path. For example:

```bash
python -m rl_environment.gym_curriculum \
  --stage L1 --train-episodes 100 --validation-episodes 1 \
  --test-tasks 1 --max-vitis-compiles-per-episode 10 --seed 111 \
  --output build/gym_curriculum_L1_100x10_seed111_v1 \
  --execute-native --max-total-vitis-calls 1090
```

Do not warm-start weights between stages in this v1 protocol: this is a **controlled increasing-complexity benchmark** with common GNN architecture, not a lifelong pretrained policy. If warm-start transfer is later scientifically desired, register a separate matched trained-from-scratch-vs-warm-start experiment and keep both traces.

Once several complete native stage studies have passed audits, compare them without mixing raw cross-task latency:

```bash
python -m rl_environment.gym_curriculum_report \
  --study L0=build/gym_curriculum_L0_100x10_seed111_v1 \
  --study L1=build/gym_curriculum_L1_100x10_seed111_v1 \
  --output build/gym_curriculum_L0_L1_comparison_v1
```

Outputs: `stage_scorecard.csv`, `heldout_same_task_comparison.csv`, `training_learning_curve.csv`, per-stage reward/feasibility PNGs when Matplotlib is installed, and `stage_report.json` with per-stage qualified fraction and paired same-task trained-vs-untrained ratios. Everything is source-attested and read-only; it invokes zero native programs.

## E. Publication-quality experiment gates

- Prespecify seeds, model dimensions, training budget, source SHA and all heldout tasks. **No selecting only the best-looking seed, reward smoothing as ground truth, or silently changing HLS clock gates.**
- Show training reward *and* HLS feasibility, first-to-last checkpoint progression, source-specific reason distribution and honest 20-episode rolling means (descriptive).
- Report all heldout tasks, including zero-feasible outcomes; compute speedups only where both methods have strict qualified results at the same number of actual native calls.
- Distinguish real Vitis HLS cycles×requested-clock latency estimates from routed physical Fmax. Strict accuracy caps and resource caps are hard constraints throughout.
- Repeat successful stages across independent seeds and multiple pre-registered test dimensions before claiming statistical superiority. More than one heldout topology/seed will be needed for journal-quality evidence.
- Preserve the complete previous `gym_100_train_10_hls_seed101_learning_v3` source-pinned original training checkpoint and the already run 24ns 22-call smoke; they are **separate previous experiments** and must not be overwritten or silently mixed into the new curriculum.

## F. All-seven completeness and learning evidence

After ALL seven separately verified full studies exist, use `python -m rl_environment.gym_curriculum_report --require-all-seven` with seven `--study L0=... --study L1=... ... --study L6=...` arguments and a **new** `--output`. The report refuses missing stages, smoke campaigns, fewer than 100 original training episodes, any training episode missing its 10 HLS calls, and missing equal-budget heldout controls.

The status is deliberately exploratory: `OBSERVED_PILOT_IMPROVEMENT_NOT_STATISTICAL_PROOF`, `NO_CONSISTENT_PILOT_IMPROVEMENT_NOT_PROOF_OF_NO_LEARNING`, or `INCONCLUSIVE_NO_PAIRED_FEASIBLE_TEST_LATENCIES`. A single heldout task/seed cannot demonstrate population-wide learning or prove that learning is impossible. Report all seven outcomes, including negative results.
