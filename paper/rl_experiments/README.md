> **FROZEN RL TRACK — 2026-11-01.** New RL algorithm implementation, environment changes and new Gym/Vitis study starts are paused in favor of the [FPGAI compiler-IR paper](../compiler_ir/README.md). The previously completed 22/22-qualified original L0 smoke at source `5221094af585bd501e842deca94d19c2289bb628` is preserved; an already-started full L0 campaign may complete and be independently audited/read-only reported using its unchanged workstation checkout. Do **not** `git pull` in that running source-pinned checkout or infer any RL learning success until actual full results are available. This folder's plans are historical/prospective backlog, not currently approved additional native runs.

# Native-HLS Gym paper experiments — prospective protocol

**Status: protocol only, not measured results.** This folder defines how to report the environment and the candidate agents in a reproducible paper. Current source-pinned original-Vitis observations are historical evidence, not automatically outcomes of the new L0–L6 curriculum. Never insert simulated figures or placeholder scores into results tables.

## Scientific questions

- RQ1 (environment validity): Do the L0–L6 tasks generate diverse compiler-legal hardware actions, real source-attributed CSim/CSynth evidence, and auditable numerical/resource/timing qualification?
- RQ2 (learning): With the same real-Vitis call budget, does a policy trained from scratch achieve lower **qualified estimated whole-network latency in ns** on heldout models than its untrained counterpart and non-neural baselines?
- RQ3 (algorithm choice): Do graph PPO, A2C, REINFORCE or other actually implemented policies improve median matched heldout latency, qualification rate and sample efficiency versus graph actor-critic? Do not claim these candidate algorithms exist until registered and tested.
- RQ4 (generalization): Can a model-size-independent graph policy trained on shallower task graphs transfer to heldout deeper graphs in L6? No cross-stage pretrained initialization in the existing L0–L6 v1 contract.
- RQ5 (ablation): How do message passing, auxiliary feasibility classification, action-history information and joint clock selection affect qualified heldout latency? Each ablation has its own versioned protocol, Git SHA and output root.

## Paper tables and figures

1. **Environment specification table**: stage, graph size/depth, operator family, legal action knobs, numeric/clock settings, training/test distribution and qualification gates.
2. **Native qualification funnel**: planned calls, actual original HLS calls, CSim-passed, numerically correct, resource-legal, timing-feasible, qualified, unavailable and parse-invalid counts. Include incomplete studies.
3. **Learning curves**: raw per-episode return, rolling window, native call index, feasibility and best qualified HLS ns; show individual seeds and uncertainty. Never smooth away zero-feasible episodes or state learning from reward alone.
4. **Heldout paired comparison**: for the **same task and equal native budget**, show best qualified HLS latency for trained frozen, untrained and heuristics, absolute and relative deltas, paired uncertainty, and missing-feasible counts.
5. **Agent comparison table**: algorithm/version/seed, parameter count, training native calls, heldout native calls, qualified-task coverage, paired latency difference and native wall time. Separate source-only validation from original-HLS verification.
6. **Transfer and ablation figures**: L6 heldout depth results, message-passing on/off, learned clock vs explicitly fixed clock (distinct task contracts), auxiliary classifier on/off; never mix incompatible reward scales.

## Comparison rules

The source L0–L6 full-study contract requires **at least 100 complete online training episodes with exactly 10 original Vitis evaluations per episode**. The minimum is 1,000 training evaluations **per stage, agent and seed**, plus additional equal-budget validation, heldout and baseline evaluations. Use independent seeds, report actual counts, and preregister the seed list, hyperparameter-search budget, stopping rules and excluded-case handling. A 22-call smoke is for source qualification, not a paper learning result.

All competing methods must use the identical frozen task manifest, original model and numerical inputs, resource limits, compilation version, vendor version, execution budget, and fair native-call accounting. Baseline RNG streams must be independent. Training and test task fingerprints must be disjoint. No model-specific trained network, pretrained agent, surrogate QoR, Vivado/place-route claim, or inferred achieved Fmax.

Only compare **qualified HLS worst-case cycles multiplied by the source-attested requested clock ns**; failures remain failures, not zero latency. If either arm has no qualified result, report coverage and censoring instead of silently dropping paired tasks. Across differently sized networks, prefer matched-task relative improvements over averaging raw absolute ns. A full stage conclusion requires an independently rerun read-only audit.

## Per-study evidence

Use separate, never-overwritten output roots such as `build/paper_rl/<protocol_id>/<stage>/<agent>/<seed>/` (runtime artifacts excluded from Git). Preserve:
- immutable protocol manifest, task split fingerprints and frozen experimental hypotheses
- Git SHA, dependency versions, algorithm implementation/configuration, source-checkout preflight
- initialized/checkpoint state and RNG seeds (private, never commit weights)
- each original generated source/Tcl/CSim/CSynth report, source hashes, HLS clock uncertainty, qualification reasons and wall-clock receipt
- episode and trial JSONL, independent audit report, complete raw data and plotting script
- any interruptions, mixed-revision continuations and unavailable evidence labeled explicitly

Publication requires institutional/IP permission and a full proprietary-output and secret audit. See `../../rl_environment/GYM_CURRICULUM_L0_L6.md`, `../../rl_environment/PAPER_EVIDENCE_SPEC.md` and `../../CONTRIBUTING.md`.

## Execution order

1. Audit old results read-only; do not reclassify legacy experiments as prospective curriculum data.
2. Verify the new compiler binding and CPU suite (user workstation: 747 passed, source binding verified on 2026-11-01).
3. Complete and audit an independent L0 native smoke before any full training.
4. Run and independently audit graph actor-critic L0; disclose negative results.
5. Implement and CPU-validate one additional candidate agent before spending further native calls.
6. Expand to other levels and multiple seeds only when the original HLS qualification funnel supports meaningful comparison.
