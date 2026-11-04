# Independent review response and R2q source-memory falsification protocol

**Date:** 2026-11-03  
**Status:** Independent review received; **MAJOR REVISION**; R2p and R2q original workstation evidence VERIFIED. R2q original immutable revision `7890c6f363cc9dd61c8bfa29ae121107ee3d86de`: 17 focused and 1,516 full pytest passed; original 36-case source study and separate read-only recomputation audit agreed. See the **verified results** below. New R2r emitted-source shadow/GCC mutation-adequacy experiment has been implemented but not executed.  
**Reviewer file:** `CST_INDEPENDENT_REVIEW_20261103.md` (Claude's standalone report, supplied by author; originally outside GitHub). Its Git evidence tags are **[CODE]**, **[RUN]**, **[RBA]** author-reported vendor data, **[DERIV]** reviewer-derived, and **[LIT]** references. We do not upgrade [RBA] into independently inspected raw Vivado measurements.

## 1. Verified evidence that remains frozen

The author's original R2p source revision `dac395598f2ab96bc9b7a41255b0529c72b22593` passed **29 focused, 1,506 full regression tests** (all passing) and independently ran its create-only six-action, two-domain original GCC E1 study:

| R2p status | Cases |
|---|---:|
| Original actual GCC-E1 qualified | 3 |
| Compiler rejected | 2 |
| Source admitted but bounded E0 UNKNOWN | 1 |
| **Entire fixed source-action cohort** | **6** |

The read-only source and population audit independently agrees with the **3/2/1** counts; it does **not** execute GCC. No original Vitis, Vivado or RL training calls. Original R2n (3/3/3 out of 9), R2o (9 E1/3 rejected out of 12), and original R2i/R2j/R2l/R2m vendor evidence remain immutable. **R2p source E1 is a regression of cross-domain status preservation, not a new RL learning discovery.**

The R2m **physical** evidence is a separate source-pinned six-pair OOC cohort selected **after seeing E3 HLS BRAM-positive results**: all 12 designs routed at requested **32 ns**, all have noncritical DRC warnings, and **0/6 matched pairs qualify for strict clean-DRC E4**. Routed BRAM **6/6 ties at 1.5**, DSP **6/6 ties**, LUT/FF **three increases + three decreases**, descriptive WNS **four increases + two decreases**, mean **+0.335 ns**. **30/36 other physical pairs remain UNKNOWN**. Do not infer achieved Fmax, power, physical savings or statistically representative causal effects.

## 2. Scientific criticism and decisions

| Priority | Review threat | Decision / evidence now required |
|---|---|---|
| **B1 FATAL to present allocation claim** | 36/36 scratch savings compare against disjoint **no reuse**; tensor-level baseline captures much of the gap | Measure a **feasible** whole-tensor first-fit and an explicitly named tensor concurrency **lower bound** on identical source events, plus element-exact fixed-order minimum. Do not describe the tensor lower bound as attained placement |
| **B2 FATAL to present Phi memory headline** | Phi savings depend on committed-value accounting; reader `acc` arrays are live from first MAC and simultaneously exist as full-sized declared arrays | Compute both committed and hypothetical in-place output lifetimes from first reduction read, preserve full **declared** R2h accumulator + counter counts. The in-place model is a **counterfactual**, not existing emitted C++ or guaranteed hardware |
| **B3 FATAL to priority claim** | Interval coloring, polyhedral storage, StreamTensor, Filament, MODeL and related HLS already overlap | Narrow novelty to the exact combined **event semantics / map-conformance / emitted-source witness**; read peer mechanisms and document semantic/capability comparison. No firstness statement without evidence |
| **B4** | HLS −1 BRAM estimate only at >64-word no-reuse scratch; cycle improvement may be address-table/binding artefact | **Future new CSynth study**: storage binding × address generation × padding, with exact testbench and report source SHA. No retroactive reinterpretation or rerunning original 72 cases |
| **B5** | `Phi` dictates a **stipulated sequential issue order**, not hardware timing | State a source-order theorem under explicit loop/accumulator assumptions. No RTL/Fmax/physical temporal claim |
| **B6** | Event-level mutations and shared source specification do not prove emitter conformance | **Future GCC study** with independent shadow memory and at least one off-by-one/slot-table emitter mutation, preserving failing cases |
| **B7 (C3)** | R2o E0 legality label is constant across valid reader-PE edits (reviewer 333/333 admitted) | **Stop calling C3 a distinct new RL method**. A typed action's *certified quantitative capacity change* may be informative, but that is different from mapping-dependent legality and requires new experiments. No trained RL or sample-efficiency claim |
| **B8–B10** | Weak 32 ns OOC timing, single route each; narrow template; permissive 0.1 fixed-point tolerance | Preserve original physical data, report warning-only; devise **prospective** timing/precision experiments only after B1/B2/C have scientific signal |

Claude independently reports (not yet independently reproduced by R2q) **2,583 reductions / 1,532 ties / 0 increases** in 4,115 Phi interventions under committed-value accounting, versus **0 reductions / 4,115 ties** under reader-output in-place accounting. These results concern the reviewer's extended grid, **not** the original 27 R2f interventions. A further reported observation is that the R2h emitter declares both full-sized `acc` and per-element scratch. This is why R2q must report two non-equivalent objectives without silently favoring one.

## 3. R2q implementation and exact interpretation

R2q code **was executed and verified** at original revision `7890c6f363cc9dd61c8bfa29ae121107ee3d86de`:

- `model_ir/source_allocation_baselines.py` implements independently extracted source-value intervals and a **whole-tensor first-fit feasible block allocator**, checked with the existing actual source-event replay verifier. It also computes the **whole-tensor peak lower bound**, existing exact classical element-coloring capacity, and a separate **hypothetical in-place reader-output value-liveness** from the first H reduction READ.
- `experiments/cst_r2q_accounting_baselines.py` reuses the original **36 source graph + Phi catalog**, re-derives E0 intervals, compares both actual emitted R2h C++ array declarations and manifest hashes for disjoint vs element-exact, and reports a separately denominated **27 Phi interventions**.
- The full 36-case tables explicitly separate **scratch-only**, **all actual declared mutable arrays** (scratch + reader accumulator + counter), and **hypothetical fused in-place** (reader output live during accumulation, no second full-size `acc` array). Hypothetical means **not generated, not actually gcc-executed, not physical memory**.
- The original create-only study emits `cases.json`, `summary.json` and `baseline_ladder.csv`; the separate read-only audit re-computes all 36 source rows, checks exact Git revision and all CSV values/source hashes. Neither calls GCC, Vitis, Vivado or a trained RL policy. The source-only analysis is **not** a new physical optimization result.
- `tests/test_cst_r2q_source_accounting.py` includes a synthetic source interval/first-MAC counterexample, tensor block feasibility, replay, malformed-reader rejection, all 36 catalog controls and the reviewer's concrete **committed 17 → 16 versus hypothetical in-place 19 → 19** two-reader case.

**Verified R2q original results:** 36/36 disjoint-to-element scratch decreases, 32/36 feasible tensor-first-fit decreases versus disjoint, 36/36 element-exact decreases versus feasible tensor-first-fit, 36/36 element-exact decreases versus optimistic tensor peak lower bound. Across 27 original Phi interventions, **committed-value reduced 18, tied 9, increased 0 (maximum 2 slots)**; hypothetical in-place **reduced 0, tied 27, increased 0 (maximum 0 slots)**. This is a **source accounting falsification**, not a live fused-accumulator execution; original new GCC/Vitis/Vivado calls remain zero.

The code reuses established model semantics and actual source emitter, **not** the external reviewer's throw-away Python scripts. The alternative first-MAC lifetime uses a **conservative inclusive same-event rule**: a new accumulator becoming live during the last READ of some H value is counted as overlapping H until an explicit micro-operation or bank-port ordering proof exists. A separate synthetic negative control catches a false same-event slot reuse. A small independent arithmetic rehearsal (not the source-pinned R2q campaign) reproduced **18 committed-value reductions, 9 ties, 0 regressions** versus **27 in-place ties** across the original 27 policy interventions; the **official R2q results are still pending**. Re-derivation may corroborate or contradict Claude; preserve any mismatch and resolve assumptions, do not adjust data to the expected answer.

**Crucial scope:** This source-only R2q stage does **not** yet supply a separately implemented in-place C++ emitter, CSim or ILP/CP-SAT tensor-address global optimum. A tensor concurrency lower bound is not a feasible optimum; this distinction is made explicit in every output. These are still future parts of reviewer experiment A. The existing element minimum is classical coloring for one fixed total order, not a novel optimization algorithm.

## 4. Original local execution commands (ALREADY COMPLETE; do not rerun original v1)

**Historical reproduction commands only.** The user's original R2q study already succeeded and is frozen. Do **not** rerun or overwrite its `v1` output. The commands below document that completed execution; any new prospective cohort must use new versioned paths and source SHA. No GitHub Actions or tokens.

```bash
cd ~/Desktop/github_projects/fpgai_research

git status --short
git pull --ff-only origin main
python -m compileall -q src/fpgai_research experiments/compiler_ir tests

python -m pytest -q \
  tests/test_cst_r2q_source_accounting.py \
  tests/test_compiler_ir_series.py

python -m pytest -q
```

Only if tests pass:

```bash
python -m fpgai_research.experiments.cst_r2q_accounting_baselines \
  --output build/r2q_adversarial_accounting_36_v1

python -m fpgai_research.experiments.cst_r2q_accounting_baselines \
  --study build/r2q_adversarial_accounting_36_v1 \
  --audit-output build/r2q_adversarial_accounting_audit_v1.json

cat build/r2q_adversarial_accounting_36_v1/summary.json
```

Both output locations are **strictly create-only**. The audit requires the current **clean original Git revision** to equal the original study revision; use the source-pinned original checkout for it, and never delete original v1 results to rerun. **Do not** rerun R2i/R2j/R2l/R2m or R2n/R2o/R2p as part of R2q.

## 5. Scientific interpretation and hard stop conditions

R2q **has now been locally verified**, and the main manuscript contains the actual 36-row baseline and 27-intervention counts, not extrapolated review figures. Compare:

1. source exact slots vs **no reuse** and **feasible whole-tensor first fit**;
2. source exact slots vs optimistic whole-tensor concurrency **lower bound** (not an actual allocator);
3. actual all declared mutable array elements;
4. **hypothetical** in-place accumulation peak (distinct memory model);
5. committed vs in-place **Phi effects** across exactly 27 controlled interventions.

If the Phi effect vanishes under the natural in-place model, keep C1's final-reader relation as a *source-schedule semantics result* only; do not call it a general physical scratch/BRAM advantage. If a feasible tensor reuse already captures almost all total mutable storage benefit, drop "new allocator" as a primary contribution. If the valid Phi-edit legality label is constant, keep C3 as a constrained engineering research interface; a quantitative certified `delta_capacity` property is not by itself a novel RL algorithm.

**Unfinished next gates:** B/large-scale Phi bounds and theorem, C/HLS storage-binding and addressing factorial, D/dynamic shadow-memory emitter mutation, E/full action-state sensitivity/proposer experiments, and prospective tightly controlled physical repeats only if the source and CSynth mechanisms survive. Claude's independent review also asked for journal-level related-work capability evidence and removal of outdated manuscript assertions.

## 6. Submission wording

**Candidate narrowed title:** *Checking Mapping-Conditioned Tensor-Value Lifetimes in Generated FPGA Source: A Bounded Event Semantics and an HLS-to-Routing Case Study.*

**Current status: MAJOR REVISION, NOT YET SUBMISSION READY.** The most defensible scientific direction is a testable event/refinement semantics and source-checker contribution, with the honest negative HLS-to-physical observation. The RL-usable typed architecture decision interface remains a **secondary bounded system property**, not a demonstrated scientific improvement in learning. The full original resource/WNS table stays in the paper, with selection, DRC and 32 ns timing qualifications.

Historical manuscript sprint notes were moved verbatim to `HISTORICAL_MANUSCRIPT_SPRINT_NOTES_20261103.md`, leaving `MANUSCRIPT_DRAFT.md` to begin with its actual paper title.
