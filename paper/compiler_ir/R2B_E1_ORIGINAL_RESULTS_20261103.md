> **November 3, 2026 — original independent R2b causal audit verified.**
> The workstation successfully completed the **read-only**
> `cst_r2b_causal_audit` for original E1 source revision
> `5d449ca595dbdfdf7a837ce862eb5fa8f89ce9ce`.
> All **24/24** saved source cases were reverified with
> source SHA, GCC stdout, independent NumPy numerical evaluation
> and exact H-only read ranks/consumer identity.
> The **18 layout-scoped contrasts are 9 distinct graph/Phi
> interventions** (two memory-layout controls each).
> Results: **9/9** changed a global event index,
> **6/9** changed H-only last-read rank or consumer,
> **3/9** changed final consumer identity,
> and **3/9** had no H-only last-read rank/consumer
> change. All **12/12** disjoint-vs-reuse layout
> pairs were numerically/event equivalent.
> Source scratch savings from the layout controls
> were **8, 8 and 12 elements** for the respective
> fixed graph shapes, independent of reader Phi:
> **no mapping-conditioned capacity saving is demonstrated**.
> Audit invoked **zero new GCC/Vitis/Vivado executions**.
>
> **Test-suite status during that workstation run:**
> focused **3 passed, 1 failed**; full **1,256 passed,
> 1 failed**. The sole failure was an expected-exception
> *wording mismatch* in the duplicate-case mutation test.
> The lower-level source auditor correctly rejected
> the duplicate, saying `Duplicate original R2b source case`.
> The test has now been corrected on GitHub and a
> separate unsupported-policy rejection test added.
> These latest test changes have **not yet been
> executed on the workstation**. Original experiment
> validity and evidence are unaffected by this assertion mismatch.
>

# R2b original GCC E1 — observed Phi-dependent source READ ordering

**Date:** 2026-11-03. **Original workstation source revision:**
`5d449ca595dbdfdf7a837ce862eb5fa8f89ce9ce`.
This is a **bounded research compiler E1** observation
with a fixed, prospectively selected PE-group source issue rule,
not an assertion of novel prior-art firstness.

## What was actually executed and independently audited

- Full original workstation regression: **1,253 tests passed**,
  including **8 focused** R2b source-campaign tests.
- Create-only original R2b output
  `build/cst_r2b_original_gcc_e1_v1`:
  **24/24 real GCC compiled C++ source files executed**,
  **24/24** actual source event sequences and independent
  float32 NumPy functional results passed.
- Original complete population: **3 fixed five-op MatMul diamonds**
  × **4 reader Phi policies** × **2 scratch layouts**.
  Original reported 18 layout-scoped nonbaseline contrasts
  (3 shapes × 3 Phi edits × 2 layouts); zero
  original Vitis/Vivado calls.
- The separate read-only
  `cst_r2b_source_audit` verified **24/24** original
  saved source and stdout SHA-256 receipts, independently
  reparsed original actual READ/COMMIT events and final
  consumer source ranks, without rerunning GCC or vendors.

## Exact original result and denominators

| Observable measured from actual original GCC source | Layout-scoped contrasts | Unique graph/Phi interventions |
| --- | ---: | ---: |
| Change in global last-READ **event index** (may include intervening COMMITs) | **18 / 18** | **9 / 9** |
| Change in last H READ **rank or final consumer/output** | **12 / 18** | **6 / 9** |
| Change in final H reader **consumer identity** | **6 / 18** | **3 / 9** |
| Unchanged last H-only READ rank and reader/output | **6 / 18** | **3 / 9** |

**Correct statistical unit:** The two source scratch layouts
for the same (graph, Phi) are **not independent Phi
interventions**. The original 18 layout-scoped contrasts
are **nine distinct mappings, each present twice**.
The more conservative denominators are therefore
**6/9** genuine H-only last-read ordering differences
and **3/9** final-reader-consumer identity changes.
This arithmetic is supported by original summary counts
and matched source layout checks; a NEW independent,
read-only `cst_r2b_causal_audit` will validate
all nine row-level counts from saved original inputs,
GCC READ traces and the independent NumPy evaluator.
**That NEW second audit has not yet been executed
on the workstation.**

**The old serial-source study remains a separate
original negative result.** It found 0/18 changed
last-use and 0/18 changed scratch capacity after
18 effective edits, because the original serial
stage order did not interleave two readers.
The R2b schedule is a prospectively different
lowering algorithm, not a reinterpretation of
that original observation.

## What the positive result legitimately establishes

Under a **fixed prospective Phi-dependent C++
source scheduling algorithm** for the supported
two-reader static MatMul graphs, a legal PE mapping
change can alter **actual GCC-executed source-level
element consumption order**, including which
reader is last. This is stronger than a changed
Phi-owner hash or an unexecuted source schedule.

An original GCC event ordinal is **not hardware time**.
There is **no demonstrated reduction in memory
capacity conditioned on Phi**: 12/12 source
scratch-layout savings are a comparison of
different disjoint/shared storage policies,
not a same-layout Phi-dependent capacity saving.
The generated source uses independent local
accumulator arrays; a smaller measured scratch
array does not establish a smaller total physical
memory implementation.

No new native Vitis CSim/CSynth, Vivado, RTL
bank-port schedule, achieved Fmax, power,
trained RL model or externally matched
published compiler comparison was performed.
Do not call this hardware memory optimization,
a correctness theorem over arbitrary DAGs,
or scientific firstness over ST-Flow, Allo,
HIDA, StreamTensor or any other peer.

## Independent read-only post-run reanalysis

The **original case files must remain immutable**;
the next command reuses the 24 saved cases and
performs NO new original GCC executions:

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m pytest -q tests/test_cst_r2b_causal_audit.py
python -m pytest -q

python -m fpgai_research.experiments.cst_r2b_causal_audit \\
  --study build/cst_r2b_original_gcc_e1_v1 \\
  --output build/cst_r2b_independent_causal_audit_v1
```

The output is create-only and includes
`r2b_causal_audit.json` containing the nine
**unique** intervention records and the twelve
layout controls. The auditor independently
recomputes the original FunctionalModel
and Phi state, verifies saved input bytes,
re-evaluates NumPy output against the actual
GCC numeric trace, compares raw H-only final
reader rank/identity across exact matched
controls, and rejects any mismatch with
the original case CSV/summary denominators.

That audit is a secondary validation of the
same original experimental data, not a new
independently reproduced original compiler
run or an external scientific peer baseline.

## Next research gates

1. Extend beyond static 3-graph, fixed five-op
   MatMul model into independent NCHW graph
   shapes/operators and source-level genuinely
   interleaved producer/consumer dependencies.
2. Evaluate exact small optimal aliasing
   decisions conditioned on source-verified
   element lifetime, using an independent
   exhaustive verifier and exact source
   capacity (including all intermediate arrays);
   preserve ties and regressions.
3. Verify native fixed-point CSim/CSynth and
   timing/resource feasibility under
   consistent numerical constraints; the
   previous v6 12-case warning-only routed
   cohort is a different immutable study.
4. Compare source-pinned external peer
   artifacts using equivalent functions,
   precision, mapping controls and supported
   experimental features. Scientific
   priority/firstness remains UNKNOWN.
5. The CST policy-neutral RL-compatible
   interface remains secondary to the
   CST Model IR; no RL training/convergence
   outcome follows from the E1 study.

**Publication status: not submission ready.**
