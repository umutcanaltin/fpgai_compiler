> **2026-11-03 original R2d runtime-loop GCC cohort — workstation VERIFIED.**
> At original SHA `09ac395ef5c73cae59940c4761f3255d695882d1`,
> `compileall` succeeded; focused R2b/R2d source tests
> **23/23 passed**, full pytest **1,310/1,310 passed**
> in 122.16s. Original new create-only source cohort
> `build/cst_r2d_original_runtime_loops_v1` completed
> **12/12** real GCC compilations and executions, with
> **12/12** runtime-loop Phi order, reduction/alias and
> independent NumPy checks qualified. The original
> separate read-only audit under
> `build/cst_r2d_runtime_independent_audit_v1`
> rechecked **12/12** stored source/trace SHA pairs,
> mapped READ order, alias/reduction, and NumPy
> numerical outputs, with zero new GCC/Vitis/Vivado
> calls. Source numerical absolute errors were
> approximately 5.93e-8 to 1.19e-7.
>
> This is **independently checked source-level
> runtime loop index behavior**, not measured
> FPGA-cycle scheduling, RAM port/BRAM resources,
> general DAG behavior or external scientific firstness.
> Original artifacts MUST NOT be re-executed or
> overwritten. Prior R2b/R2c and C2 E0 cohorts
> remain separately source pinned.
>

# R2d — independent runtime-subscript C++ source-order experiment

**Prospective, not yet executed on the user's workstation.**
2026-11-03. Research repository `fpgai_research`.
This new study addresses independent Claude review finding
**F1**: original R2b/R2c GCC source traces are printed
by straight-line C++ blocks with source-literal event
positions, so the trace cannot independently verify
the compiler's intended Phi issue order.

## Historical evidence and scope

The user-verified original C2 E0 mutation study at source
`82ef9f2c0fe101dadcf6a1c013540b8f3780abf6`
completed **96/96 selected E0 mutants detected,
zero survivors**, with all 8 prespecified fault
classes over 3 static models × 4 reader mappings.
The same workstation ran **21/21 focused tests**,
and a full regression with **1,294 passed,
1 failed**. The only failed assertion
expected `SOURCE_EVENT_SEQUENCE_MISMATCH` after
manually forging a GCC trace, whereas the newly
enforced independent Phi issue oracle correctly
rejected it earlier with
`Independent Phi source issue proof rejects GCC trace`.
A test-only expectation fix was committed
subsequently. The user's original 96-mutant
population and output folder must NOT be rerun.

Older R2c source experiment at
`e6db1ec5458a3d6e335b8eed665945c033a8b868`
passed **1279** tests, with
12/12 matched original GCC scratch-source
pairs and 9/9 changed bounded optimal
scalar scratch-slot counts under Phi edits.
That is a previous, independently preserved
study, not new R2d evidence.

## Research question

Can a compiler generate **loop-structured C++**
whose logical element READ addresses are
calculated at **C++ runtime**, and can a
separately implemented mathematical Phi
issue reference predict the actual READ
order and ownership over bounded
heterogeneous reader mappings?

A successful R2d E1 source test would be
stronger than accepting that a statically
unrolled event-list generator reproduces
its own printed events. It would NOT prove
physical FPGA source-to-RTL event/time
equivalence, RAM-bank/port legality, or
new schedule algorithms beyond prior art.

### New independently scoped implementations

1. `codegen/hls/r2d_loop_driven_branch.py`
   emits a C++ runtime loop over the ordinal
   `0..m*n*q-1` for two independently mapped
   MatMul readers. Within each ordinal,
   `r2d_decode_issue` reconstructs the
   logical `row, col, reduction` subscripts,
   forms a C++ `source_idx` and performs a
   *real* `scratch[source_addr]` load.
   The event trace prints that runtime index
   and computes the per-operator output index
   from the same variables. The source performs
   dynamic reduction completion and Add/Relu.
   No Python READ event list is transcribed into
   individual C++ statements.
2. `validation/r2d_loop_driven_cpp.py`
   compiles and executes the original C++
   source with GCC, parses actual stdout
   and independently checks read order/owner
   against the closed-form
   `model_ir/independent_phi_issue_reference.py`,
   then independently checks mapped reduction
   completeness/alias safety and NumPy
   Functional Model output. It does
   **not** assert mere equality to the
   generator's prewritten event list.
3. `experiments/cst_r2d_loop_campaign.py`
   freezes a separate original source-pinned
   **3×4 = 12-case** experiment and preserves
   generated source, original stdout, full
   model/Phi/input receipt and a separate
   result manifest, with create-only paths
   and retained failure artifacts.
4. `experiments/cst_r2d_loop_audit.py`
   independently reparses saved original
   stdout and checks all original SHA/NumPy/
   source-order/reduction certificates
   **without recompiling** or new vendor calls.
5. `tests/test_cst_r2d_loop_driven_cpp.py`
   deliberately mutates the runtime C++
   rank decoder so that every output can
   still receive its full reduction but the
   mapped READ order changes, then requires
   the independent Phi oracle to reject.
   It also checks early COMMIT and wrong
   output values. Source identity, odd q
   partial PE groups and one C++ address
   variable per scratch load are tested.
6. `tests/test_cst_r2d_loop_campaign.py`
   verifies 12 complete source cases,
   read-only audit, source trace SHA
   tampering, failed case refusal and
   summary-count inflation refusal.

**Conservative interpretation:** The
R2d C++ loop computes runtime indices
from a deterministic ordinal and a fixed
mapped reader policy. The source still
has only one host thread and is logically
serialized. This is independent
**source issue-rule conformance testing**,
not independent proof of a hardware
concurrency schedule. Source E0 policy
and E1 actual GCC behavior must remain
separately labeled.

## Prespecified original source population

| Setting | Contract |
|---|---|
| Functional graph | `X W0 → H; H W1 → A; H W2 → B; A+B → G; Relu(G)` |
| Shapes `(m,k,n,q)` | `(2,2,3,4)`, `(2,3,5,4)`, `(3,3,4,4)` |
| Reader Phi | scalar, reader0 PE2, reader1 PE2, both PE2 |
| Source layout | Existing last-reader-reuse whole-tensor offsets |
| Original tool | GCC `-std=c++17 -O0 -ffp-contract=off`, runtime index logging |
| Output checks | Independent closed-form source issue rank, complete A/B and G reductions, alias safety, independent NumPy |
| Original GCC budget | 12 compilations and executions, zero Vitis/Vivado |
| Evidence preservation | source.cpp, case.json, raw original_gcc_trace.txt, verified.json, cases.csv, summary.json; FAILED.json on failures |

The emitted source has **seven explicit
mutable arrays**: scratch, acc_a/b,
add_a/b, and two integer reduction
completion counter arrays. The
source manifest lists these separately
and **does not compare C++ byte or
array sizes to native synthesized BRAM**.

## Exact workstation instructions

First verify new source and tests. Do not
repeat an original run after an assertion
fails. The original source directories
are immutable. The two output paths below
must be new/unused.

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests

# Regression and independent fault controls:
python -m pytest -q \
  tests/test_cst_r2b_source_campaign.py \
  tests/test_cst_r2d_loop_driven_cpp.py \
  tests/test_cst_r2d_loop_campaign.py

# Full regression:
python -m pytest -q

# Only after ALL tests pass, new R2d original GCC source study:
python -m fpgai_research.experiments.cst_r2d_loop_campaign \
  --output build/cst_r2d_original_runtime_loops_v1

# New read-only audit (no GCC re-execution):
python -m fpgai_research.experiments.cst_r2d_loop_audit \
  --study build/cst_r2d_original_runtime_loops_v1 \
  --output build/cst_r2d_runtime_independent_audit_v1
```

A mutation-test failure or a new actual
source issue mismatch MUST be preserved
and reported, not silently fixed by
changing the independent reference. If an
original R2d case fails, save its
`FAILED.json`, diagnose the code, then
run a changed source revision in a new
create-only output directory.

## Scientific status and next gates

At authoring time **the R2d code, new tests,
12-case original source run and audit
have NOT been workstation-verified**.
The original 96/96 E0 mutation result is
a separate completed E0 cohort; it
establishes only detection of prespecified
faults, not arbitrary soundness.

Next, full producer CONTRIBUTE semantics,
hardware port/timing/equivalence
witnesses, truly general fan-out DAG
composition, external published peer
parity and original native fixed-point
CSim/CSynth/strict clean-DRC comparisons
remain missing. The current scientific
publication gate stays closed.
