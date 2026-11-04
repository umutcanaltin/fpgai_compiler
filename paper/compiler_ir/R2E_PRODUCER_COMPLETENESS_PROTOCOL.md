# CST R2e — independently witnessed producer MatMul contribution completeness

**Prospective new research experiment, 2026-11-03.**
**Implementation is committed but has not yet run on the user's workstation.**
Only the research repository `fpgai_research` is changed.
No old GCC, C2 E0 mutation, native HLS, or Vivado original
case is rerun, copied as new original evidence, or overwritten.

## Motivation — what the original R2d experiment did and did not establish

Original R2d at source
`09ac395ef5c73cae59940c4761f3255d695882d1`
has an independently audited original workstation result:
**23 focused tests passed, 1,310/1,310 full tests passed;
12/12 original GCC runtime-loop source cases qualified; 12/12
saved original cases rechecked by the independent read-only auditor**.

That is a genuine *bounded host-source* demonstration that
reader Phi issue order and mapping owner conform to a
separate arithmetic oracle and final output agrees with the
NumPy Functional Model. It is **not** hardware scheduling.

The original `validate_branch_events` C2 verifier checks
that all consumer H READs precede A/B reduction COMMIT, and
that both Add operands precede G COMMIT. **It cannot
independently verify the H producer's input×weight
reduction completeness**, because the original event
language has only intermediate-value `READ` and `COMMIT`
and no producer `CONTRIBUTE` events.

A final ReLU output check also does not necessarily expose
incorrect intermediate H producer values. This is the
adjudicated Claude F2 producer-completeness gap.

## New prospective scientific contract

Under the same bounded five-op branch

```
X[m,k] @ W0[k,n] --> H[m,n]
H[m,n] @ W1[n,q] --> A[m,q]
H[m,n] @ W2[n,q] --> B[m,q]
A + B --> G
ReLU(G) --> Y
```

the *runtime producer source* must emit, adjacent to
its actual arithmetic loop, for each `H[row,col]`:

- One `CONTRIBUTE` for every distinct `kk in [0,k)`,
  with `input_index=row*k+kk` and
  `weight_index=kk*n+col`.
- One separate `PROD COMMIT` only **after**
  every contribution for that exact H output.
- A corresponding typed `H COMMIT` immediately
  after the real H scratch write.
- One `PROD VALUE` witness containing the actual
  computed H value immediately following its
  typed H COMMIT.

The independent source-event oracle reconstructs
the expected population and operand coordinates
from MatMul mathematical dimensions. It rejects
early/duplicate/missing commits, missing/duplicate
contributions, premature or forged source indices,
missing or nonfinite values and inconsistent
H COMMIT witness order.

The separate GCC validator compares the actual
produced **intermediate H numerical values** directly
against a NumPy `X@W0` reference, and the final
model output against the independent Functional
Model. The read-only audit uses a separate
explicit scalar-sum H reference as another check.

A printed `CONTRIBUTE` by itself does **NOT**
prove that arbitrary C++ performed the indicated
multiplication; numerical H and final output
checks are distinct obligations. Test coverage
deliberately corrupts a real multiply while
leaving the contribution witness unchanged,
and corrupts a numerical H witness while
leaving the real final output unchanged.

## New implementation files

| Component | Role |
|---|---|
| `validation/r2e_independent_producer_oracle.py` | Independently reconstructs H producer contribution completeness, input/weight positions, commit readiness and runtime H-value cardinality from saved stdout |
| `codegen/hls/r2e_producer_traced_branch.py` | Separate R2d-based source variant that adds dynamic producer-term and intermediate-value witnesses adjacent to actual C++ source arithmetic and scratch commits |
| `validation/r2e_producer_cpp.py` | Compiles/runs new original GCC source; combines independent producer oracle, reader Phi oracle, branch reduction/alias replay and two independent numeric checks |
| `experiments/cst_r2e_producer_campaign.py` | New source-pinned create-only 12-case GCC execution campaign; saves C++/input/trace/case receipts and any failure |
| `experiments/cst_r2e_producer_audit.py` | Read-only original artifact SHA, H producer scalar-sum, Phi/consumer and final NumPy source re-audit; no new GCC/vendor calls |
| `tests/test_cst_r2e_producer_completeness.py` | Early/duplicate/missing/wrong producer contributions, missing typed commit, wrong actual C++ arithmetic, wrong H value and source operand trace mutations |
| `tests/test_cst_r2e_producer_campaign.py` | Full 12-case source evidence, read-only audit and tamper/summary/FAILED case controls |

**The previous R2b/R2c/R2d generator, checker and
original experimental evidence remain unchanged.**
This new work introduces a separate, explicitly
bounded producer-observation stream; it does
not silently extend the semantics of all older
source certificates or reinterpret them as
having independently verified producer arithmetic.

## Prospective original experiment matrix

| Property | Frozen plan |
|---|---|
| Functional shapes `(m,k,n,q)` | `(2,2,3,4)`, `(2,3,5,4)`, `(3,3,4,4)` |
| Reader mapping | scalar, reader0 PE2, reader1 PE2, both PE2 |
| Producer mapping | scalar only (source prototype limitation) |
| Issue policy | R2d's fixed loop-driven reader PE-group issue rule |
| Scratch | Existing whole-tensor last-reader reuse offsets |
| Independent checks | Producer `k`-term completeness; H values; reader issue; A/B and G reduction readiness; scratch alias safety; final NumPy |
| Original GCC | **12 new original compilations/executions**, bounded and create-only |
| Vitis/Vivado | Zero new calls |
| Failure retention | `FAILED.json`, original source, input and original case ID retained |

The 12 cases are **not** evidence from 12 independent
trained networks, arbitrary graphs, different producer
architectures, or physical RTL timing. They are fixed
small source examples selected before execution.
Unexpected failures and negative outcomes must
remain in the record; do not silently drop them
from the intended denominator.

## Workstation procedure

Run in the current **clean tracked** checkout;
the new original output folders must be unused.
Tests may create transient GCC executions within
pytest; they are NOT original source-pinned
paper-case receipts.

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests

python -m pytest -q \
  tests/test_cst_r2e_producer_completeness.py \
  tests/test_cst_r2e_producer_campaign.py

python -m pytest -q

# Only after the new regression passes.
# NEW original create-only R2e experiment:
python -m fpgai_research.experiments.cst_r2e_producer_campaign \
  --output build/cst_r2e_original_producer_gcc_v1

# Independent READ-ONLY source/numerical review, no GCC/vendor reruns:
python -m fpgai_research.experiments.cst_r2e_producer_audit \
  --study build/cst_r2e_original_producer_gcc_v1 \
  --output build/cst_r2e_readonly_producer_audit_v1
```

If any new R2e regression fails, do **not**
start the new original 12-case experiment.
If an original case fails, preserve its
source, trace and `FAILED.json`, then
fix source code in a *new revision* and
choose a distinct new original output
directory. Do not rerun or overwrite
`build/cst_r2d_original_runtime_loops_v1`,
`build/cst_r2d_runtime_independent_audit_v1`,
or any older original cohort.

## Remaining scientific limitations and priorities

Even if R2e fully qualifies, it establishes a
bounded **source-level reduction and numerical**
property, not bit-exact `ap_fixed` verification,
arbitrary C++ correctness, independent HLS/RTL
scheduling, FPGA clock-cycle source event order,
PE concurrency, BRAM/port arbitration,
power/performance or general multi-clock system
semantics.

Next steps: a typed generic producer
`CONTRIBUTE`/consumer `READ`/release
certificate across more than one operator
family and fan-out 2–4; independent
operator-level source/RTL consistency, exact
physical memory compatibility, and
source-pinned comparisons with established
scheduling/storage and compiler work.

**Publication status remains NOT READY,
and external scientific firstness is UNKNOWN.**
