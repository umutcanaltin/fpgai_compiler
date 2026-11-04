# R2c — Exact source-value lifetime allocation: scientific study contract

**Research compiler only, November 3, 2026.** This file specifies a
NEW prospective source and GCC campaign. **At the time of this commit,
the new tests and campaign have NOT yet executed on the user's
workstation.** No original R2b/GCC, historical native HLS, v6 routed
Vivado, frozen RL or Claude review artifacts are modified.

## Motivation and precise research hypothesis

The prior source-pinned R2b study independently established the
bounded Phi-aware reader-order effect: from 24 original GCC cases,
18 layout-scoped contrasts represented 9 distinct graph/mapping
interventions; **6/9** changed actual H-only final READ
rank/consumer, **3/9** changed final consumer identity and
**3/9** did not change the H-only final consumer/read rank.
However, **0 mapping-conditioned source capacity savings** were
established in the old whole-tensor offset scheme.

The new falsifiable R2c hypothesis is **not** that interval coloring
is a novel algorithm (it is standard), or that source-time is hardware
time. It is:

> For a fixed compiler-generated total source event order, typed
> tensor-value COMMIT/last-READ semantics enable a code-generated
> **per-element** scratch address map that safely aliases dead values
> while preserving the original graph's numerical result; the
> minimum required unit scratch slots can be exactly certified by
> peak simultaneously live values and independently checked on
> small examples using exhaustive enumeration.

The research question is whether this **operational connection**
between independently mapped reader Phi, value lifetime certificate,
memory assignment and actual emitted source addresses yields a
meaningful legal storage decision beyond the old whole-tensor baseline,
not whether an off-the-shelf allocator can color intervals.

## New research components

- `model_ir/element_scratch_allocation.py`:
  exact inclusive [COMMIT, final READ] value lifetimes,
  deterministic earliest-release min-heap unit-slot allocation,
  full event-by-event address replay, independent max-live
  lower bound and separate bounded exhaustive k-coloring oracle.
- `validation/r2c_independent_slot_oracle.py`:
  independent raw-event liveness extraction (without importing
  the allocator's interval construction) and exhaustive
  small-case slot-coloring oracle. The oracle is intentionally
  bounded and cannot be used as a scalability claim.
- `codegen/hls/r2c_allocated_branch.py`:
  real bounded C++ `scratch[slot]` stores/loads for
  each tensor element, all consumer Phi READ source orders,
  identical source-index/address variables in adjacent
  `FPGAI_SLOT` runtime witnesses, Add/ReLU arithmetic.
- `validation/r2c_allocated_cpp.py`:
  original GCC compile/run, strict runtime dynamic slot
  + READ/COMMIT replay and independent NumPy graph
  numerical equivalence at fixed `atol=1e-5`,
  `rtol=1e-4`. Non-numeric/alias/trace mismatches
  fail closed. No vendor calls.
- `experiments/cst_r2c_allocation_campaign.py`:
  NEW 12 original matched source pairs (3 shapes ×
  4 reader Phi policies), each **two real GCC
  source executions**: strongest existing
  whole-tensor last-reader-reuse baseline versus
  new independently certified per-element allocation.
  Save the actual original source and stdout SHA,
  input, model/Phi provenance, declared mutable source arrays,
  original failures and all case identities.
- `experiments/cst_r2c_allocation_audit.py`:
  read-only independent reconstruction of all 12
  pairs from saved evidence, including NumPy
  outputs, actual GCC slot/typed READ witnesses,
  complete value address replay, peak-live
  optimality, all original source temporary arrays
  and all summary denominators.
- `tests/test_cst_r2c_element_allocation.py`,
  `tests/test_cst_r2c_allocated_cpp.py`,
  `tests/test_cst_r2c_allocation_campaign.py`:
  real source, minimality, exhaustive-oracle,
  early overwrite, missing final reader, wrong
  source numeric output, malformed slot witness,
  altered CSV, failed original case and create-only
  controls.

The emitted source stores a 1D scratch pool of
`N` scalar data_t slots. It additionally allocates
**four arrays**, each of size `m*q`:
`acc_a`, `acc_b`, `add_a`, `add_b`.
All comparison rows report BOTH
`scratch_unit_slots` and
`scratch_unit_slots + 4*m*q` total **declared mutable source array elements** (output
and read-only weights excluded). A smaller scratch array is NOT automatically a smaller
total C++ temporary footprint, FPGA storage implementation or RAM cell
count: source scalar locals are not counted, and native synthesis
may optimize all these arrays differently.

## Prespecified complete original experiment population

| Dimension | Fixed value |
|---|---|
| Functional model | Three static MatMuls: `X@W0=H; H@W1=A; H@W2=B; A+B=G; Relu(G)` |
| Shapes `(m,k,n,q)` | `(2,2,3,4)`, `(2,3,5,4)`, `(3,3,4,4)` |
| Reader Phi policies | `scalar`, `reader0_pes2`, `reader1_pes2`, `both_pes2` |
| Source event scheduler | One fixed, unchanged prospective R2b PE-group/alternating reader order |
| Control per mapping | Whole-tensor last-reader-reuse scratch layout, original GCC |
| Treatment per mapping | Exact per-value source scratch assignment, original GCC |
| Original source compilation/execution budget | 24 GCC compilations/executions, ZERO Vitis/Vivado |
| Scientific unit for Phi-dependent storage | **9 unique graph/Phi interventions**, not 12 compiler pairs or duplicated control layouts |
| Fixed numeric payload | Original deterministic input per graph, same per-case weights |
| Negative controls | Invalid overlap, missing final READ, stale/forged slot witness, changed source numerical output |

The allocator has a closed-form optimum **only for a fixed
total source order with equal-sized unit scratch values**:
the peak count of simultaneously live source values is a
lower bound and unit-interval graph coloring can meet it.
The extra `validation/r2c_independent_slot_oracle.py`
independently reconstructs raw event live ranges,
without importing the production extractor; its
exhaustive backtracking solver is used ONLY on
small bounded graph cases. This reduces shared
implementation risk but does not prove any
unbounded source/hardware theorem. This does
not solve general bank-constrained memory allocation,
heterogeneous tensor sizes, temporal HLS cycles,
streaming arbitration or the joint schedule/memory
search problem.

**Pair comparison:** same functional/model Phi/
numeric policy/actual source event order, comparing
the fully disjoint/last-reader reuse baseline (we use
the stronger *last-reader reuse* control), versus
per-element allocation. Report reductions, ties
AND regressions in all 12 cases. The exact
`source_scratch_slots_saved` can be positive,
zero or negative; never force a positive claim.
For causal effects of Phi itself compare the
optimized capacity and source address table
across each of the nine non-scalar mappings
against scalar for the **same graph**, and
distinguish different address maps from different
slot counts. A changed map SHA alone is not a
mapping-dependent memory capacity improvement.

## Workstation verification and new execution

Run in the tracked clean `fpgai_research` checkout,
after the new regression tests pass. Claude's new
review Markdown is a separate artifact and should
not affect the compiler or historical evidence.

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests

python -m pytest -q \
  tests/test_cst_r2c_element_allocation.py \
  tests/test_cst_r2c_allocated_cpp.py \
  tests/test_cst_r2c_allocation_campaign.py
python -m pytest -q

# NEW create-only original source experiment; do not reuse an old path:
python -m fpgai_research.experiments.cst_r2c_allocation_campaign \
  --output build/cst_r2c_original_matched_gcc_v1

# Independently recheck all original cases; no GCC/vendor executions:
python -m fpgai_research.experiments.cst_r2c_allocation_audit \
  --study build/cst_r2c_original_matched_gcc_v1 \
  --output build/cst_r2c_independent_allocation_audit_v1
```

Both output paths MUST be previously unused. The original
campaign saves every baseline and treatment source,
input, original stdout, verification receipt and
failure evidence under its own case directory.
**Never rerun historical R2b or v6 original native
cases.** A failed new original R2c case remains
in its source-pinned folder with `FAILED.json`;
do not delete or overwrite it. Fix the new code at
a new commit and create a different future
experiment folder instead.

## Publication gates

R2c remains an **implemented but unexecuted,
unverified prospective research contribution
candidate** until the user's original source-pinned
GCC run and read-only auditor pass. After they
pass, allowed claims are limited to:

- observed real source-level per-element address
  safety/numerical equality **for the exact 12 cases**;
- exact source-unit-slot optimality under a
  fixed serialized event order, and separately
  oracle-verified very small cases;
- full declared mutable source array counts and
  matched control differences with negative cases;
- observed Phi-dependent address/capacity
  differences, only where numeric and exact
  counts support them.

The study **cannot** establish physical BRAM
savings, FPGA cycle ordering, RAM port feasibility,
achieved Fmax, general arbitrary-DAG support, an
improved trained RL policy or scientific firstness.
Independent code/prior-art review remains essential.
Published systems already investigate scheduling,
memory reuse, lifetimes, SDF/dataflow and polyhedral
mapping; any firstness claim requires precise
source-pinned external comparable feature parity.
