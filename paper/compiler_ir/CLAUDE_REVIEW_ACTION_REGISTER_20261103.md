# Independent Claude review — adjudication, remediation, evidence and follow-ups

**Date:** 2026-11-03. Research-only `fpgai_research`.
**Review basis:** Claude read-only document `SCIENTIFIC_REVIEW_README.md`,
reviewed commit `578178890607fa085ef0c30a9a270e123108a3bb`.
That was **before** the original R2c source/slot study and
before the code changes recorded in this document.

This is **not** a replacement for Claude's review. It is a
code-grounded response identifying accepted criticisms,
partly resolved issues, disputed interpretations, and the
next falsifiable research gates. Historical user workstation
original evidence remains source-pinned and MUST NOT
be rewritten, rerun, or retroactively claimed as passing
new checks. New changes in this register have **not yet
passed the user's local regression suite**.

## 1. Original R2c evidence newly available after Claude's review

The user supplied the complete original `e6db1ec5458a3d6e335b8eed665945c033a8b868`
workstation output:

- `compileall` completed, **21 focused tests passed**,
  full Python regression **1279 tests passed**.
- New `build/cst_r2c_original_matched_gcc_v1` original
  campaign: **12/12 matched original source-GCC pairs**,
  **24/24 original GCC compilations/executions**,
  12 matched numerics/typed trace and dynamic slot
  witnesses accepted. No new Vitis/Vivado calls.
- The separate create-only
  `build/cst_r2c_independent_allocation_audit_v1`
  accepted all 12 pairs and 24 original case sources,
  independently reparsed source-runtime address witnesses,
  recomputed the NumPy results and verified source-local
  peak-live/certified unit-slot minima.
- 12/12 pairs reduced scalar-unit scratch slots relative
  to the **whole-tensor last-reader reuse baseline**,
  0 ties and 0 regressions in the exact 12-case cohort.
  All 9 non-scalar (graph,Phi) interventions changed
  the minimum scalar scratch unit-slot count, not just
  the assignment SHA.

Original full 12-pair source results, count of scalar scratch
slots per compiled program:

| shape (m,k,n,q) | old tensor reuse | scalar new | reader0 PE2 | reader1 PE2 | both PE2 |
|---|---:|---:|---:|---:|---:|
| (2,2,3,4) | 22 | 17 | 16 | 16 | 16 |
| (2,3,5,4) | 26 | 19 | 18 | 18 | 17 |
| (3,3,4,4) | 36 | 26 | 25 | 25 | 24 |

Each C++ treatment also declares `acc_a`,
`acc_b`, `add_a`, `add_b` arrays of
`m*q` elements each; the stored audit correctly
counts scratch **PLUS these four mutable arrays**,
excluding C++ scalar locals, boundary inputs/outputs
and read-only weights. This is NOT measured FPGA BRAM,
hardware-clock lifetime or actual RAM port feasibility.

**Do not confuse allocation models.** Claude's
pre-R2c predictions 18/17, 21/20, 27/26 refer
to **contiguous whole-tensor base** exact allocation.
The new original R2c numbers refer to a more permissive
**independent per-element scalar-slot** allocation,
therefore cannot be compared directly as a replication
of Claude's predicted numeric optimum.

**Independent E1-property warning:** R2c's emitted
C++ uses static literal-derived addresses/indices.
GCC genuinely executes stores/loads and produces
numerical results, but the *ordering* follows the
generator's prewritten straight-line trace. Call
the order evidence **E0-source-schedule replayed
by GCC**, while E1 covers actual numeric execution
and checked address-witness consistency. A genuinely
independent loop-structured runtime ordering
experiment is still required (review finding F1).

## 2. Source-audited response to Claude findings

| Claude ID | Status after current code changes | Scientific disposition |
|---|---|---|
| **F1 event replay** | **ACCEPTED, NOT FIXED**: R2b/R2c emit straight-line static source and compare with their own generated schedule | Implement independent loop-driven source accesses and instrument *calculated* runtime subscripts; cross-check against an independent mathematical schedule, not the generator |
| **F2 premature COMMIT** | **ACCEPTED, PATCHED, UNTESTED**: new branch verifier requires every mapped H contribution for A/B COMMIT, both Add operands before G COMMIT | New exact mutation tests for A/B/G; H producer reduction is **not present in the READ/COMMIT event language**, so its completeness is NOT independently proved |
| **F3 Phi-blind capacity** | **ACCEPTED, R2c RESOLVES A BOUNDED VERSION**: original 12/12 source scratch advantages and 9/9 Phi-caused per-element min-slot changes at three fixed shapes | Add independent contiguous-base-vs-unit-slot comparison on identical **Phi-conforming** interleaved schedules; a new optional source-events parameter is committed but untested |
| **F4 omitted accumulators** | **PARTIALLY REMEDIATED IN R2c**: five declared mutable C++ arrays counted | Do not call this complete C++ memory footprint, total physical BRAM, or verified synthesis resource allocation; scalar locals excluded |
| **F5 common-mode coordinate oracle** | **ACCEPTED, NEW INDEPENDENT REFERENCE COMMITTED, UNTESTED** | New closed-form PE-group/owner oracle does not import resolver.coordinate or generator; new deliberate common-mode mutation test |
| **F6 Phi issue conformance** | **ACCEPTED, PATCHED, UNTESTED**: separate mathematical source issue check required for R2b/R2c emission and GCC trace validation | General source legality can accept a different safe order; **Phi-policy conformance is a distinct obligation** |
| **F7 C3 UNKNOWN policy table** | **ACCEPTED, NOT YET FIXED** | The current mask cannot claim certificate-informed multi-reader Phi search, nor R5 trained RL. Move learned performance into separately evaluated work |
| **F8 brittle experimental expected outcomes** | **PARTIAL**: new prospective C2 mutation study reports all survived/killed mutants, rather than deleting surprising results | Historical original R2/R2b outputs are immutable; future original campaigns must preserve unexpected outcomes and use status receipts, not overwrite |
| **F9 bare asserts** | **PARTIAL PATCH, UNTESTED**: Euclidean partition certificate now uses an explicit error, active under `python -O` | Audit remaining runtime-critical source exceptions and separate them from experiment-local sanity assertions. Consolidate latest paper status |
| **CI / IP** | **UNRESOLVED** | No publishing, licence decision, CI permissions or repository public-release action is implied by this work |

## 3. New independent C2 certificate and mutation coverage

The newly committed research additions:
- `model_ir/independent_phi_issue_reference.py`: the
  independently specified source issue ordinal for each
  `(m,ni,kk)` in a PE group, including incomplete
  final PE groups, and an owner tuple derived by
  arithmetic without calling `resolver.coordinate`.
  It checks a **fixed source issue policy**, NOT
  hardware parallelism or concurrent PEs.
- `branch_matmul_contract.validate_branch_events`:
  rejects `PREMATURE_REDUCTION_COMMIT` for A/B
  before all required H contribution READs, and
  `PREMATURE_ADD_COMMIT` for G before both Add
  operand READs, with a concrete counterexample.
  Does not claim independent H producer completion.
- Independent issue gate integrated with the
  existing R2b certificate, R2b/R2c code emission
  and E1 GCC event validation.
- `experiments/cst_certificate_mutation_adequacy.py`:
  an original prospective 3 shapes × 4 Phi policies
  × **8 prespecified fault classes = 96** E0
  mutants: premature A/B/G commits, live alias,
  wrong owner, duplicated READ, missing READ,
  and safe-but-policy-nonconforming READ order.
  It stores **all survivors** and per-checker
  kill counts; NOT an E1 or hardware fault study.
- `branch_exact_allocator.exact_mapped_branch_slots`
  accepts a verified optional interleaved schedule
  for **contiguous tensor-base comparison**;
  its older generated-serial default is retained.
- Tests for premature COMMIT, independent owner
  common-mode errors, legal-but-nonconforming order,
  original serial exact allocation compatibility,
  incomplete PE-group issue and the full 96-mutant
  matrix.

**New code and tests are committed but not workstation
verified yet. Do not claim a new mutation kill rate
or regression pass before the new user receipts.**

## 4. Independent prior-art threat: schedule and storage

The following verified public research predates CST
and makes a universal `Phi → lifetime → allocation`
firstness claim untenable without narrower evidence:

1. William Thies, Frédéric Vivien, Jeffrey Sheldon
   and Saman Amarasinghe, *A Unified Framework for
   Schedule and Storage Optimization*, PLDI 2001,
   DOI [10.1145/378795.378852](https://doi.org/10.1145/378795.378852).
   Joint schedule/storage reasoning, occupancy vectors
   and affine dependencies.
2. Michelle Mills Strout, Larry Carter, Jeanne
   Ferrante and Beth Simon, *Schedule-Independent
   Storage Mapping for Loops*, ASPLOS 1998,
   DOI [10.1145/291006.291015](https://doi.org/10.1145/291006.291015).
   Reuse across legal schedule families.
3. Somashekaracharya G. Bhaskaracharya, Uday
   Bondhugula and Albert Cohen, *SMO: An Integrated
   Approach to Intra-Array and Inter-Array Storage
   Optimization*, POPL 2016. Schedule-dependent
   intra-/inter-array sharing.
4. Lefebvre/Feautrier and Darte/Schreiber/Villard
   exact/provable storage optimisation should
   be examined from full texts, not guessed from
   an abstract. Add artifact pointers where available.

**Candidate narrower paper contribution (unproven):**
a *compositional* per-operator mapping→fanout tensor
readiness/lifetime relation with independently checked
source/RTL correctness and quantitatively characterised
map-induced memory effects. Its distinctness from
published polyhedral/scheduling/dataflow systems is
NOT established by these four citations alone.
An externally pinned peer-parity study is still mandatory.

## 5. Ordered next gates

1. **Gate C2 correctness NOW**: run focused new
   mutations and complete pytest. Any exception,
   regression or survivor must be reported. Do not
   rerun old original R2/R2b/R2c vendor/GCC cohorts.
2. **Gate controlled allocator model comparison**:
   compare old tensor reuse, exact contiguous
   tensor-base and exact per-element unit-slot
   allocation under identical source schedule/Phi.
   Retain unknown/timeout cases; *do not* fill them
   with invented best values.
3. **Gate independent source schedule**: build
   loop-structured, runtime-subscript-instrumented
   R2b successor. Compare full order with closed-form
   oracle and deliberate generator wrong-order mutation.
   Keep E0-vs-E1 property attribution explicit.
4. **Gate C2 scientific adequacy**: augment E0
   96-mutant experiment with E1 source executions and
   other fault classes, including producer
   CONTRIBUTE-before-COMMIT and illegal hardware
   port use only when semantics actually represent them.
5. **Gate external parity and general graphs**:
   move beyond three fixed five-op MatMul diamonds
   to actual fan-out 2–4 graph families and real
   ONNX models, with peer comparison (e.g. Allo,
   HIDA, ST-Flow, Stream-HLS) at equivalent precision
   and event constraints. No hardware or novel
   scientific claims until those original results exist.
6. **Gate hardware**: source-pinned original
   Vitis CSim/CSynth then strict clean-DRC Vivado,
   independently prespecified DRC policy and
   replicated implementation variation only on
   hardware-relevant predicted differences.
7. **Gate C3**: independent RL paper; policy-neutral
   API is an engineering feature until training
   and held-out generalisation are measured.

## 6. New workstation commands — code and tests ONLY first

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests

python -m pytest -q \
  tests/test_cst_r2b_interleaved_schedule.py \
  tests/test_cst_independent_phi_issue.py \
  tests/test_cst_phi_schedule_exact_allocator.py \
  tests/test_cst_certificate_mutation_adequacy.py
python -m pytest -q

# ONLY after tests pass; this executes NO GCC/Vitis/Vivado.
# A fresh create-only original E0 mutation report:
python -m fpgai_research.experiments.cst_certificate_mutation_adequacy \
  --output build/cst_C2_e0_mutation_adequacy_v1
```

**Do not re-execute any historical 24-case R2b GCC,
12-pair R2c GCC or v6 routed Vivado results.**
Those original results remain tied to their
historical source revision. A fresh mutation
study must use a new unused output directory.

## 2026-11-03 — original 96-case C2 E0 mutation adequacy completed

At original source `82ef9f2c0fe101dadcf6a1c013540b8f3780abf6`,
the user ran the new E0 original create-only
`build/cst_C2_e0_mutation_adequacy_v1` study.
All **96 prespecified mutant cases** were rejected:
8 fault classes × 3 graph shapes × 4 Phi reader policies,
including 12 premature A, 12 premature B,
12 premature Add, 12 illegal live H aliases,
12 wrong owners, 12 duplicate H reads,
12 missing H reads, and 12 safe-but-
Phi-issue-nonconforming source orders.
The last class was intentionally accepted by
the generic branch safety certificate (12/12)
and rejected by the separate independent
Phi issue policy specification (12/12).
No original GCC/Vitis/Vivado calls occurred;
the observed source E0 result is **not** an
independent proof of all future mutation classes.

Focused new remediation suite 21/21 passed.
Full regression **1294 passed, one failed** due to
an expected-error-regex mismatch in
`test_mutated_runtime_trace_and_source_identity_are_rejected`:
the independent oracle rejected the tampered
GCC trace before a generated-event sequence equality
assertion could run. A test-only correction
was committed to the new revision afterwards;
the user has NOT yet supplied post-correction
workstation test results. Never rerun, delete
or overwrite the original 96-case folder.

Follow-up F1 R2d code added after the original
cohort: loop-driven C++ source computes actual
H READ subscripts from a runtime ordinal, while
a distinct Python closed-form Phi issue reference
checks executed GCC stdout independently of the
generator's precomputed event list. New
source/order/numeric mutants, 12 original
prospective cases and a read-only SHA/NumPy
audit are committed. The R2d code and
all new tests are **unexecuted on the
user's workstation as of this record**.
See `R2D_RUNTIME_LOOP_INDEPENDENT_ISSUE_PROTOCOL.md`.
No previous original R2b/R2c/Vivado experiment
must be rerun or reclassified as R2d.

## November 3, 2026 — R2d original execution complete; R2e implementation pending

The user's source-pinned original R2d source study at
`09ac395ef5c73cae59940c4761f3255d695882d1`
is **complete**: 23 focused tests, 1,310 full pytest
tests, 12 original GCC runtime-loop cases, and all
12 independently read-only audited source/trace/NumPy
cases passed. There were zero Vitis/Vivado calls.
The new runtime issue order is source-level, not a
measured FPGA cycle schedule.

A **NEW, prospective and unexecuted R2e implementation**
now responds to the missing H producer contribution
certification: full separate runtime
`CONTRIBUTE` index/source witnesses, producer
`COMMIT`, typed H source `COMMIT` and adjacent
actual numerical H value witnesses. An independent
oracle checks every reduction term and input/weight
address; a GCC validator compares actual intermediate
H and final Y against separate NumPy arithmetic
checks and checks reader Phi and scratch legality.

The next **12-case original R2e GCC cohort** and
its separate read-only SHA/NumPy audit are
implemented but **have NOT been executed on the
user's workstation**. R2e code/tests have not
passed workstation regression yet. This is a
NEW source experiment, not retroactive proof
about the completed historical R2d source files.

Correct interpretation: the R2e source trace is
instrumented next to real C++ reduction operations,
and H/final numerical output is independently
checked. This is NOT a formal proof of all
possible numerical inputs, bit-exact `ap_fixed`
arithmetic, arbitrary graph producers, FPGA
hardware clocks, PE parallelism, BRAM, RAM port
legality or external novelty.

**Run the new R2e focused and full pytest suite
before launching any new R2e original campaign.**
See [R2e full prospective protocol](R2E_PRODUCER_COMPLETENESS_PROTOCOL.md).
