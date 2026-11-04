# R2f — bounded compositional 2–4 reader source certificates

**Protocol registered 2026-11-03.** Research-only Model IR E0 study.
Implementation is committed but **new tests and experiment have not
yet been executed on the user's workstation**.

## Independent preceding evidence (preserved, not re-run)

At immutable source revision
`d41568c1f913e245692815f733a537799c6df7bd`,
the user's original R2e workstation run reported:

- Python tests: **20/20 focused, 1,330/1,330 complete**.
- Original R2e: **12/12 real GCC producer-contribution cases**.
- **312/312** independently counted producer contributions,
  12/12 intermediate H numerical checks, and 12/12
  final consumer/numerical checks.
- Independent read-only original artifact auditor: **12/12
  original GCC cases**, recomputed H scalar-sum values, NumPy
  final values, source/trace SHA and mapped reader correctness.
- New original native Vitis/Vivado invocations **zero**.

These results are for a **fixed two-reader five-operator
MatMul diamond**, so they do not establish generic graph
composition, streaming producer/reader execution, bit-exact
fixed point or independent hardware scheduling.

Never overwrite or rerun
`build/cst_r2e_original_producer_gcc_v1`
or `build/cst_r2e_readonly_producer_audit_v1`.

## Scientific hypothesis and novelty boundary

The next *falsifiable* question is whether one independent
source-level certificate calculus can handle **varying
fan-out degrees** with independently chosen reader
architectures, the producer's reduction contributions
and a multi-stage Add fan-in, preserving:

1. Every producer MatMul H output commits only after
   precisely its k input×weight contribution coordinates.
2. Every reader output commits only after each of
   its n mapped H reduction reads.
3. Every Add-chain value commits only after both
   operand reads (including an earlier Add result).
4. A value's physical logical scratch slot cannot
   be overwritten before all of its model-edge
   consumers have read it; a final read releases
   its value.
5. An independent Phi source-issue rule can be
   checked *separately* from general source
   readiness and safe aliasing.
6. A different legal producer/consumer interleaving
   can satisfy value readiness without being
   incorrectly rejected as globally serial.
7. For any fixed, total source event order,
   minimum equal-size unit scratch slots equal
   the maximum simultaneous live scalar values.
   **This interval-coloring fact is prior art**,
   not a claimed algorithmic invention.

The work explores bounded **compositional legality**,
not an arbitrary graph theorem or scientific firstness.
The mature schedule/storage prior-art gate is still open:
PLDI 2001 Thies et al., ASPLOS 1998 Strout et al.,
POPL 2016 SMO and other systems.

## Exact newly supported functional family

For R readers with R ∈ {2,3,4}:

```text
        X @ W0 --> H
                    |-- H @ W1 --> A0 ---+
                    |-- H @ W2 --> A1 ---+--> G0
                    |-- H @ W3 --> A2 --------> G1
                    |-- H @ W4 --> A3 ------------> G2
                                                     |
                                                 ReLU --> out
```

Only use the first R consumer branches and first R−1
left-associative Add operations. All readers
have shape `(m,q)` and independent constant
weights; producer H is `(m,n)`. The graph
is topologically represented by native
`FunctionalModel` and `ArchitecturalModel`
objects, with per-operator effective `pes`
and `simd`. Producer, Add and Relu source
mapping is scalar only; tiled, serialized,
pipelined and replicated source schedules
fail closed.

## Committed implementation and required checks

- `experiments/cst_fanout_models.py`:
  deterministic FunctionalModel families for
  R=2,3,4 with one producer and R readers.
- `model_ir/fanout_contract.py`:
  a separate immutable typed
  `FanoutEvent` (`CONTRIBUTE`,
  `COMMIT`, `READ`) and exact
  model/architecture admission rules.
  Source-readiness, producer/reader/Add
  completeness, per-model-edge identity,
  mapped reader owners, scratch overlap
  and final use are verified independently.
  It accepts legal producer/consumer
  interleavings where source readiness permits.
- A second, closed-form **forward Phi
  rank** reference checks issue order
  independently from the generator's
  runtime-style ordinal-to-coordinate decoder.
  This is a specific source-issue policy,
  not an HLS achieved schedule.
- `experiments/cst_r2f_fanout_campaign.py`:
  fixed 36-case source study and actual
  last-read/owner plus fixed-order
  unit-slot count; reports positive, tie
  and regression counts without assuming
  an improvement.
- `tests/test_cst_r2f_fanout_contract.py`
  and `tests/test_cst_r2f_fanout_campaign.py`:
  fanout 2/3/4 legal cases, early
  producer/reader/Add commit, wrong operand
  index, false owner, live-slot alias,
  safe-but-issue-nonconforming order,
  safe producer/reader interleaving,
  unsupported mapping and immutable
  create-only case denominators.

## Prespecified new E0 experiment

| Dimension | Declared values |
|---|---|
| Reader fan-out | 2, 3, 4 |
| Static `(m,k,n,q)` | (2,2,3,4), (2,3,5,4), (3,3,4,4) |
| Reader mappings per graph | all scalar; first PE2; last PE2; all PE2 |
| Unique cases | **3 × 3 × 4 = 36** |
| Non-scalar counterfactuals | **3 × 3 × 3 = 27** |
| Explicit source events | producer contribution and commit, all mapped H reads, reader commits, Add-chain operand reads and commits, last Relu reads |
| Readiness/alias policy | exact typed graph edges and per-element final READ |
| Logical source scratch metric | bounded unit-element fixed-order interval packing; actual C++ source not emitted |
| GCC / Vitis / Vivado | **0 / 0 / 0** |
| Evidence tier | **E0 only** |

The campaign records source-read order, H-only final
consumer/rank, an exact unit-slot bound for the
fixed event order, and the mapped differences.
All 27 Phi edits must appear in the denominator,
including ties and regressions. A changed event
SHA or value assignment is **not** automatically
a demonstrated capacity reduction.

The default event generator finishes the producer
before readers, as in R2e. The independent verifier
can also accept an **E0-only, alternate safe
producer/reader interleaving**; no C++ or native
implementation is claimed for that alternative.

### Workstation validation

First run the tests. If any fails, **do not**
start the create-only R2f original cohort.

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests

python -m pytest -q \
  tests/test_cst_r2f_fanout_contract.py \
  tests/test_cst_r2f_fanout_campaign.py

python -m pytest -q

# Only after both pytest invocations pass:
python -m fpgai_research.experiments.cst_r2f_fanout_campaign \
  --output build/cst_r2f_fanout_36_e0_v1
```

This must be an **unused output folder**.
Neither a successful R2f E0 experiment nor
a 36/36 test pass would demonstrate actual
source C++, GCC behavior, HLS dataflow,
hardware cycle-level release, BRAM capacity,
RAM banking/ports, power or Fmax.

**Next scientific gates:** introduce a
generic, typed per-edge source execution
backend; independently audit actual
fanout 2–4 GCC producer/read/commit traces
and NumPy intermediate/final outputs;
extend from pure MatMul to a second
independently checked operator family;
then native Vitis HLS and clean-DRC
Vivado matching with pinned peer baselines.

**Publication submission remains blocked.**
