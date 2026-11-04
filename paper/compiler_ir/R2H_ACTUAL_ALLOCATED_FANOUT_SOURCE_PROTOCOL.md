# R2h — matched executable fan-out source memory allocation

**Prospective, source-implemented but NOT YET run on the user's workstation.**
November 3, 2026. Research repository only. Historical R2f and R2g
original evidence and old native/physical campaigns are immutable.

## Verified original evidence we begin with

The user supplied original source SHA
\`0a92f4fa0630d81c51bb53ad04bb42304a67d2e5\`
from their workstation:

- **18/18 new focused tests**, **1375/1375 full pytest tests** passed.
- **36/36 original real GCC programs** compiled and executed for
  fan-out 2–4, three fixed shapes and four reader mappings.
- **36/36 independent read-only source/trace audits** passed,
  including source SHA, mapped runtime reader issues, full producer/
  consumer/Add readiness and intermediate/final numeric values.
- No original Vitis/Vivado calls.
- All original R2g C++ used **disjoint scratch**.
  Therefore prior R2f's 18/27 logical Phi-induced slot
  reductions were NOT yet source-executed or physical savings.

The original R2g output folders are immutable:
\`build/cst_r2g_original_fanout_36_gcc_v1\`
and \`build/cst_r2g_fanout_36_readonly_audit_v1\`.
R2h must not overwrite or reinterpret them.

## What R2h directly tests

The falsifiable new claim is:

> Given one fixed, independently checked source event order for
> 2–4 MatMul fan-out readers and a left-associated Add chain,
> an element-level unit-scratch assignment certified against every
> actual source last READ can be MATERIALIZED in runtime C++ loads
> and stores, with the same actual tensor-event order and all
> intermediate NumPy values as a disjoint allocation control.

The experiment differs from a mere E0 logical capacity statement:
both control and treatment C++ actually use \`scratch[address]\`
where \`address\` is derived from a static per-tensor lookup table
at runtime, and every source scratch READ/COMMIT prints the
SAME C++ address variable used by the actual operation.
The read-only auditor reconstructs all address references from the
actual original GCC stdout and verifies the event-by-event alias
proof independently against the fixed source schedule and input.

**This is not a new interval-coloring algorithm.** A certified
minimum unit-slot count for one totally ordered event sequence is
the standard equality between interval graph chromatic number
and peak live-value count. The potential CST contribution lies
in typed producer/reader/Add composition with independently
mapped reader Phi and source-executed correct memory reuse.
Scientific firstness still requires external artifact comparison.

## Exact new fixed cohort

| Dimension | Prospective value |
|---|---|
| Fan-out | 2, 3, 4 independent mapped MatMul readers |
| Shapes | (2,2,3,4), (2,3,5,4), (3,3,4,4) |
| Mapping | all scalar, first reader PE2, last reader PE2, all PE2 |
| Matched graph/Phi cases | 3 × 3 × 4 = **36** |
| Source-control arm | Disjoint slot-per-intermediate tensor, same C++ dynamic issue algorithm |
| Treatment arm | Certified \`per_element_optimal\` source scalar-slot map, same dynamic issue algorithm |
| Original GCC runs | **72** (2 per graph/Phi), no Vitis/Vivado |
| E1 validation | Complete producer/reader/Add READ/COMMIT traces, runtime addresses, independent Phi issue, all intermediate/final NumPy values |
| Allocation safety | Exact event-by-event fixed-source lifetime replay, independent of runtime slot lookup string |
| Source storage accounting | Scratch + reader accumulator array + reader count array (each R*m*q), plus scalar locals excluded |
| Physically measured hardware | **None** |
| Failure integrity | Source, manifest and any failed-case receipt retained, no deletion or silent denominator change |

The experiment reports two different contrasts:

1. **36 matched layout-arm pairs**: disjoint vs optimized under
   the SAME graph, Phi policy, input, source event order and operations.
   Report source scalar-slot savings, ties and regressions, and
   total declared mutable source array element savings.
2. **27 graph/Phi counterfactual edits**: optimized scalar
   vs one of three mapped-reader variants for the SAME graph.
   This estimates a fixed-source-order Phi effect only.
   It is not 27 independent real networks and does not imply
   achieved FPGA clock/resource advantage.

A source optimised array can still map to separate or duplicated
BRAM/register resources during HLS, or require unavailable
read/write ports. The C++ source uses a serialized total event
order. **No physical savings, timing, BRAM or II claim is allowed.**

## New source implementation

- \`codegen/hls/r2h_allocated_fanout_cpp.py\`:
  fail-closed emitter transformation of the separately
  checked R2g dynamic-loop algorithm, with two arm-specific
  constant address tables, same C++ dynamic indices, and
  inline \`FPGAI_SLOT\` records before every typed READ/COMMIT.
  A changed original source emission pattern refuses to compile.
- \`validation/r2h_allocated_fanout_cpp.py\`:
  original GCC compile/run, strict paired source slot/value event
  adjacency parser, independent source readiness/Phi check, separate
  event-by-event allocation safety and all intermediate NumPy.
  No reliance on a Python-generated event list as the only
  runtime correctness criterion.
- \`experiments/cst_r2h_matched_source_campaign.py\`:
  36 original matched pairs = **72 original GCC programs**;
  source manifest, exact input, runtime stdout and
  result receipt per arm, plus failed original case receipt.
- \`experiments/cst_r2h_matched_source_audit.py\`:
  read-only independent reconstruction of every original
  model, mapping, source/trace hash, dynamic address sequence,
  numerical tensor, capacity pair and all 27 Phi contrasts.
  No new GCC/Vitis/Vivado invocations.
- \`tests/test_cst_r2h_allocated_fanout_cpp.py\`:
  real GCC paired source tests; malformed memory-address
  witness, absent witness, mutated MatMul arithmetic,
  source SHA and unsupported layout controls.
- \`tests/test_cst_r2h_matched_source_campaign.py\`:
  all 72 original source programs, original record audit,
  mutation of saved trace, savings and missing original case.

## Commands — run ONLY after new tests pass

The new two output directories must not already exist. Do not
rerun any previous R2f/R2g originals.

\`\`\`bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main &&
python -m compileall -q src/fpgai_research tests &&
python -m pytest -q \
  tests/test_cst_r2h_allocated_fanout_cpp.py \
  tests/test_cst_r2h_matched_source_campaign.py &&
python -m pytest -q

# Only after both pytest invocations pass:
python -m fpgai_research.experiments.cst_r2h_matched_source_campaign \
  --output build/cst_r2h_original_36_matched_72_gcc_v1

# Read-only original source-slot and numerical audit:
python -m fpgai_research.experiments.cst_r2h_matched_source_audit \
  --study build/cst_r2h_original_36_matched_72_gcc_v1 \
  --output build/cst_r2h_independent_source_address_audit_v1
\`\`\`

If a new test fails, do not launch the new original study.
If an original run fails, preserve \`FAILED.json\`, source and
original version. Fix at a new commit and use a fresh output
directory. Existing original R2f/R2g/older studies remain frozen.

## Next actual paper gates: native physical experiments and prior-art parity

A successful R2h original E1 closes only the **source-memory
materialization gap**. The next meaningful paper evidence must
include source-pinned paired native Vitis CSim/CSynth/RTL/physical
Vivado on a real target device, comparable fixed-point precision,
real RAM-port legality, resource use, latency and Fmax. The
report must track clean DRC and negative/timeouts; synthesize both
source-layout arms with identical timing/constraints and share
original audit artifacts. Native vendor tools have not been run
by R2h.

External peer comparison is another independent requirement:
Thies et al. (PLDI 2001), Strout et al. (ASPLOS 1998),
SMO (POPL 2016), plus relevant current FPGA compiler systems.
Without equivalent feature parity and original artifacts,
scientific firstness and paper submission readiness are UNKNOWN/NO.

Frozen RL remains unchanged. The compiler is not an arbitrary
operator/ONNX graph backend yet, and no RL-trained improvement is
part of the current source paper.
