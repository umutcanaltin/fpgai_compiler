# R2g — executable fan-out 2–4 source conformance

**Prepared November 3, 2026; prospective, NOT YET WORKSTATION VERIFIED.**
Research-only. No changes to frozen original GCC/HLS/Vivado or RL studies.

## Original R2f result that motivated this experiment

The user executed the original R2f source-event cohort at exact revision
`98676f7786687eddace0cab9124868448f9bdc24`:

- **27/27 focused R2f tests** and **1,357/1,357 full Python regression tests** passed.
- All **36/36 prespecified** E0 fan-out readiness, reduction, Add-chain
  and independent Phi issue certificates passed.
- Among the **27 unique non-scalar Phi interventions**, **18 reduced**
  fixed-order logical scratch-slot capacity, **9 tied** and **0 regressed**;
  18 also changed the final H reader rank.
- **Zero GCC, Vitis or Vivado** invocations in that original E0 study.
- The source-pinned original folder `build/cst_r2f_fanout_36_e0_v1`
  is **immutable**: never rerun or overwrite it.

This is **logical E0** evidence. The 18 reductions are NOT physical
BRAM, vendor HLS or actually optimized C++ source-memory reductions.

## Falsifiable new R2g scientific question

Does the new compiler source lowering preserve, for an entire
**2–4 reader** MatMul fan-out and Add chain, the E0 graph's:

1. Producer input/weight index and complete reduction-before-COMMIT.
2. Every mapped consumer READ identity, last-use and Phi issue rank/owner.
3. Every reader's complete reduction-before-COMMIT.
4. Every Add-chain edge's two READs and value readiness.
5. Each intermediate's NUMERICAL value, not merely final ReLU output.

**E1 qualification:** actual C++ runtime computes each reader's
`(row, col, kk)` subscript from an ordinal and its reader width.
The generated C++ executes real `scratch[index]` loads and stores,
reduction arithmetic and Add/Relu, then prints witnesses with runtime
indices and intermediate values. A second, separately implemented
Python mathematical **forward Phi rank** oracle checks stdout.
The experiment does **not** compare stdout to a Python-generated
event list as its sole schedule criterion.

The independent numerical reference evaluates every intermediate
tensor in the FunctionalModel with NumPy float32. The source proof
reconstructs actual producer/reader/Add events from raw GCC output,
checks typed reducer completion and disjoint source scratch safety,
and verifies the independent Phi policy.

The source reads/writes use the fixed disjoint baseline. **R2g
does not emit the 18 E0 source-optimal scalar-slot layouts.**
No link to physical memory savings is permitted until a later
source-level allocated backend and independently measured HLS/Vivado
study exist. This is an E1 source semantics experiment, not the
paper's final physical QoR experiment.

## Exact original cohort, implementation and failures

| Dimension | Prespecified contract |
|---|---|
| Fan-out | R = 2, 3, 4 MatMul readers |
| Shapes `(m,k,n,q)` | (2,2,3,4), (2,3,5,4), (3,3,4,4) |
| Reader Phi | all scalar; reader0 PE2; last reader PE2; all PE2 |
| Total | **3 × 3 × 4 = 36 original GCC source cases** |
| Source ordering | Runtime ordinal decoder, serial producer then mapped readers, left Add chain, final Relu |
| Original execution | GCC C++17, `-O0 -ffp-contract=off`, float32; NOT Vitis |
| Independent checks | Complete producer/reader/Add events; H-only issue and owners; all committed intermediate values; final NumPy |
| Source scratch | Disjoint only (NOT the E0 optimized scalar-slot layout) |
| Vitis/Vivado | Zero calls |
| Evidence | `source.cpp`, `case.json`, `original_stdout.txt`, `verified.json`, `cases.csv`, `summary.json`, failure `FAILED.json` |
| Experimental data policy | New/unused create-only directory, original failed cases remain immutable |

New files:

- `codegen/hls/r2g_fanout_cpp.py`: dynamic C++ ordinal decoder and actual MatMul/Add/Relu accesses for R=2–4, explicit event/value runtime witnesses.
- `validation/r2g_fanout_cpp.py`: original GCC compile/execute, strict dynamic event parser, independent Phi, alias/reduction and all-intermediate numerical checks.
- `experiments/cst_r2g_fanout_source_campaign.py`: source-pinned original 36-case GCC study.
- `experiments/cst_r2g_fanout_source_audit.py`: read-only original SHA/Phi/producer/numeric audit, zero new GCC/vendor calls.
- `tests/test_cst_r2g_fanout_cpp.py`: actual GCC examples plus corrupt runtime ordering, missing producer term, incorrect H/Add arithmetic, wrong numerical witness, early reader commit and source-SHA controls.
- `tests/test_cst_r2g_fanout_source_campaign.py`: all 36 original source cases, read-only audit, original-trace tampering, summary count inflation, missing/failed case and create-only controls.

## Workstation commands — sequential fail-closed

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main &&
python -m compileall -q src/fpgai_research tests &&
python -m pytest -q \
  tests/test_cst_r2g_fanout_cpp.py \
  tests/test_cst_r2g_fanout_source_campaign.py &&
python -m pytest -q

# ONLY after the preceding tests pass; NEW original output:
python -m fpgai_research.experiments.cst_r2g_fanout_source_campaign \
  --output build/cst_r2g_original_fanout_36_gcc_v1

# Independent read-only audit, NO new gcc or native vendor tools:
python -m fpgai_research.experiments.cst_r2g_fanout_source_audit \
  --study build/cst_r2g_original_fanout_36_gcc_v1 \
  --output build/cst_r2g_fanout_36_readonly_audit_v1
```

**Both directories must be unused.** A failed original R2g case must
retain its original source and `FAILED.json` rather than silently
resetting the corpus to 36 passing examples. Fix in a new commit and
use a **new** source-pinned directory. No older original studies
should be rerun. As of this protocol **the new tests and original
36-case GCC cohort are UNEXECUTED and unqualified**.

## Publication readiness — still not met

This new bounded source result (if verified) would strengthen
C1/C2 by extending independently checkable execution semantics to
multiple fan-out degrees. It would **not prove scientific firstness**,
a general DAG compiler, streaming hardware synchronization,
RAM bank/port legality, bit-exact `ap_fixed`, physical BRAM savings,
achieved Fmax, power, or RL training benefits.

The strongest remaining next experiments are:

1. Source-level **same-order** optimized per-element layouts with a
   numerical/control comparison against the disjoint R2g source,
   keeping all declared mutable arrays visible.
2. Original native Vitis CSim/CSynth and strict clean-DRC Vivado
   paired study for hardware-relevant layout/mapping choices;
   positive/negative and timeout cases all retained.
3. Pinned and reproducible external compiler parity against
   pre-existing schedule/storage and FPGA systems, with equivalence
   in shapes, numeric format, device, toolchain, constraints.
4. A second independent operator family beyond MatMul and actual
   graph fan-out + multi-stage partial scheduling.
5. Keep C3 policy/RL performance **unclaimed** until training and
   held-out-model experiments. The RL environment remains frozen.

The scientific paper may be drafted from bounded evidence,
but submission readiness remains **NO** until the
hardware/peer-parity/reproducibility gates are met.
