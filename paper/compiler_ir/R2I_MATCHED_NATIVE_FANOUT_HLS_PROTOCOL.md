# R2i — Matched native Vitis HLS source-memory experiment

**Prepared:** 2026-11-03. **Status: code committed, NO R2i native run verified.**
Research repository `fpgai_research`; RL remains untouched.
No original R2b–R2h, NCHW HLS or Vivado study is rerun or overwritten.

## Original scientific evidence at the handoff

At **original source SHA**
`efc6497cdab2ec5734a0c3750972655b9ea397b9`
the user's workstation verified **13 focused and 1,388 full Python tests**,
the original `R2h` **36 matched pairs / 72 real GCC executions**,
and **36/36** read-only independently audited source-memory pairs.
The actual emitted C++ used distinct **disjoint versus unit-optimal
scratch arrays**, with a runtime address witness for every typed
source load/store and all intermediate/final NumPy values checked.

- Source slot reuse saved scalar scratch-array elements
  in **36/36** matched disjoint-versus-optimized GCC pairs.
- Comparing optimized source capacity against the same graph's
  scalar mapping, **18/27 non-scalar Phi interventions reduced
  unit-slot count**, **9/27 tied**, **0/27 regressed**.
- The independent read-only auditor rechecked all **72/72
  original source/trace records** without new GCC/HLS/Vivado calls.
- Mutable array accounting also includes per-reader accumulators and
  reduction counters; C++ scalar locals are excluded.
- The completed historical R2h original study and SHA are immutable.

**Important:** Those are source-level scalar scratch-array counts,
not FPGA BRAM primitives, timing closure, achieved frequency,
hardware cycle liveness or RAM port feasibility.

## Why this new experiment is necessary

CST's source-level use of an E0-certified lifetime map
cannot establish an FPGA area improvement by itself:
Vitis HLS may map small scratch arrays to registers/LUTs,
duplicate memory to supply extra ports, remove dead arrays,
change clock/II, fail C simulation, or synthesize neither
arm. Earlier native NCHW physical data already showed
non-monotone results (BRAM saving in one of six pairs,
ties in four, a regression in one), as well as
**zero strict clean-DRC E4 pairs**. Do not assume
R2h source-space reductions imply better native QoR.

The main falsifiable question:

> When exactly matched, independently source-verified
> disjoint and lifetime-certified per-element C++ arrays are
> submitted to Vitis HLS 2023.2 at the **same** part,
> clock, uncertainty, numerical inputs and fixed-point
> precision, does either source arm pass real CSim/CSynth,
> and how do estimated BRAM_18K, LUT, FF, DSP,
> worst-case latency and timing margin compare?

This is **E2/E3 source-attributed Vitis evidence**.
A reported HLS BRAM difference is *not* physical
placement-and-routing BRAM savings or routed Fmax.

## Fixed, prospective population

All 36 original graph/mapping pairs from R2h are
represented by two source-memory arms each:

| Axis | Prespecified values |
|---|---|
| MatMul reader fan-out | 2, 3, 4 |
| Graph dimensions `(m,k,n,q)` | (2,2,3,4), (2,3,5,4), (3,3,4,4) |
| Reader Phi | all scalar, reader0 PE2, final reader PE2, all PE2 |
| Native source arms | `disjoint`, `per_element_optimal` |
| Native source packages | **72** packages in **36** exact matched pairs |
| Device | `xck26-sfvc784-2LV-c` (KV260, native fixed device) |
| Target period | 32.0 ns |
| Explicit uncertainty | 8.64 ns |
| Fixed-point | `ap_fixed<16,6,AP_RND,AP_SAT>` |
| CSim numerical tolerance | 0.1 absolute (predeclared; report exact error and failures) |
| Native run | Explicit opt-in, one complete matched pair at a time |
| Source-only preparation | 0 vendor calls |
| Physical Vivado runs | Not part of R2i |
| Missing/native failure status | `UNKNOWN/REJECTED`, never zero savings |

The two source arms share **exact same** FunctionalModel,
effective ArchitectureModel, numerical input, generated
C simulation testbench SHA and exact Tcl SHA.
The model source SHA and source scalar scratch-array
counts **must differ**; optimized scratch is smaller.
The generated source is not silently rewritten
by a vendor pragma in this primary paired protocol.
Vitis is allowed to reject either or both arms,
and this is a valid negative scientific result.

The geometry is small. Even 36 valid HLS pairs
may report all-zero or tied estimated BRAM values.
A later, separately preregistered medium/large
resource study is needed before broad hardware claims.
No case may be removed post hoc because it does not
improve the preferred metric.

## Reuse existing audited toolchain infrastructure

The R2i preparer calls the canonical
`write_vitis_package`, including the original
`make_testbench` and `make_vitis_tcl`.
The native executor calls canonical `run_vitis`,
`summarize_case` and
`cst_nchw_hls_qualification.qualify_case`,
which enforce source/Tcl/testbench lineage,
original vendor identity, actual CSim metrics,
CSynth XML resource availability and
uncertainty-adjusted scheduling feasibility.
Physical qualification is always false
at this stage; no report is promoted just
because a process returned zero.

New modules:
- `experiments/cst_r2i_fanout_native_prepare.py` —
  prepare 72 create-only matched source, testbench,
  Tcl and case manifest packages, plus independent
  SHA/case-denominator audit; **no Vitis calls**.
- `experiments/cst_r2i_fanout_native_execute.py` —
  dry run by default; requires both explicit
  `--execute-native` and
  `--confirm-isolated-native` to invoke
  the native Vitis runner. A read-only
  `--audit-native` reports every case/pair,
  including missing/failed results, with all
  estimates null until independently qualified.
- `tests/test_cst_r2i_fanout_native.py` —
  all 72 complete packages and 36 matched
  source/TB/Tcl pairs, invalid inputs,
  byte-level tampering and refusal to
  execute a vendor tool without confirmation.

## Workstation procedure

Always run from a clean tracked checkout. Preparation creates
a brand new `build/r2i_fanout_native_36_pairs_v1`
directory; it must not already exist.

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main &&
python -m compileall -q src/fpgai_research tests &&
python -m pytest -q tests/test_cst_r2i_fanout_native.py &&
python -m pytest -q &&
python -m fpgai_research.experiments.cst_r2i_fanout_native_prepare \
  --output build/r2i_fanout_native_36_pairs_v1
```

The first independent audit makes **NO Vitis calls**:

```bash
python -m fpgai_research.experiments.cst_r2i_fanout_native_prepare \
  --audit build/r2i_fanout_native_36_pairs_v1

python -m fpgai_research.experiments.cst_r2i_fanout_native_execute \
  --study build/r2i_fanout_native_36_pairs_v1 \
  --audit-native
```

The initial native pilot is **one complete exact pair**;
it is an irreversible original source-attested vendor run:

```bash
python -m fpgai_research.experiments.cst_r2i_fanout_native_execute \
  --study build/r2i_fanout_native_36_pairs_v1 \
  --start-pair 0 --max-pairs 1 \
  --execute-native --confirm-isolated-native \
  --timeout-seconds 7200
```

Review the actual Vitis result and audit before proceeding
to any next pair. The 36-pair population must never be
redeclared as `36 original native pass cases` after
running only one. An original invocation failure is retained
as a failed vendor receipt (including stdout/stderr),
not silently discarded.

The post-invocation **read-only audit** is:

```bash
python -m fpgai_research.experiments.cst_r2i_fanout_native_execute \
  --study build/r2i_fanout_native_36_pairs_v1 \
  --audit-native
```

Do not rerun pair 0 in its original package.
If code, source or numerical formulation needs correction,
commit a new revision, create a new and clearly distinct
original native study folder and preserve the negative
original pilot.

## Strict evidence classifications

| Stage | Required evidence | Claims still forbidden |
|---|---|---|
| R2i source preparation | 72 stable source/TB/Tcl SHA-matched files | CSim/CSynth or hardware benefits |
| Original native CSim (E2) | Matched source-attested vendor identity, CSim log with exact samples and numeric result | Native resources, QoR and Fmax |
| Original native CSynth (E3) | E2 plus attested original report, complete resources, bounded latency, feasible uncertainty-adjusted scheduling | Physically routed resources/Fmax |
| Original routed Vivado (E4) | **Separate** prior-defined strict source/RTL/report/DRC/timing provenance, zero prohibited DRCs | Board deployed bitstream or measured power |
| Peer comparison | Reproducible pinned implementation of existing compiler, equivalent numeric/device/toolchain constraints | Firstness or universal superiority without evidence |

All 36 source pairs must appear in tables,
even when the result is `UNKNOWN`,
`FAILED_ORIGINAL_CSIM`, incomplete HLS report,
resource regression or a Vitis timeout.
**No scientific superiority claim follows
from having more source-array slots saved.**

The publication gate remains **NOT READY** until actual
source-attested native measurements, strict routed E4
for meaningful pairs and valid external comparisons
are available. This protocol deliberately avoids
claiming results that have not run.
