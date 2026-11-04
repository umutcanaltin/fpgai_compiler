# CST R2j — Strict source-pinned Vivado physical validation

**2026-11-03 — Implementation committed; physical Vivado runs NOT YET executed.**

## Why this gate matters

The completed original R2i experiment was run at immutable source
`b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`.
The original read-only native audit reported **72/72 real Vitis HLS
invocations and CSim qualified**, **48/72 E3 CSynth qualified**,
and **24/36 matched pairs both E3 qualified**. Among those
24, **6** have 1 fewer *HLS-estimated* BRAM_18K, **18** tie,
and **none** regress. **12/36 pairs have HLS
`latency_best = undef`** and must remain E2-only/unknown.
There are **0** clean-DRC routed physical measurements so far.

These are source-attributed HLS estimates. No physical
BRAM benefit, routed WNS/Fmax, RAM port or board bitstream
has been established. The 12 unqualified R2i pairs
cannot be silently excluded from the cohort denominator.

The immediate physical question is whether at least one of
the **six E3-qualified HLS memory-reduction pairs**
survives Vivado implementation under identical source
constraints and does so with independently verified
complete routing, nonnegative WNS, valid original RTL/ROM
hashes and **zero DRC violations**.

## Prospective physical design

- Use the **unchanged, original** native HLS study in
  `build/r2i_fanout_native_36_pairs_v1` after completion.
- The R2j **source-preparation bridge explicitly permits
  `require_current_source=False`**, since the immutable
  original R2i study is pinned at `b2d5dcc4` and
  the current GitHub `main` has later documentation/code.
  Its original native study, data and SHA are not rewritten.
- Independently read original R2i source/TB/Tcl manifests
  and native receipts. Copy only real, source-attested
  Vitis-generated Verilog and original .dat/.mem ROMs
  from independently E3-qualified cases into a new,
  separate, create-only physical study directory.
- Every one of the **72** original cases remains recorded
  in the physical study manifest, including the **24 E2-only
  cases** blocked from Vivado. All **36** matched pairs
  remain in physical results; unknown stays unknown.
- Reuse the existing OOC Vivado Tcl, native Vivado runner,
  Verilog/ROM input and run attestation, Vivado route/status/
  timing/utilization/DRC report parsers and independent
  read-only audit. Never alter DRC severity or waive
  unconstrained board I/O requirements.
- Target the original FPGA part
  `xck26-sfvc784-2LV-c` and 32ns requested clock
  in the first physical study. This is an OOC HLS module
  implementation on a target **part**, NOT a KV260
  board-pin-constrained, deployable bitstream.
- Original Vivado invocation and source/staged ROM
  identities are SHA-attested. Literal `$readmemh`
  post-run disappearance is accepted only under the
  existing narrowly defined original vendor-reported
  successful-read attestation, NEVER as unchanged ROM
  contents.
- **E4 strict qualified** requires complete routing,
  original report SHA attribution, timing WNS ≥ 0 ns,
  complete LUT/FF/DSP/BRAM resource metrics and a
  verified zero-violation DRC summary. A warning-only
  result is separate, descriptive
  **OOC routed observation**, not clean-DRC E4.
- Keep all regressions, report failures, missing RTL/IP
  formats, ROM issues, DRC warnings and timeouts.
  Do not substitute 0 for an unknown resource.
- The first paired physical pilot is pair
  **20: case_040 disjoint versus case_041 optimal**.
  Original native HLS estimated BRAM_18K **3 vs 2**,
  worst-case latency **1243 vs 1208** cycles.
  This selected pair tests whether an *already observed*
  HLS estimate survives placement/routing; it does
  not represent the full 36-pair cohort.

## New implementation

- `experiments/cst_r2j_fanout_physical.py`: 72-case
  source-pinned RTL/ROM package preparation, complete
  physical-cohort read-only audit, explicit opt-in
  bounded Vivado runner and complete 36-pair
  matched original routing export.
- `tests/test_cst_r2j_fanout_physical.py`: source-only
  and no-vendor guards for all 72 original cases and
  36 unknown physical pairs, create-only export,
  strict range/confirmation validation and manifest
  tamper detection.
- `experiments/cst_r2i_fanout_native_effects.py`:
  independent 36-pair native HLS effects export with
  conditional-on-E3 counts, 12 unknown preserved,
  and per-case source slots, resources and latency deltas.
- `tests/test_cst_r2i_fanout_native_effects.py`:
  no-vendor test fixture containing 72 cases, six
  BRAM reductions, 18 ties and 12 undefined-latency
  pairs, including falsified resource/denominator
  negative controls.

The R2j physical module reuses the previously tested
NCHW physical audit policies and vendor helpers,
with a separate namespace and fixed 72/36 R2i
fan-out denominator. Native Vitis and previous
historical physical studies are not modified.

## Workstation procedure

The R2i native original campaign is **already complete**.
It is now safe to update the local Git checkout with
the new R2j source. **Do not delete or rerun any
original R2i study package.**

Verify the new code first:

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main &&
python -m compileall -q src/fpgai_research tests &&
python -m pytest -q   tests/test_cst_r2i_fanout_native_effects.py   tests/test_cst_r2j_fanout_physical.py &&
python -m pytest -q
```

If any test fails, **stop**, preserve study inputs and
fix the implementation in GitHub before launching
a new original physical campaign.

Next create the **read-only native effects export**,
without any new Vitis/Vivado calls:

```bash
python -m fpgai_research.experiments.cst_r2i_fanout_native_effects \
  --study build/r2i_fanout_native_36_pairs_v1 \
  --output build/r2i_fanout_original_full_effects_v1
```

Prepare a **new** original physical study directory,
copying only eligible source-attested RTL and ROMs:

```bash
python -m fpgai_research.experiments.cst_r2j_fanout_physical \
  --native-study build/r2i_fanout_native_36_pairs_v1 \
  --prepare-output build/r2j_fanout_physical_36_pairs_v1
```

Preparation performs **zero Vivado calls**. The output
must be a new unused path. The entire original 72-case
denominator is recorded, with 24 E2-only cases
explicitly blocked.

Inspect the first candidate pair without running tools:

```bash
python -m fpgai_research.experiments.cst_r2j_fanout_physical \
  --study build/r2j_fanout_physical_36_pairs_v1 \
  --start-case 40 --max-cases 2
```

If both cases are marked `PREPARED_NOT_RUN`,
run exactly this irreversible original Vivado pair:

```bash
python -m fpgai_research.experiments.cst_r2j_fanout_physical \
  --study build/r2j_fanout_physical_36_pairs_v1 \
  --start-case 40 --max-cases 2 \
  --execute-native --confirm-isolated-native \
  --timeout-seconds 14400
```

Never rerun either case in its original directory.
Any original vendor failure must remain visible.

Then export the **whole 36-pair physical denominator**,
including unattempted pairs and clean-DRC and
warning-only classifications:

```bash
python -m fpgai_research.experiments.cst_r2j_fanout_physical \
  --study build/r2j_fanout_physical_36_pairs_v1 \
  --export-output build/r2j_fanout_physical_first_audit_v1
```

The export is create-only; use a different output folder
for a later completed cohort audit.

## Explicit scientific limits

An OOC routed placement on a target part, even with
zero DRC and nonnegative WNS, does not establish
a deployed KV260 full-system bitstream or power.
No universal C++/graph FPGA synthesis guarantee or
firstness is implied by isolated examples.

The independent paper still requires peer compiler
comparisons with matching part, toolchain, precision,
sources, constraints and reporting rules. Existing
interval coloring/lifetime storage scheduling is
prior art. The IR's scientific contribution should be
restricted to independently witnessed typed
operator composition, heterogeneous mapping contracts
and whatever repeatable source/HLS/physical effects
survive complete audit.

**R2j physical results are NOT_YET_MEASURED.**
