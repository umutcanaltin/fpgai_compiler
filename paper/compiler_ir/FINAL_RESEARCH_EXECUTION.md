## 2026-11-03 — Original R2j physical pair20 observed, NOT clean-DRC E4

Source-pinned original R2i study `b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`
is preserved. R2j source/test checkout at `adaa373145ed2e3ecf508b78091db4620338e777`
passed **8 focused and 1402 full tests**. Complete native R2i
read-only effect exporter: **72/72 E2 CSim**, **48/72 E3 CSynth**,
**24/36 both-E3 native pairs**, **6 HLS-estimated BRAM savings**,
**18 HLS ties**, **12 E3-unknown pairs**, no HLS regression on
qualified cases. All **24/24 qualified pairs** reduced
HLS-estimated worst-case cycles, mean 24.9583 cycles
and median 23 cycles. These are NOT physical measurements.

R2j source-pinned physical staging prepared **48/72**
original E3-qualified Verilog/ROM packages and kept
**24/72 E2-only cases blocked** in the full 72-case denominator.
Original routed pilot was preselected pair20: `case_040`
disjoint and `case_041` per-element optimal.
Both original Vivado OOC invocations exited 0,
completed route and met timing; routed
**WNS +26.356 ns versus +27.304 ns**,
descriptive optimized-minus-control WNS **+0.948 ns**.
Both have independently classified disclosed DRC warnings:
**2/72 warning-only routed observations, 1/36 matched
routed-observation pairs, ZERO strict E4-qualified cases
or pairs**. Do not misrepresent as clean DRC.
Original physical counts and exact warning rule IDs were
**not included in the supplied terminal paste**, so
actual routed BRAM improvement is UNKNOWN.
Original physical tree and first export are immutable;
do not rerun these two Vivado cases.

New subsequent read-only diagnostic tool
`experiments/cst_r2j_physical_diagnostics.py`
and tests read the existing original physical audit to
show DRC IDs, routed BRAM/LUT/FF/DSP, descriptive
effects and source provenance, with no native calls.
Those new tests and tooling have **NOT** been verified
on the user's workstation yet. See
[R2j original warning-only pilot record](R2J_ORIGINAL_WARNED_PHYSICAL_PILOT_20261103.md).
Continue no new original physical calls until actual
DRC warning categories/resources have been inspected.
Scientific firstness and submission remain NOT READY.

## November 3, 2026 — FULL ORIGINAL R2i NATIVE COHORT FINISHED; R2j physical prospective

**Original research source immutable at**
`b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`.
User workstation completed **72/72 original Vitis HLS executions**
with **72/72 independently qualified numerical CSim (E2)**.
Only **48/72 original CSynth reports (24/36 matched pairs)**
met the strict source/latency/resources/uncertainty E3 gate.
The other **24 cases / 12 pairs** passed CSim but original
Vitis `latency_best = undef`; they remain **UNKNOWN E3**
without imputed cycles or resources.

Across all **36** prespecified pairs: **6** report one lower
**HLS-estimated BRAM_18K** for the optimized arm,
**18** HLS-estimated BRAM ties, **0** qualified BRAM
regressions, **12 UNKNOWN**. All 24 E3-qualified
pairs showed lower **HLS-estimated worst-case cycles**,
but this conditional finding excludes 12 E3-unknown pairs
in the original denominator. There is **NO** new placed/routed
clean-DRC physical E4 data and **NO** FPGA Fmax/bitstream
or independently established external novelty.

The original complete 72-case study
`build/r2i_fanout_native_36_pairs_v1` is frozen.
The exact original native audit was saved as
`build/r2i_fanout_native_final_audit.json`.
See [R2i complete original native results](R2I_FULL_ORIGINAL_NATIVE_HLS_RESULTS_20261103.md).

New subsequent R2j implementation, **NOT YET
workstation-tested and with no Vivado executions**:
read-only 36-pair native effects exporter plus separate
72-case/36-pair OOC Vivado source/RTL/ROM packaging,
strict routing/DRC audits, opt-in physical case runner.
Original 24 E2-only cases remain visible but blocked
from physical qualification. First preselected HLS BRAM
reduction pilot: pair20, case040 vs041, 3 vs2
HLS-estimated BRAM_18K, not yet physically verified.
See [R2j strict physical protocol](R2J_STRICT_FANOUT_PHYSICAL_PROTOCOL.md).

**Publication remains NOT READY** pending routed
hardware comparisons and valid external peer parity.

## 2026-11-03 — R2i native fanout handoff implementation (UNEXECUTED)

After independently verified original R2h (1388 full tests, 36/36
matched source scratch reductions, 72 original GCC executions and
72/72 read-only source trace verification at
`efc6497cdab2ec5734a0c3750972655b9ea397b9`),
the new R2i study prepares the **entire exact 36-pair
population / 72 native Vitis source packages** from the
same R2h source layout lowering.

R2i reuses canonical Vitis `write_vitis_package`, numerical
`tb.cpp`, fixed-target/clock/uncertainty `run_hls.tcl`,
and existing provenance-aware vendor runner,
`summarize_case`, and `qualify_case`. The native
source and audit commands make **zero original vendor calls**.
Original native CSim/CSynth is **explicit opt-in**,
bounded by pair and requires both
`--execute-native --confirm-isolated-native`.
The native auditor shows incomplete/failed cases
as UNKNOWN/REJECTED (not zero benefit); complete
qualified HLS metrics are estimates, NOT achieved
FPGA clock or routed BRAM.

**The 72 prepared sources, native tool calls, E2/E3 reports,
and strict Vivado E4 results are NOT yet workstation-verified.**
Run focused/full tests, prepare new
`build/r2i_fanout_native_36_pairs_v1`,
and inspect a one-pair pilot before extending.
See [R2i experiment protocol](R2I_MATCHED_NATIVE_FANOUT_HLS_PROTOCOL.md).

## 2026-11-03 — original R2g completed; R2h matched actual source memory pending

**Source-pinned original R2g `0a92f4fa0630d81c51bb53ad04bb42304a67d2e5`:**
the user observed 18/18 focused tests, **1375/1375** full regression
passing, **36/36** original GCC runtime-loop C++ fanout programs
passing producer/reader/Add Phi and all-intermediate numeric
checks, and a separately executed **36/36 read-only audit**
of saved original source, runtime trace and numerical results.
The original R2g scratch layout was **disjoint**; no optimized
allocation was executed in that original cohort.
The original source/input/trace artifacts are frozen.

**New R2h (prospective; not yet executed on the workstation):**
A matched **36-pair / 72-GCC-program** source experiment
now emits control disjoint versus certified minimum
per-element scratch array layouts with the same runtime
fanout source order, Phi and input. Both arms write/read
through actual runtime C++ table-selected scratch
addresses and print the **same C++ address variable**
next to the source memory operation. All actual event
addresses, full producer/reader/Add source readiness,
Phi issue and intermediate/final numerical values
are audited independently, with a separate read-only
72-source verification campaign and mutation tests.

**Scientific limits:** If R2h passes, the measured effect
will be **C++ scalar-scratch array capacity under one
serialized source event order**. It does not establish
physical FPGA BRAM, port legality, clocked schedule,
throughput, bit-exact HLS or scientific firstness.
Only an original native HLS / strict clean-DRC
physical matched study and externally pinned
feature-equivalent peer comparison can address
those remaining paper gates.
Full source protocol:
[R2H_ACTUAL_ALLOCATED_FANOUT_SOURCE_PROTOCOL.md](R2H_ACTUAL_ALLOCATED_FANOUT_SOURCE_PROTOCOL.md).

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

> **MOST RECENT FOLLOW-UP — November 3, 2026.**
> The independent `cst_r2b_causal_audit` already
> completed successfully, saved as
> `build/cst_r2b_independent_causal_audit_v1`.
> **Do not rerun the 24 original GCC cases** or
> overwrite their original output or this audit.
> Its only failing test was a `pytest.raises`
> expected-error-message mismatch. The test was
> repaired; run focused/full tests only, not
> another experiment:
>
> ```bash
> cd ~/Desktop/github_projects/fpgai_research
> git pull --ff-only origin main
> python -m pytest -q tests/test_cst_r2b_causal_audit.py
> python -m pytest -q
> ```
>

> **MOST RECENT STATUS — 2026-11-03 00:04.**
> All **12/12 source-pinned v6 Vivado original case
> invocations have COMPLETED** in the user's terminal.
> They are all valid **exploratory warning-disclosed**
> routed observations, and **zero are strict clean-DRC
> E4**. The remaining-case Vivado commands elsewhere in
> this historical runbook are now **obsolete: DO NOT RUN
> THEM**. The full workstation suite passed **1,236**
> tests and original CPU/GCC paper series 40/40 stages.
> The negative Phi result is **0/18 changed last-use**
> and **0/18 changed scratch capacity despite 18/18
> effective mapping edits**. Do not overclaim it.
>
> **Current next commands are read-only or create-only
> audited exports**; see the bottom section and
> [R2 negative finding](CST_R2_NEGATIVE_FINDING_AND_NEXT_FALSIFICATION.md).
>

# CST Model IR — completion controls and the remaining scientific gates

**Status:** executable research infrastructure, **NOT a finished research result or
submission-ready paper**. Updated 2026-11-02. The user's workstation verified
1182/1182 Python tests and 71/71 focused tests at source
`ad748e8b42c2a58aabf6f6176b006c10a7ee11df`, then prepared the
**v6 original Vivado physical study** with 12/12 `PREPARED_NOT_RUN`
cases and zero Vivado receipts. Source SHA for original Vitis HLS
packages: `8814bd0005e8a852348b49f19131b2410117216b`.

This runbook replaces manual completion across the six matched NCHW
source-storage pairs. **The new completion/paper/counterfactual modules
still require independent workstation testing.** CPU test fixtures are
never original paper observations. Historical `v3/v4/v5` and the original
`build/nchw_native_v1` may not be modified or rerun.

## 1. Code verification — no FPGA tools

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q \
  tests/test_cst_nchw_physical.py \
  tests/test_vivado_postrun_rom_attestation.py \
  tests/test_vendor_tools.py \
  tests/test_cst_research_completion.py \
  tests/test_cst_paper_gate.py \
  tests/test_cst_phi_counterfactual_effects.py \
  tests/test_cst_prior_art_environment.py
python -m pytest -q
```

Do not execute original Vivado until this gate passes. GitHub Actions
failures with no usable logs do not supersede actual workstation
tracebacks or justify a passed claim.

## 2. Read-only v6 plan and exact population

```bash
python -m fpgai_research.experiments.cst_research_completion \
  --study build/nchw_bram_vivado_rom_v6
```

Require: 12 source-matched HLS cases, 6 complete disjoint/shared
pairs, 12 `PREPARED_NOT_RUN`, all source/original ROM hashes matching,
zero original Vivado receipts and no blocking case.
This command invokes **zero** native tools and writes **zero** files.

## 3. First original Vivado case and independent re-audit

After preflight success, consciously opt into a single original
foreground vendor invocation:

```bash
python -m fpgai_research.experiments.cst_research_completion \
  --study build/nchw_bram_vivado_rom_v6 \
  --execute-native --confirm-isolated-native \
  --max-new-calls 1

python -m fpgai_research.experiments.cst_nchw_physical \
  --study build/nchw_bram_vivado_rom_v6 --audit
```

The source-pinned v6 `$readmem` receipt may use the prospective,
pre-execution SHA-256 + original vendor-read-success witness.
This attests the observed read, not the vendor's entire internal
memory payload or any RTL equivalence proof.

**Do not proceed if the result is UNKNOWN, REJECTED, FAILED, or
an unexpected warning rule.** Leave its immutable files in place;
the completion controller will refuse automatic continuation.
No code change qualifies an original failed receipt retrospectively.

Accepted continuation states:
`QUALIFIED_VIVADO_OOC_ROUTE_TIMING_E4` (strict DRC zero)
or `OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS`
(full source/ROM/report/routing/timing provenance, but **not E4**).
The warning-only tier is *pilot-informed* by v5 and explicitly
exploratory, never a preregistered primary physical result.
Allowed rule categories are only `DPIP-2, DPOP-3, DPOP-4,
RTSTAT-10` at severity Warning, with all counts preserved.

## 4. Finish all twelve original physical cases

Only after case_000 meets one of those **explicit** source-attributed
classifications, execute the remaining bounded original cases:

```bash
python -m fpgai_research.experiments.cst_research_completion \
  --study build/nchw_bram_vivado_rom_v6 \
  --execute-native --confirm-isolated-native \
  --max-new-calls 12
```

The controller skips original successfully classified cases
without re-running them and executes each still-prepared case exactly
once. It stops at the first unknown/failed result. This is a
**foreground, licensed Vivado** campaign and may involve eleven
additional vendor calls, each with a 14,400-second upper bound;
never interpret this as an asynchronous run.
If a case fails, preserve it and report remaining cases as UNKNOWN.
Do not use loops that delete failed cases or regenerate original
source. Mixed clean-DRC and warning-allowed results are not merged
into the strict primary E4 denominator.

## 5. Evidence exports and independent HLS–physical paper audit

The following are read-only except for their explicit **create-only
output directories**, and invoke zero native FPGA tools:

```bash
python -m fpgai_research.experiments.cst_research_completion \
  --study build/nchw_bram_vivado_rom_v6 \
  --export-output build/nchw_v6_paper_export

python -m fpgai_research.experiments.cst_paper_gate \
  --native-study build/nchw_native_v1 \
  --physical-study build/nchw_bram_vivado_rom_v6 \
  --output build/cst_paper_gate_v1
```

The physical export includes `cases.csv`, `pairs.csv`,
`matched_pairs.json`, `original_physical_audit.json`, and
`summary.json`. Strict E4 pair effects are populated **only
when both cases are clean-DRC E4**. Separately named
`descriptive_*` effect fields may be populated where BOTH
original cases have qualified, disclosed exploratory observation
provenance. Unknown pairs remain null and all six pairs remain
in the denominator.

The paper gate separately checks 12/12 HLS E3, six source-matched
HLS pairs, original C++ hashes and revisions, full routed
observations and strict E4 pairs. It will NOT automatically
declare paper submission ready: original trained-model generality,
independently executed peer systems, causal novelty and institutional
IP/release permission cannot be established from these receipts.

## 6. Phi-conditioned source counterfactuals: investigate novelty, not merely hashes

Run a NEW original source-only GCC study only in a create-only folder
(not over historical E1 cases). Then use the separate secondary
analysis, which reports all 18 controlled Phi changes from 24
prespecified original GCC source cases:

```bash
python -m fpgai_research.experiments.cst_two_phi_reader_campaign \
  --output build/cst_two_phi_reader_e1_final

python -m fpgai_research.experiments.cst_phi_counterfactual_effects \
  --input-csv build/cst_two_phi_reader_e1_final/cases.csv \
  --output build/cst_two_phi_effects_final
```

Important scientifically: a change to `reader*_phi_sha256`,
`source_event_sha256` or generated C++ hash can reflect
identity/loop order only. The analysis explicitly reports
whether **source-executed last-use** and **source scratch capacity**
also change at *fixed Functional graph, producer Phi and source
layout*. Requested Phi edits with no effective change count as
negative evidence, not omitted cases. A positive result still
does not prove priority versus other compilers. Neither the
secondary CSV nor its tests substitute for independently replaying
the original E1 GCC vendor/source log.

## 7. Remaining work required to claim finished research

- **A scientific contribution:** demonstrate a nontrivial
  `Phi → element-wise readiness/complete reduction → all consumer
  READs → final RELEASE → physical address/port` counterexample
  that a clearly defined restricted baseline cannot express.
  Show actual executed source, independently computed semantic oracle,
  legal/illegal interventions and falsifiable cross-graph controls.
- **External comparison:** reproduce source-pinned Allo and
  ST-Flow artifacts, plus HIDA, StreamTensor and Wavelet where their
  semantics/tools admit an honest comparison. Freeze the original
  source checkout, model/shapes, data, numeric policy and exact
  compiler commands. Unsupported, inaccessible and unexecuted
  comparisons are UNKNOWN, never counted as FPGAI wins.
  See `PRIOR_ART_PINNED_ARTIFACTS.md`.
- **Trained-model generality:** original provenance-checked MNIST
  ONNX, data and outputs on the actual workstation; do not invent
  unavailable model weights. Native fixed-point semantics must match
  before reporting generalization.
- **Submission:** update the full manuscript with only original
  independently audited evidence, complete failure denominators,
  limitations and publication approvals. Avoid novel-first,
  optimal-BRAM, achieved-clock, power and defense/robotics claims
  without appropriate original reproducible evidence.

The professional Morfics five-layer multi-vendor compiler and the
frozen RL/Gym work remain separate; no experiments in this
runbook modify those projects.


## 8. External peer-compiler source checkout preflight

A new read-only CLI verifies the **exact pinned source SHA,
GitHub origin and clean repository status** before any external
artifact experiment. The pinned `ST-Flow` source is its
`PACT26` evaluation tag and requires Python 3.12 and Vitis
HLS 2024.2 for native CSim; FPGAI's original cases used
Vitis HLS 2023.2, so hardware numbers cannot be treated
as like-for-like unless version/method are aligned or
explicitly reported as a limitation. The pinned Allo, HIDA
and Wavelet checkouts are *source snapshots*, not proven
paper-specific evaluation revisions. StreamTensor currently
has no verified runnable source pin in this study.

Example **after separately cloning/checking out the proper tag**:

```bash
python -m fpgai_research.experiments.cst_prior_art_environment \
  --peer ST-Flow \
  --checkout ../ST-Flow
```

For a durable, separate, create-only preflight receipt add
`--output build/peer_stflow_preflight_v1`. Peer source
presence by itself verifies **none** of numerical parity,
multi-reader Phi contracts, certified scratch last use,
underlying source compiler behavior, E3 native outcomes or
scientific priority. An executable source-matched adapter
and actual original logs/outputs must be developed and
run per peer separately; no source-only preflight can
substitute for this stage.


## 2026-11-03 full CST C1/C2/C3 experiment launch (CPU/GCC only)

The user has verified the preceding code at SHA
`98ee6097...`: **110 focused and 1221 full tests passed**,
and one original `v6/case_000` Vivado 2023.2 call
was classified **OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS**.
Do **not** run this original case again. Its exact WNS, resources
and DRC counts are not present in the received transcript.
Original strict E4 cases remain 0, physical matched pairs 0,
prepared v6 cases 11. The source/test updates committed after
this revision still require fresh workstation testing.

The canonical `experiments/compiler_ir/run_all.sh`
paper series now includes all pre-existing CPU/real-GCC
CST source stages PLUS:
`rl_compatible_ir_r3_r4` (C3 variable-graph source/mask
invariance and proposer-independence) and
`two_phi_counterfactual_e1` (C1/C2 actual GCC 24-case
two-reader Phi-change secondary effects). It also runs
the complete `test_cst_*.py` source-level test family,
without importing, training or modifying the separate
frozen RL agent experiments.

### Test and inspect the exact original CPU plan

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q
bash experiments/compiler_ir/run_all.sh \
  --profile paper --skip-onnx --plan
```

`--plan` does not create outputs or execute experiments.
Omit `--skip-onnx` to include independent ONNX checks
if all required packages are installed. Ensure no unrelated
native/learned experiment is running in the checkout.

### Execute all original CST CPU/GCC paper series in one terminal

Choose a **new unused output path**:

```bash
bash experiments/compiler_ir/run_all.sh \
  --profile paper --skip-onnx \
  --output build/cst_c123_paper_20261103
```

All series outputs are create-only; the runner records source
HEAD SHA, exact commands, raw case files, original GCC traces,
stage statuses and hashes. It stops downstream steps when
dependencies fail and preserves every original failure;
it never fabricates a passed stage. An interrupted study
can only resume under the SAME source SHA/configuration
and an exact matching `--resume` path (do not run the
same `--output` again). There are **NO original Vitis
or Vivado invocations** without additional explicit native
flags. Fresh source preparation under the series is NOT
a replacement for the historical immutable NCHW/Vitis
source package.

At the end review the generated
`evidence_index/paper_evidence.json` and
`evidence_index/paper_evidence.md`. The index reports
source-level completion for C1/C2/C3 separately, but
even all three `source_evidence_complete=true` means
**not** scientific firstness, hardware improvement, or
trained RL advantage. The Phi analysis must trace its
exact source CSV bytes back to the same original
`two_phi_readers_e1` GCC stage.

### Continue the existing v6 physical campaign independently

After all new unit tests pass, the **already-executed**
`case_000` must be left intact. To complete its FIRST
matched source-storage comparison, run **only** the
next case, `case_001`, by setting `--max-new-calls 1`:

```bash
python -m fpgai_research.experiments.cst_research_completion \
  --study build/nchw_bram_vivado_rom_v6

python -m fpgai_research.experiments.cst_research_completion \
  --study build/nchw_bram_vivado_rom_v6 \
  --execute-native --confirm-isolated-native \
  --max-new-calls 1

python -m fpgai_research.experiments.cst_nchw_physical \
  --study build/nchw_bram_vivado_rom_v6 --audit
```

If and only if the independent audit accepts both original
source-attributed, timing-met routed observations and preserves
the two warning lists and complete DRC descriptions, the
first descriptive physical matched difference can be
computed in a create-only export. If case_001 is rejected,
freeze its original evidence and do NOT run an automatic
repair/rerun. Later independent case-pairs may be run
under the bounded completion controller as appropriate
after acceptance, always preserving DRC-clean strict E4
versus warning-only exploratory evidence.

### Remaining independent scientific dependencies

Code readiness alone is not completion of scientific claims:
actual source-pinned upstream Allo and ST-Flow comparator
execution (with exact source/toolchain/numeric parity),
independently matched trained-model evidence,
Phi-value lifetime causal differences rather than hash
renaming, and manuscript review/IP permissions remain
outstanding. R5 trained RL performance comparisons are
NOT required to say the compiler has an RL-compatible
architecture interface; they ARE required to claim better
RL sample efficiency or faster convergence.


## 2026-11-02 late checkpoint: completed source-only paper series and first native pair

**New original results:** `1228/1228` unit tests and
`40/40` CPU/GCC paper stages PASSED at historical committed
SHA `1cefdd8566658cb6b69e12444c9b5ce9ba6e4a31`.
The original `v6/case_000` and `v6/case_001` now each
have an independently accepted WARNING-ONLY routed observation,
giving 2/12 original Vivado receipts, first 1/6 descriptive
routed pairs, **0 strict E4 cases/pairs**, 10 unexecuted cases.
Both exact cases are frozen: DO NOT rerun or overwrite them.
The original terminal log does **not** include their numeric
resource/timing values.

### Read original Phi and RL results (zero vendor calls)

```bash
cat build/cst_c123_paper_20261103/two_phi_counterfactual_e1/summary.json
cat build/cst_c123_paper_20261103/rl_compatible_ir_r3_r4/summary.json
cat build/cst_c123_paper_20261103/evidence_index/paper_evidence.json
```

Crucially, `changed_last_use_contrasts` might be 0:
if so, report it as negative causal evidence. An owner
hash or source C++ difference is NOT a true lifetime effect.

### Original physical pair effect review, no new vendor execution

The new source-only review should first be tested locally
(`python -m pytest -q`); it has not run on the user's PC
at the time of this documentation update.

```bash
python experiments/compiler_ir/summarize_research.py \
  --cpu-series build/cst_c123_paper_20261103 \
  --physical-study build/nchw_bram_vivado_rom_v6 \
  --output build/cst_firstpair_readonly_review_v1
```

This is a **create-only** report output. Omit
`--output` for a completely read-only stdout review.
Its report `cross_cohort_review.json` contains
`original_vivado.paired_comparisons[0]`: ORIGINAL
per-case LUT/FF/DSP/BRAM/URAM, WNS and warning rule
counts with signed `disjoint_minus_shared` differences.
A positive BRAM difference means shared reported less
BRAM; no unqualified result may be substituted with zero.
The source-only CPU study revision and original HLS source
revision are recorded separately and are NOT pooled.

Also perform independently source-pinned
`python -m fpgai_research.experiments.cst_paper_gate
--native-study build/nchw_native_v1
--physical-study build/nchw_bram_vivado_rom_v6`
without `--output` for a read-only HLS/physical case
provenance check.

### Ten remaining native Vivado cases

Only after reviewing the first matched pair and fresh
source tests, the existing original v6 case_002–case_011
can be executed, **once per case**, with this explicit,
foreground bounded command:

```bash
python -m fpgai_research.experiments.cst_research_completion \
  --study build/nchw_bram_vivado_rom_v6 \
  --execute-native --confirm-isolated-native \
  --max-new-calls 10
```

This may perform up to 10 actual native Vivado invocations.
It stops automatically on the first unaccepted case; unknown
or failed originals remain unchanged. It neither retries
case_000/001 nor makes future plans in the background.
Re-audit each original result and produce a NEW create-only
physical export when the cohort reaches its stopping point.
Any results that remain missing must stay explicitly UNKNOWN.

## FINAL 12-CASE COHORT: export only, NEVER rerun original vendor calls

All original v6 source-pinned Vivado cases `case_000`…
`case_011` have now been executed **exactly once**;
all 12 are full-route/timing-met but have disclosed noncritical
DRC warnings, so **0/12 strict clean-DRC E4**.
The 40-step source-only paper campaign and 1,236 full
workstation tests passed at the recorded historical revision.
The next step is **only audited physical/statistical
aggregation**, not more original native calls.

After pulling the current committed code and passing
the new tests, create **fresh, unused** output directories:

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_cross_cohort_review.py
python -m pytest -q

# Complete source-pinned original routed study audit, no vendor invocation:
python -m fpgai_research.experiments.cst_research_completion \
  --study build/nchw_bram_vivado_rom_v6

# Original six-pair full denominator, all case/DRC/timing/resource records:
python -m fpgai_research.experiments.cst_nchw_physical \
  --study build/nchw_bram_vivado_rom_v6 \
  --export-output build/nchw_v6_full_physical_export_v1

# Independently compare C1/C2/C3 source and original physical cohorts:
python experiments/compiler_ir/summarize_research.py \
  --cpu-series build/cst_c123_paper_20261103 \
  --physical-study build/nchw_bram_vivado_rom_v6 \
  --output build/cst_full_cross_cohort_review_v1

# Recheck original HLS→Vivado case revisions and distinct evidence tiers:
python -m fpgai_research.experiments.cst_paper_gate \
  --native-study build/nchw_native_v1 \
  --physical-study build/nchw_bram_vivado_rom_v6 \
  --output build/cst_original_paper_gate_full_v1
```

If any `--output` path already exists, choose a NEW path.
Do not delete or overwrite original evidence to force a rerun.
Read:

```bash
cat build/nchw_v6_full_physical_export_v1/summary.json
cat build/nchw_v6_full_physical_export_v1/matched_pairs.json
cat build/cst_full_cross_cohort_review_v1/cross_cohort_review.json
cat build/cst_original_paper_gate_full_v1/readiness.json
```

In the new cross-cohort review,
`original_vivado.six_pair_prespecified_descriptive_rollup`
reports all 6 original paired signed resource differences,
their shared-less/equal/shared-more counts, WNS slack
tradeoffs and missing denominators; strict-E4 values
are separately empty because every case had DRC warnings.
The complete six-pair *numbers* are not available in
the uploaded terminal log and must be obtained from
original untouched case reports.

**Scientific falsification:** `original_cpu.phi_causal_effect_gate`
records `NEGATIVE_CAUSAL_RESULT_ZERO_LAST_USE_AND_CAPACITY_EFFECTS`
for the original 24 GCC /18 Phi interventions.
This cannot be turned into a success by changing the
definition of last-use, reporting only different Phi-owner
hashes or re-running old sources.
Use [the new prospective R2b protocol](CST_R2_NEGATIVE_FINDING_AND_NEXT_FALSIFICATION.md)
to test an actual interleaving source scheduler in NEW
prospectively versioned study packages; no future
outcomes are claimed here.


## Prospective, separate R2b E0 implementation — not historical source evidence

Do not rerun or alter the old E1/Vivado cases. A
new bounded `interleaved_branch_schedule` E0
prototype now contains a deterministic one-rule
two-reader PE-group issue plan and first-class
independent logical per-element last-use indices.
It can test **whether a proposed legal issue order
would change last-use under a Phi edit**.
It does **not emit/execute C++** or qualify E1/HLS.

```bash
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_r2b_interleaved_schedule.py
python -m pytest -q
python -m fpgai_research.experiments.cst_r2b_interleaved_schedule \
  --output build/cst_r2b_logical_e0_v1
```

This is create-only and new-revision source-pinned.
Read its `summary.json` and `cases.csv`
without interpreting changing E0 release ordinals
as a real GCC or FPGA lifetime effect.
**The next sprint needs emitted source execution
and an independent numerical oracle; this remains
unimplemented and blocks the strong R2b claim.**


## 2026-11-03 prospective R2b original GCC source execution (not yet tested locally)

The latest original source-pinned user E0 report at
`90bf86d666bfeb5fcad56a56734db43cd3dc37f2`
had **1245 full tests passed**; 12/12 E0
interleaving certificates, 9/9 interventions changed
**global event last-use ordinals**. This alone does
NOT prove true final-reader ordering changes,
C++ execution or reduced source memory.

The new **independently named E1 study** is
`cst_r2b_source_campaign`: 3 graphs ×
4 Phi reader policies × 2 scratch layouts =
**24 original GCC case sources**. Each source
actually reads/writes C++ scratch memory in its
prospectively fixed PE-grouped reader issue order.
`verify_r2b_gcc` checks the actual stdout
typed event stream against bounded certificates
and independently compares outputs against
NumPy's FunctionalModel interpretation.
An additional *H-only read-rank and final
reader identity* metric prevents differences in
intervening COMMIT count from being mistaken
for a changed lifetime. Source files, original
input, all GCC stdout, original verification
and all 24 case denominators are preserved.

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_r2b_source_campaign.py
python -m pytest -q

# New create-only original E1; choose an unused output:
python -m fpgai_research.experiments.cst_r2b_source_campaign \
  --output build/cst_r2b_original_gcc_e1_v1

# Original SHA/typed-trace audit, NO recompilation or vendor calls:
python -m fpgai_research.experiments.cst_r2b_source_audit \
  --study build/cst_r2b_original_gcc_e1_v1

cat build/cst_r2b_original_gcc_e1_v1/summary.json
```

**As of this edit, the new source code and tests have
NOT run on the user's original machine.**
Do not pre-fill manuscript effect counts or cite
an E1 outcome until the actual original CLI,
numerical verification and read-only audit pass.
Preserve the original `build/cst_r2b_logical_e0_v1`
and all earlier serial-E1/Vivado study artifacts;
do not rerun those cohorts. If the new original
GCC attempt fails, retain the emitted source,
case IDs and `FAILED.json`, correct the code in
a new source revision, then use a new study folder.
Detailed method and limitations:
`R2B_E1_ORIGINAL_GCC_PROTOCOL.md`.


## New post-original read-only R2b causal verification — do not run GCC again

The original R2b E1 24-case source campaign and
its first auditor **already passed** on the
user's machine, source pinned at `5d449ca...`.
Do **NOT** execute it again; this second audit
is independent read-only reanalysis of those
exact files. A newer code revision adds stronger
denominator and functional-payload checks;
its unit tests have not yet been run by
the user.

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_r2b_causal_audit.py
python -m pytest -q

python -m fpgai_research.experiments.cst_r2b_causal_audit \
  --study build/cst_r2b_original_gcc_e1_v1 \
  --output build/cst_r2b_independent_causal_audit_v1

cat build/cst_r2b_independent_causal_audit_v1/r2b_causal_audit.json
```

`--output` is optional and create-only.
Omit it for a fully read-only stdout report.
The auditor preserves 24 original source
cases, 12 independent storage-layout controls,
and explicitly collapses 18 layout-scoped Phi
comparisons to **nine unique graph/Phi
interventions**. It refuses to accept changed
global event ordinals as a strong H-only
final-read source result. Source NumPy outputs
are re-evaluated from saved original float32
inputs; zero original GCC/Vitis/Vivado calls.
Never overwrite the old output folder.

**New scientific status after the original
E1 evidence:** 6/9 distinct graph/Phi
interventions showed changed actual H-only
source final-read rank or consumer identity;
3/9 changed final-consumer identity.
No same-layout Phi-caused source scratch
capacity benefit, native hardware schedule,
peer compiler novelty or trained RL
benefit was demonstrated.


## New R2c per-element lifetime and scratch mapping research — prospective

**Latest actually verified user checkpoint (not R2c):**
the fixed R2b causal audit regression was repaired and
the user's machine passed **5 focused** and **1258 full**
Python tests on source commit
`578178890607fa085ef0c30a9a270e123108a3bb`.
The independent original R2b causal audit is complete
and remains untouched. Claude's independent review
will write only its own Markdown file, not source code.

**NEW R2c implementation is committed but not yet
run on the user's PC.** The new research modules
add finite source per-value intervals, a deterministic
optimal unit-slot allocator with separate bounded
exhaustive reference oracle, exact event-by-event
alias checker, real allocated C++ scratch access
source generator, original GCC runtime slot witnesses,
independent NumPy comparisons and a prespecified
12-matched-pair / 24-GCC-source experiment.
A separate read-only audit reconstructs the 12
saved pair artifacts, original source/trace hashes,
full typed slot event stream, exact minimum live
unit slots and numerical output.

The verified source array-footprint comparisons
must include **five declared mutable arrays**:
scratch plus acc_a, acc_b, add_a, add_b
(4*m*q additional array elements). C++ scalar
locals, read-only weights, boundary inputs/outputs
and physically mapped FPGA RAM resources are
**NOT** included and MUST NOT be claimed.

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q \
  tests/test_cst_r2c_element_allocation.py \
  tests/test_cst_r2c_allocated_cpp.py \
  tests/test_cst_r2c_allocation_campaign.py
python -m pytest -q

# Only after the new regression passes. NEW original source cohort:
python -m fpgai_research.experiments.cst_r2c_allocation_campaign \
  --output build/cst_r2c_original_matched_gcc_v1

# Read-only re-audit of that cohort; no gcc/Vitis/Vivado invocations:
python -m fpgai_research.experiments.cst_r2c_allocation_audit \
  --study build/cst_r2c_original_matched_gcc_v1 \
  --output build/cst_r2c_independent_allocation_audit_v1
```

Both output directories must be NEW and unused.
No original R2b, native HLS or Vivado calls should be
repeated. Source code, tests and sprint plan were
implemented; no R2c original case result or new
full-test pass has yet been observed.

See [R2c prospective source-allocation experiment](R2C_EXACT_VALUE_LIFETIME_STORAGE_STUDY.md).

## Corrected regression and prospective R2d source-original launch

The completed `build/cst_C2_e0_mutation_adequacy_v1`
output has 96/96 detected selected mutants and is
**immutable**; do NOT rerun it. The same original
full regression at `82ef9f2...` had 1294 pass /
1 expected-exception regex mismatch; this exact
test was repaired on main, untested on workstation.

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q \
  tests/test_cst_r2b_source_campaign.py \
  tests/test_cst_r2d_loop_driven_cpp.py \
  tests/test_cst_r2d_loop_campaign.py
python -m pytest -q

# Only if green: NEW create-only 12-case runtime C++ GCC source experiment
python -m fpgai_research.experiments.cst_r2d_loop_campaign \
  --output build/cst_r2d_original_runtime_loops_v1

# Read-only independent audit; no GCC/vendor executions
python -m fpgai_research.experiments.cst_r2d_loop_audit \
  --study build/cst_r2d_original_runtime_loops_v1 \
  --output build/cst_r2d_runtime_independent_audit_v1
```

The two new R2d directories must NOT already
exist. Keep all source-pinned original R2b/R2c
and C2 E0 mutation experiments unchanged.
The new R2d implementation is not yet
observed working on the user's machine.

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

## 2026-11-03 — original R2e producer completeness qualified; R2f fanout E0 unexecuted

**New original user-provided receipt, immutable original revision
`d41568c1f913e245692815f733a537799c6df7bd`:**
`compileall` succeeded; R2e focused tests **20/20 passed**,
entire pytest suite **1,330/1,330 passed**.
The independent R2e original source experiment
`build/cst_r2e_original_producer_gcc_v1`
completed **12/12 original GCC compilations/executions**,
with **312 complete producer contribution witnesses**,
12/12 source H reduction/COMMIT checks,
12/12 intermediate H numeric checks,
12/12 independent reader Phi/alias and final
NumPy checks. The separate read-only audit
`build/cst_r2e_readonly_producer_audit_v1`
confirmed **12/12** saved original source and
trace SHA, producer scalar-sum H numeric values
and full consumer numerical traces, performing
**0 GCC/Vitis/Vivado calls**. Original source and
all historical campaigns are frozen.

**Implemented afterward, but NOT tested on the
user's workstation:** CST R2f bounded compositional
source certificates for **2/3/4 reader MatMul
fanout**, complete producer/consumer/Add-chain
readiness, independent fixed Phi issue conformance,
source scratch liveness and a prespecified
**36-case E0** graph/mapping study. A safe
producer-reader interleaving is *logically*
admissible; it has no executable source/hardware
receipt. See
[R2F_COMPOSITIONAL_FANOUT_E0_PROTOCOL.md](R2F_COMPOSITIONAL_FANOUT_E0_PROTOCOL.md).

R2f has **not** demonstrated general DAG
compilation, a new interval-coloring algorithm,
actual GCC source-level translation,
native HLS, clean-DRC physical BRAM,
RAM port timing or external scientific firstness.
Research remains **not paper submission ready**.

## November 3, 2026 — R2f original E0 verified; R2g 36-case host-GCC E1 prospective

Original R2f cohort at
`98676f7786687eddace0cab9124868448f9bdc24`:
**27/27 focused; 1357/1357 full regression; 36/36 source
certificates; 18/27 changed logical unit-slot capacity;
9/27 ties; no regressions.** All 36 have complete
producer/reader/Add/final E0 readiness and independent
Phi source order; no original GCC/Vitis/Vivado.
This data cannot be equated to physical BRAM
or actual C++ optimized scratch reuse.
Historical original E0 source folder remains frozen.

New R2g source-level experiment implemented on main:
actual runtime-loop C++ read indices for fanout 2–4,
real producer and reader reductions, Add-chain writes,
actual committed numerical intermediate values,
separate forward-rank Phi and source readiness/alias
checker and NumPy for EVERY intermediate. New
**36 original GCC cases** and an independent
read-only SHA/NumPy audit are source-pinned, create-only.
An adversarial test suite injects wrong runtime ordering,
missing producer contribution, altered arithmetic,
wrong committed H value and early reader commits.
All are NOT YET workstation executed as of this
record. See `R2G_RUNTIME_FANOUT_E1_PROTOCOL.md`.

The remaining scientific proof gap is not "more tests":
R2g disjoint-source operation is different from
the 18-case R2f optimized *logical unit-slot*
counterfactual. A same-order actual optimized-source
comparison, Vitis CSim/CSynth, strict clean-DRC
Vivado measurement, exact external peer parity
and broader operators remain separate publication
gates. RL remains frozen.
