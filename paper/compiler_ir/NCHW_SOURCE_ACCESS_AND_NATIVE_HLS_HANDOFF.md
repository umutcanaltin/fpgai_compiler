# NCHW CST — observed C++ read/write refinement and source-paired native Vitis handoff

**Research status, 2026-11-02.** Two separately verified original source revisions:

- **GCC observed source event proof:** `77fe5dfcd96ad81ff8d71f36fe57cb88a47411e4` — [37/37 quick + 37/37 paper CI](https://github.com/umutcanaltin/fpgai_research/actions/runs/41480551778), [raw ZIP artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41480551778/artifacts/13159323224); full regression **1,092 passed, five skipped**.
- **Original Vitis case PREPARATION:** `e5502be096bd4b8ab0cf15b82e2ad4c96d0eac84` — [38/38 quick + 38/38 paper CI](https://github.com/umutcanaltin/fpgai_research/actions/runs/41481941961), [raw ZIP artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41481941961/artifacts/13159804282); full regression **1,095 passed, five skipped**. Exactly **zero original Vitis calls** in the preparation series.

Later source-only changes add an opt-in native executor, independent read-only E2/E3 evidence classification and per-case manifest integrity. Their qualification must use their own newer CI, not be backdated to either SHA above.

## 1. Source-observed READ/WRITE event relation (E1)

`validation/nchw_access_instrument.py` transforms the genuine HLS-emitted NCHW C++ *only for an opt-in GCC experiment*: the actual produced-tensor array subscripts become host-only reference proxies, whose conversion operator logs an actual READ and whose assignment logs the actual WRITE. This binds observation to executed source data accesses, rather than printing an independent desired event schedule. It observes every produced intermediate tensor READ, all produced tensor WRITEs, and final output WRITEs; constant kernels/external input reads are not part of the reported intermediate-access denominator. The ordinary Vitis HLS source is not replaced by the observer.

`validation/nchw_dynamic_access_oracle.py` derives expected per-stage source-element counts with independent NumPy tensor identity grids and the original numerical interpreter. This is input-dependent: the generated ReLU ternary may read a **positive source value twice**, and MaxPool reads the initial candidate **once before the four-tap loop**, so the first tap is actually read twice. It would be scientifically incorrect to equate source-level access multiplicity with the E0 mathematical minimum.

`validation/cpp_nchw_full_access.py` compiles and runs the opt-in actual source in GCC and checks complete per-source-element/per-producer-stage READ/WRITE multisets, real execution order, each producer's write before a consumer read, no duplicated/omitted stores, and no overwrite of a still-live shared scratch address. A numerical final-output check independently uses the NumPy oracle. All identities and source event hashes are retained. Negative controls mutate actual source access indices, forged scratch offsets or the source digest, and reject missing or malformed event evidence.

**What this does NOT prove:** individual access attribution to every Conv output reduction owner (the separately executed Conv Phi trace covers those), arbitrary C++ formal memory semantics, Vitis fixed-point arithmetic equivalence, native HLS scheduling cycles, bank/port inference, RTL, or physical place-and-route timing.

**Original prespecified E1 study `nchw_real_source_rw_e1`:** 3 distinct static mixed-operator NCHW graph families × 2 independent Conv Phi policies × 2 genuinely generated C++ storage plans = **12 GCC executions**. All **12/12** actual event/causality and NumPy checks passed; **3,424 actual produced-tensor READs, 1,928 WRITEs**, six matched source scratch pairs with strictly smaller C++ array allocations under conservative reuse (minimum eight elements), and **zero Vitis calls**. No hardware-qualified cases. Actual source layout differs between each matched pair; these are not imagined E0-only savings.

## 2. Original HLS source-paired native study PREPARATION (E1, NOT E2/E3)

`experiments/cst_nchw_native_pairs.py` calls the canonical `write_vitis_package`, testbench and Tcl writers to generate **12 immutable case directories (`case_000` through `case_011`)**: 3 graphs × 2 Conv mappings × {disjoint C++ scratch, real shared C++ scratch}. Each pair has the identical Functional Model IR/weights, NumPy samples, architecture, fixed-point representation, original CSim testbench, part, requested HLS clock and explicit uncertainty, and numerical tolerance. C++ source SHA and scratch capacity genuinely differ. Each original case includes `model.cpp`, `tb.cpp`, `run_hls.tcl` and `manifest.json`. The study manifest retains case, source, testbench, Tcl, architecture, samples and clock provenance.

**Default pinned source settings** (editable with CLI arguments): AMD Kria KV260 `xck26-sfvc784-2LV-c`, 32.0 ns requested HLS period, 8.64 ns explicit uncertainty, and 0.1 absolute CSim numerical tolerance. These are **requested constraints**, not known achieved clock or physical QoR. Generated source is original `ap_fixed` HLS code, not GCC `FPGAI_HOST_FLOAT` observers. The native HLS tool may reject pointer aliases or produce RAM allocation/timing that differs from source array capacity.

The read-only `audit_prepared()` checks complete paired source hashes and the canonical per-case manifest, including numerical tolerance/clock/part; mismatched, stale, incomplete or mutated packages fail closed. The source-only artifact explicitly reports `csim_evidence_status=NOT_RUN`, `csynth_evidence_status=NOT_RUN`, `native_qualified_cases=0`.

### Workstation commands — isolated clone, explicit foreground opt-in

From the **current source-pinned, isolated research checkout** (do not modify the original RL/Gym checkout or historical experiments):

```bash
python -m pip install --no-deps -e .

# Use a new output directory; never overwrite an earlier study.
python -m fpgai_research.experiments.cst_nchw_native_pairs \
  --output build/compiler_ir/nchw_native_pairs_original

# Read-only verification; zero Vitis invocations.
python -m fpgai_research.experiments.cst_nchw_native_pairs \
  --audit build/compiler_ir/nchw_native_pairs_original

# Read-only native evidence audit: all native metrics null before Vitis.
python -m fpgai_research.experiments.cst_nchw_native_execute \
  --study build/compiler_ir/nchw_native_pairs_original --audit-native

# Preview the first exact matched pair. Zero Vitis invocations.
python -m fpgai_research.experiments.cst_nchw_native_execute \
  --study build/compiler_ir/nchw_native_pairs_original --max-pairs 1

# Only with Vitis HLS genuinely installed, after isolating this checkout:
python -m fpgai_research.experiments.cst_nchw_native_execute \
  --study build/compiler_ir/nchw_native_pairs_original \
  --execute-native --confirm-isolated-native \
  --start-pair 0 --max-pairs 1 --timeout-seconds 7200

# Read original source-attested CSim / CSynth evidence. Failed/unknown stays null.
python -m fpgai_research.experiments.cst_nchw_native_execute \
  --study build/compiler_ir/nchw_native_pairs_original --audit-native
```

The execution entry point is **not called by the default one-command CPU series**, and will refuse reruns over any case with vendor logs/projects/receipts. Both opt-in flags, an explicit small matched-pair budget and source-integrity gates are required. Progress/logs are native and source-bound; a process return code is never substituted for CSim numerical acceptance or original XML evidence.

The read-only native audit uses the existing independent `summarize_case` provenance checks: it promotes only original vendor receipts and fresh source/Tcl/testbench-bound evidence. A stale or un-attributed XML file does **not** count. HLS worst-case cycles, BRAM and other estimated resources are reported only for pairs with original source-attested CSim and CSynth. Neither an estimated target frequency nor a HLS resource count is a physical placed-and-routed measurement.

## 3. Remaining SCIENTIFIC AND PUBLICATION GATES

1. Run the original **12 Vitis cases** on a licensed workstation, starting with two matched smoke cases, with full original receipts. Record failed/unsupported/missing results in the denominator; qualify fixed-point CSim then CSynth, HLS clock target vs estimate and uncertainty, resource/II/latency, and actual memory ports. A zero exit code or source-level array shrink does not qualify an FPGA benefit.
2. Implement and test a defensible **hardware bank/port, clock and cycle-level contract** grounded in actual vendor schedules and source. Separate HLS-estimated from placed/routed clock performance, and preserve UNKNOWN when reports are absent.
3. Execute **artifact-matched external prior-art baselines** (pinned Allo/HIDA/ST-Flow/StreamTensor/Wavelet where genuinely overlapping), and trained-network benchmarks, with equal numeric/clock/part constraints. Existing pins alone are not a novel-firstness conclusion.
4. Extend the generalized CST source/semantic proof to more operators, multi-input/output topologies, numeric policy, held-out graph families and independent formal witnesses. The current bounded NCHW 2–12 operator subset is not arbitrary ONNX.
5. Keep **RL training frozen**. An eventual RL-ready decision mask must use the same semantics and distinguish source-level verification from UNKNOWN hardware feasibility.

**Publication and superiority status: OPEN.** The increment is genuine bounded source-level evidence; no claimed original Vitis QoR, board timing, physical BRAM improvement or established scientific firstness is supported yet.

## 4. Native HLS qualification and explicit vendor memory intent (implementation checkpoint, awaiting latest CI)

The academic compiler now provides `codegen/hls/nchw_vendor_directives.py`, with explicit **AMD Vitis** source policy `auto` (default) or `bram_2p`. The latter requests exactly one `#pragma HLS BIND_STORAGE variable=__cst_scratch type=ram_2p impl=bram` directly after the real C++ scratch declaration, **on both source variants**. The request has its own C++ SHA and manifest field; a source audit ensures it cannot be silently dropped or duplicated. It is a *vendor-specific lowering intent*, not part of the vendor-neutral Model IR and NOT evidence that HLS actually inferred that primitive.

`experiments/cst_nchw_hls_qualification.py` is a strict **read-only** original Vitis qualifier. To accept a case at E3, it requires original source-attested, numerically passing CSim with exact prepared sample/output counts and tolerance; a fresh, input-hash-attributed CSynth XML; original tool identity; a positive worst-case latency estimate; reported LUT/FF/DSP/BRAM resource fields; and HLS *EstimatedClockPeriod ≤ TargetClockPeriod − exact source-attested ClockUncertainty*. Missing XML, malformed timing, incompatible target, missing required resources or infeasible scheduling never become qualified numeric/BRAM performance. A native HLS source pair is comparable only if **both** sides pass all these gates, with exactly matched part/clock/uncertainty/testbench/model/Phi. Such a comparison is labeled **HLS estimates only**, never achieved frequency or physical BRAM savings.

`experiments/cst_nchw_native_execute.py` reports these separate E2/E3 qualification states after actual foreground Vitis runs; a tool return code is not a substitute. `experiments/cst_nchw_native_export.py` makes a **new**, create-only report directory with `cases.csv`, `pairs.csv`, `summary.json` and `full_native_audit.json`. All 12 prespecified cases and six pairs are retained, including unattempted, failed, unsupported and rejected outcomes. Absent vendor evidence produces zero *qualified cases* and explicit unknown/null metrics, not resource savings of zero.

Example on the **isolated, version-pinned Vitis 2023.2 workstation**:

```bash
# Source-only preparation; vendor pragma requested on BOTH variants.
python -m fpgai_research.experiments.cst_nchw_native_pairs \
  --output build/nchw_vitis_bram_2p_source_pairs \
  --scratch-binding bram_2p

# No Vitis calls: audit source, testbench, Tcl, per-case and matching hashes.
python -m fpgai_research.experiments.cst_nchw_native_pairs \
  --audit build/nchw_vitis_bram_2p_source_pairs
python -m fpgai_research.experiments.cst_nchw_native_execute \
  --study build/nchw_vitis_bram_2p_source_pairs --audit-native

# Run original CSim and CSynth for the first matched pair ONLY.
python -m fpgai_research.experiments.cst_nchw_native_execute \
  --study build/nchw_vitis_bram_2p_source_pairs \
  --execute-native --confirm-isolated-native --start-pair 0 --max-pairs 1

# If the first pair has generated evidence, run the OTHER five only.
python -m fpgai_research.experiments.cst_nchw_native_execute \
  --study build/nchw_vitis_bram_2p_source_pairs \
  --execute-native --confirm-isolated-native --start-pair 1 --max-pairs 5

# No vendor invocations; immutable full-denominator report. Fresh output.
python -m fpgai_research.experiments.cst_nchw_native_export \
  --study build/nchw_vitis_bram_2p_source_pairs \
  --output build/nchw_vitis_bram_2p_original_audit
```

For an **auto-inferred** study, prepare a *separate new source-pinned study directory* with `--scratch-binding auto`. Do not mix auto and explicit BRAM binding in the same matched pair or treat different clock/numeric policies as storage-only comparisons. Check tool warnings and RTL RAM inference: a requested pragma is not proof of successful implementation.

**Qualification boundary:** no native Vitis was invoked by these Python changes or their CPU-only tests. Original vendor CSim/CSynth, inferred memory ports, directives accepted/ignored, post-route timing and actual physical savings are still unresolved until the workstation executes the above commands with retained original logs and reports.

## 5. Original auto-inference versus explicit BRAM_2P source-policy experiment (2026-11-02)

**New user-supplied workstation transcripts, not checked-in vendor artifact files.** The completed explicit `bram_2p` source study at compiler SHA `8814bd0005e8a852348b49f19131b2410117216b` has **12/12 original CSim and CSynth-qualified cases** (six matched disjoint/shared pairs, all clock-uncertainty feasible); inferred BRAM 18K for the pairs is 5→5, 64→64, 9→9, 1→1, 32→32, **7→12**, respectively. Worst-case HLS cycles are 1005→975, 588→586, 757→758, 537→537, 425→426, 1659→1633. Physically qualified cases: **zero**. This is *source-attributed HLS-level evidence as reported by the user*, not an independently archived copy of the original Vitis XML in GitHub.

The new `auto` study at compiler SHA `8fe0ef31f6acc69c25902fe3f7ab94757e6e42bd` has completed **only its first matched pair** so far: both `branch_concat` scalar variants passed original CSim/CSynth and HLS effective-clock gates; both have **0 inferred BRAM_18K, 975 worst-case cycles and +14.267 ns HLS scheduling margin**. They compare provisionally with the earlier explicit policy at 5 BRAM_18K and 1005/975 cycles. **Remaining ten auto cases have not been shown run**. Neither zero inferred BRAM nor an HLS margin proves there is no physical storage or that routed timing closes.

Scientific risk: these two studies were *generated at different Git revisions*. To allow cross-policy HLS comparisons, the new read-only `cst_nchw_cross_policy` verifier insists on byte-for-byte identical original generated C++ after removing **only the one explicitly requested BRAM pragma**; identical model/architecture/weights/numerical samples, part, clock, uncertainty, testbench, Tcl and storage arrangement; fully source-attributed vendor results. Any additional generated-source difference is **rejected**, not ignored as a "pragma-only" ablation.

The policy comparison is a 2×2 matrix per graph: `{auto, bram_2p} × {real_disjoint, real_shared}`. The command exports **all 12 policy-matched source cases** and six groups, including incomplete cases with `UNKNOWN` metrics. It also computes a four-option HLS-estimate Pareto relation over cycles, BRAM, DSP, LUT and FF **only for groups with four qualified options**. A BRAM directive is never forced or chosen by the Model IR; vendor and device capability libraries govern lowering.

### Complete the already prepared auto source study without altering its revision

**Before pulling newer code**, on the original `8fe0ef31…` checkout, preserve `build/nchw_native_auto_v1` and `build/nchw_native_v1`. They contain immutable original Vitis receipts. Never rerun cases 000/001.

```bash
git rev-parse HEAD
# Must identify the AUTO study's original 8fe0ef31... source revision:
python -m fpgai_research.experiments.cst_nchw_native_execute \
  --study build/nchw_native_auto_v1 --audit-native

# Run only cases 002–011 in the same source-pinned checkout, if still unrun:
python -m fpgai_research.experiments.cst_nchw_native_execute \
  --study build/nchw_native_auto_v1 \
  --execute-native --confirm-isolated-native \
  --start-pair 1 --max-pairs 5

python -m fpgai_research.experiments.cst_nchw_native_export \
  --study build/nchw_native_auto_v1 \
  --output build/nchw_native_auto_v1_report
```

**Then** update to a verified revision containing the cross-policy auditor, without deleting or rewriting original source studies:

```bash
git pull --ff-only origin main
python -m pip install --no-deps -e .

python -m fpgai_research.experiments.cst_nchw_cross_policy \
  --auto-study build/nchw_native_auto_v1 \
  --bram-study build/nchw_native_v1 \
  --output build/nchw_cross_policy_v1
```

All cross-study audits run **zero** vendor tools and accept older original source pins *only when all hashes and actual generated C++ canonicalization match*. If either study has an unsupported/missing case, preserve the complete denominator and unknown status. Reports are create-only. Further vendor-physical place-and-route and matched prior-art benchmarks are still open.
