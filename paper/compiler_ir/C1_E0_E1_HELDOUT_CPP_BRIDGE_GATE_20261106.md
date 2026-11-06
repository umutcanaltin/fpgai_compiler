# C1 held-out source-to-generated-C++ refinement: independent E1 gate

**Scientific protocol and result status, 6 November 2026. Both the original two-profile/four-execution smoke and the separate original full 54-profile/108-execution generated-GCC E1 study are now AUTHOR-VERIFIED. The first paper remains MAJOR REVISION / NOT SUBMISSION READY: independent Allo/CODO novelty, routed timing/latency and rights/release gates remain open.**

## Workstation verification update (6 November 2026)

After source revision `d0b72b4a00309ff01f8f159d035390c43c35789c` was pulled on the author's machine, the author ran the exact targeted test suite including `tests/test_cst_paper_c1_heldout_cpp_bridge.py`. The terminal reported **55/55 tests passed in 2.55s, with no skipped tests**. That suite's fixture-based `test_real_gcc_scalar_and_mixed_phi_paired_source_e1_smoke` and emitted-source-tamper test actually call the host GNU g++ pathway with a **synthetically constructed 54-profile E0 report and test-monkeypatched original hash/source revision**. Passing those tests is evidence that the new source-to-GCC bridge test harness can compile and execute bounded scalar/mixed-Φ sources and audit their local outputs, not an independently signed original author dataset.

**Original C1 E1 data remain NOT EXECUTED:** The subsequent `make release-links` failed with `ModuleNotFoundError: No module named 'scripts'` inside `scripts/check_public_docs.py` and the outer command used `set -e`; therefore the separately planned **author-original report-SHA-pinned** `build/paper_c1_e1_cpp_bridge_smoke_20261106_01` and 54-profile/108-run full study were NOT created or verified by that command. A dedicated CLI-path fix and regression test are now committed, but must be pulled/retested. **Do not promote fixture-based GCC tests to all 54 real original-source E1 qualification.**

## Verified original C1 two-profile E1 GCC bridge smoke (6 November 2026)

**The previously preregistered, original SHA-locked two-profile GCC smoke is now VERIFIED on the author's workstation.** Exact source revision: `5bc8e9c1ec749c1c3dcf09bf85bace91503ee0c7`. Original immutable C1 E0 source revision: `637f2745721b32d6f5a7e02c4659b0623bc9eee7`, original E0 report SHA `8fa3848fcb0993e44a83391f595861943d75e3f09b669bd93b1b27e0222f3978`. Output name `build/paper_c1_e1_cpp_bridge_smoke_20261106_01` was created once and then audited read-only. It reported:

- `profiles=2`, one scalar and one mixed-`Phi` mapped source configuration, each with `disjoint` and `per_element_optimal` actual C++ emitted storage layouts.
- `gcc_compilations=4`, `gcc_executions=4`, `no_vendor_calls=true`, `original_c1_event_hashes_matched=true`, and **positive source scratch-slot saving in 2/2 profiles** (`source_slot_savings_count=2`, ties 0). The stdout lacks exact slot counts, so no absolute savings or physical BRAM figures may be claimed.
- Independent `--study` audit verified source revision `5bc8e9c1…`, four actual C++ compilations/executions, **11 source artifacts** and original evidence manifest SHA-256 `51e022a07e35ab4ee96aed25d4a7bb990d97d49b54e98a07d977a6ed0bce4435`; audit made zero vendor/compiler calls and reported `physical_compute_latency=null`, `board_qualified=false`.

This is a real **E1 smoke observation**, not merely the earlier synthetic test-fixture GCC smoke. It demonstrates the existing source compiler / NumPy / scratch witness checks on two original C1-matched configurations, but the full new 54/108-GCC series remains **NOT EXECUTED**. Preserve the two-profile smoke and its SHA-backed receipt, even if later full-study output fails. Source-only scratch slot reduction is not Vivado mapped BRAM or any physical clock result. Full results, if they occur, must use an entirely separate create-only output `build/paper_c1_e1_cpp_bridge_full_20261106_01` with its own source revision and source-artifact SHA.

See [C1 E1 original smoke and C2 original cost results](C1_E1_SMOKE_C2_SOURCE_COST_ORIGINAL_RESULTS_20261106.md).

## Why this study is scientifically necessary

The original author's **54 bounded profiles / 486 source-event trace tests** (report SHA-256 `8fa3848fcb0993e44a83391f595861943d75e3f09b669bd93b1b27e0222f3978`, source revision `637f2745721b32d6f5a7e02c4659b0623bc9eee7`) established finite model-level agreement between independent event interpreters. That **did not execute generated C++** and could be a checked abstract contract with no implementation refinement. Earlier, separate R2h/R2r work *has* executed emitted fanout C++ and detected 216 original generated-source mutations on different selected graph populations; [R2r evidence](R2R_EMITTED_SOURCE_SHADOW_MUTATION_PROTOCOL_20261103.md). The new controlled bridge explicitly joins the **original 54 held-out C1 profiles** to actual emitted C++ without reimplementing the validator or changing either original source study.

**Existing independent mechanisms reused:** `cst_paper_c1_seeded_metamorphic._architecture` creates exactly the original finite graph, architecture and events; `verify_fanout_events` and `verify_fanout_issue` re-certify E0. `emit_allocated_fanout_cpp` generates real HLS C++ via the independent R2h source path for both `disjoint` and `per_element_optimal` scratch layouts. The existing `verify_allocated_gcc` compiles/executes the emitted C++ with GNU g++, checks actual address-adjacent scratch READ/COMMIT witnesses, checks source event ordering/actual last reader use, and independently compares all intermediate and final tensor values against the FunctionalModel NumPy implementation. The validator returns original source/architecture/weights/input/trace identities. This is actual E1 host execution, **NOT vendor HLS/RTL or cycle-level hardware**.

## Original matched cohort and frozen prerequisites

- The existing E0 study `build/paper_c1_seeded_metamorphic_20261106_01` must remain untouched; SHA-256, original source revision, population 54/486 and zero original oracle disagreements are mandatory, and **all 54 original canonical typed event hashes are cross-checked against the new revision before GCC**. A mismatch stops the experiment; do not alter the original C1 files or silently regenerate them.
- Fixed three seeds 111/222/333, fanout readers 2/3/4, mapping `scalar`/`mixed_phi`, and original 54 held-out static profile tuples; two memory layout arms, **108 actual compiler-emitted GCC compilations and 108 executions** if all cases complete. All of these are counted separately from the earlier 486 *source-event* traces; do not pool denominators.
- Each arm's source C++ SHA and actual stdout SHA, functional/architecture SHA, input SHA, exact scratch slot count, number of witnessed actual address reads/writes, and independently evaluated intermediate/final numerical absolute error are recorded. The two arms must have the **same** functional graph, architecture and input identity; reuse layout source slots may be smaller, equal, but not greater than the disjoint layout for the same fixed order.
- The earlier C1 `safe_wrong_issue_order` mutation is a value-safe **alternative to the declared schedule**, not a functional error. This new study tests actual **canonical schedule emitted C++**, not a claim that every other valid source ordering is wrong. A full interpreter-to-RTL/simultaneous-memory-port proof is out of scope.
- Require source checkout with clean *tracked* Git state and real local `g++`. No files are added or altered inside the original E0 directory. A partial/failing new E1 attempt is preserved in its own create-only `FAILED.json`; the exact original partial directory must **not** be deleted or rerun. Later fixes require a new source SHA and a new separately named study. No GitHub Actions, Vitis, Vivado, board execution or hardware resource claims.

## Correct evaluation and paper claim

**Positive criterion:** 54 profiles × 2 distinct layouts = 108 genuine GCC source executions, all 108 `R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED`, zero original canonical event SHA mismatches, source/architecture/functional/input SHA matches per pair, original actual address witness and full numerical intermediate/final checks, exact report/SHA manifest consistency and no vendor calls. Report all failures, skipped cases, and resource/slot ties, not only improvements. An E1 success supports bounded generated-C++ implementation-source refinement on these **54 selected fanout configurations**, not arbitrary-graph correctness, malicious binary memory instrumentation or mapped BRAM reduction.

**Negative/failure criterion:** if even one real GCC compile, address witness, output correctness, mapping/hash binding or E0 event SHA test fails, preserve the failure and label the 108-pair E1 study **not qualified**. A full paper should report the counterexample, not regenerate the same output, adjust the seed list or relax tolerances.

**Relationship to hardware QoR:** the R4 original hardware comparison reduces routed BRAM 7→4 but worsens LUT/FF/WNS, and both fail timing/clean DRC. A GCC source slot reduction is not a mapped BRAM, FPGA frequency, II, compute latency or end-to-end DMA result. All physical latency remains **UNKNOWN**.

## Test and execute locally: two distinct new output names

```bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m compileall -q src/fpgai_research/experiments/cst_paper_c1_heldout_cpp_bridge.py
python -m pytest -q \
  tests/test_cst_paper_c1_heldout_cpp_bridge.py \
  tests/test_cst_paper_c1_seeded_metamorphic.py \
  tests/test_cst_r2h_allocated_fanout_cpp.py
test -x "$(command -v g++)"
test ! -e build/paper_c1_e1_cpp_bridge_smoke_20261106_01
python -m fpgai_research.experiments.cst_paper_c1_heldout_cpp_bridge \
  --original-c1-study build/paper_c1_seeded_metamorphic_20261106_01 \
  --output build/paper_c1_e1_cpp_bridge_smoke_20261106_01 \
  --quick
python -m fpgai_research.experiments.cst_paper_c1_heldout_cpp_bridge \
  --study build/paper_c1_e1_cpp_bridge_smoke_20261106_01
```

Only **after** the separate original two-profile (scalar and mixed Phi) GCC smoke has passed its source SHA/audit gates should the **new, different full original output** be attempted:

```bash
cd ~/Desktop/github_projects/fpgai_compiler
test ! -e build/paper_c1_e1_cpp_bridge_full_20261106_01
python -m fpgai_research.experiments.cst_paper_c1_heldout_cpp_bridge \
  --original-c1-study build/paper_c1_seeded_metamorphic_20261106_01 \
  --output build/paper_c1_e1_cpp_bridge_full_20261106_01
python -m fpgai_research.experiments.cst_paper_c1_heldout_cpp_bridge \
  --study build/paper_c1_e1_cpp_bridge_full_20261106_01
```

Do not use `--quick` output as a 54-profile publication trial. The script never mutates source files; it creates only its new output directory, C++ source/actual stdout evidence per case, an original summary and SHA inventory.

## Change-impact map and scientific ownership

| File | Ownership and dependencies | Status |
|---|---|---|
| `src/fpgai_research/experiments/cst_paper_c1_heldout_cpp_bridge.py` | NEW experiment; imports existing C1 E0 generator, R2h allocated emitter, actual GCC/Numpy validator and R2b deterministic input; no changes in production code | Committed, **not workstation tested** |
| `tests/test_cst_paper_c1_heldout_cpp_bridge.py` | NEW source-only unit/synthetic receipt and optional real `g++` four-arm smoke; tests SHA binding, immutability, source mutation/failure preservation | Committed, **not workstation tested** |
| `paper/compiler_ir/C1_E0_E1_HELDOUT_CPP_BRIDGE_GATE_20261106.md` | NEW exact hypothesis/denominator/provenance/safe-run protocol | Committed, **not an experiment** |
| `scripts/audit_source_inventory.py`, `tests/test_audit_source_inventory.py` | Separate read-only tracked-file ownership and static import map for public maintenance; no study output or compiler semantics changes | Committed, **new tests unverified** |

**Important novelty status:** This is the strongest practical next connection from finite source-event contracts to real generated code, but the result could still be standard source-instrumented functional verification. We **must** compare the capability against a supported pinned Allo equivalent and distinguish `Phi` schedule conformance from whole-output equivalence. Existing Allo, CODO and ST-Flow already offer substantial overlapping schedule and dataflow validation; never assert exclusivity from an unsupported API/import failure.
