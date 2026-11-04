# FPGAI compiler-IR paper — preregistered experiment plan (PROSPECTIVE)

This document is a **design**, not an experiment report. There are **no new compiler-IR benchmark results** recorded here. Native code/tool changes are deferred while the already-running L0 RL campaign stays source-pinned to its original local checkout. New project planning is committed to remote `main`; do not pull during that campaign.

## Research thesis

**Novel IR focus:** operator-local architecture-semantic `Phi` with **explicit typed graph-edge architecture interfaces**, composition/compatibility and safe refinement into real HLS. Verification and equivalence-class analysis establish utility; they are not asserted as newly invented verification/search techniques.

Primary hypotheses and refutation criteria: [canonical scope](README.md).

## Experimental invariants

- Fixed Functional IR graph/weights/reference inputs per paired architecture comparison, immutable model and artifact SHA.
- Constrain topology, target FPGA part, numerical tolerance, resource caps, clock, source revision, Vitis HLS version and testbench between baseline/extended-IR comparisons; exceptions explicit in results.
- Each proposed architecture has requested + effective contract, source decision origins, graph-edge interface resolution, compiler legality status/rejection reason, Lowered IR fingerprint and generated HLS source hash where emitted.
- When a comparison uses different precision or clock, state that these differ and do not treat as a matched same-constraint QoR comparison.
- The source-lowering verifier, executed host C++ `Phi` trace and independent numerical oracle test different properties. None establishes post-route timing. CSim/CSynth reports are independent native evidence.
- Unsupported constructs, failed CSim, missing XML, timing exceptions, resource violations and no feasible design remain in denominator of attempts/qualification; never count failures as zero latency.
- Maintain create-only study roots, published seed/task selection and read-only audit; store raw native reports outside Git by default.

## E0: Source capabilities and prior-art preflight (CPU, 0 native calls)

**Inputs:** source modules `frontend/`, `model_ir/contracts.py`, `architecture/resolver.py`, `architecture/mapping.py`, `model_ir/lowering.py`, `codegen/hls/emitter.py`, `validation/cpp_phi_trace.py`.

**Procedure:** trace a simple single MatMul, chained two-MatMul/Gemm, and branch/merge case only where truly supported. Record how graph edges are represented today; where interface precision/layout/transport/temporal semantics currently reside; which formats really lower. Record source-supported/unsupported per feature and closest concrete capabilities from HIDA/ScaleHLS/FINN/HeteroCL/Calyx/CIRCT. Do not describe current one-operator `Phi` or C++ trace as newly implemented.

**Outputs:** `source_capability_matrix.csv`, `comparative_prior_art.md`, exact source SHA/command. **Gate:** one nontrivial graph-level architectural interface requirement must be identifiable and implementable for C2; otherwise change thesis.

## E1: Graph-level architectural composition / negative cases (CPU, 0 native)

**Prespecified design:** target initially at least 2 connected MatMul/Gemm operators with independently varied local PE/SIMD/serialization/tile/pipeline values, fixed Functional IR. Test widths of 3, 4, 5, 8 and 9 at relevant edges (to expose tails) only if legal importer/compiler; add other values if justified and record additions. Start from a deterministic finite configuration catalog; aim for **100–200** CPU cases including at least **20 deliberate negative cases**; these are targets, not measurements.

**Checks:**
1. Functional shape and tensor edge compatibility; no silent tensor shape change.
2. Type/numeric format compatibility (or an *implemented* explicit converter); a different PE count alone is not automatically an incompatibility.
3. Explicit mapping from producer output organization to consumer input requirements; reject unsupported transfer/granularity/handshake assumptions.
4. Same task graph fingerprints across architecture variations; independent node scopes retained.
5. No hidden fallback to uniform architecture when the per-operator contract is legal.

**Measurements:** legal/invalid/unsupported counts, reason-class confusion, per-case verification time, unique effective graph-contract SHA, errors and minimal reproducing cases. **Gate:** no false acceptance of invalid or unsupported graph interfaces; positive examples must pass actual reference execution for tested supported topology.

## E2: Restricted IR ablation — is the proposed representation useful? (CPU, 0 native)

**Baseline B0:** *same FPGAI compiler implementation*, same import, HLS emitter, device/numeric and source revision; force globally uniform per-operator architecture and fixed/implicit interface assumptions at the **input configuration layer**, rather than deliberately weakening the compiler's correctness checks. This is a restricted-expressiveness baseline, **not an external prior-art compiler**.

**Full B1:** same compiler, typed per-operator choices + explicit graph-edge compatibility contracts.

**Design:** run B0/B1 over identical network/task fingerprints and a prespecified finite choice catalog (not an unfairly larger call budget). Report: number of representable legal graph mappings, distinct effective graph contracts, distinct generated HLS hashes, how many edge incompatibilities are explicitly caught, how many source/host/numerical checks pass, elapsed CPU validation time, rejected/unknown statuses.

**Anti-confounding:** Distinct user parameters do not imply distinct physical designs; fingerprint effective structure and generated source separately. Avoid claims of QoR improvement without native evidence.

## E3: Architecture-lowering conformance (CPU and actual executed host C++; 0 native)

**Procedure:** For all manageable E1 positive graph configurations (bounded to 20k logical iterations per test initially, configurable), invoke existing independent source-loop/`Phi` oracle and compile/execute generated HLS C++ via the existing host trace, subject to source-supported models. Compare typed interface and per-node architecture to emitted implementation manifest. Check numerical correctness against independent reference on fixed inputs; enforce declared tolerance.

**Fault tests:** inject controlled, isolated defects in graph interface contracts, boundary/tail handling, PE assignment, SIMD, tiling or lowering in test fixtures; count detection, missed injected faults, false positives on valid cases, time/counterexamples. A fault which never compiles/executes should be categorized as compile-time reject, not misrepresented as a semantic detector success. Test the same faults against existing per-op oracle to show *incremental* graph-composition coverage.

**Gate:** no novel compiler result is published if the only verifier is an algorithm comparing its own internal values; include independent generated code execution and negative controls. Large-domain witnesses and algebraic derivations are **not** machine-proved source-to-RTL equivalence.

## E4: Architecture equivalence / distinctness (CPU, 0 native)

Use the same E1/E2 catalog and record three distinct identities per candidate: raw requested-contract signature, canonicalized **effective architecture+graph interface** signature, and actual emitted C++ SHA. Publish counts, class-size histogram, representative aliased requests, changed-effective/same-source cases, changed-source/same-effective cases. **Do not claim same effective SHA means identical HLS QoR, timing or RTL**. If no aliases exist, report zero rather than adjust sample.

## E5: Small original-Vitis native HLS qualification (explicit opt-in)

**Intended budget:** target 12–24 independent original HLS candidate evaluations, contingent on passing E0–E4 and available time. First a handful of structural/CSim smoke candidates; then prespecify a balanced set with distinct local mappings, supported interface types, odd-size tails, at least one restricted baseline and one independent full-IR alternative where *both* are source/legal. Each source change requires a new campaign identity; do not reuse or mix old RL L0 study directories. Do not launch until the existing source-pinned L0 job/audit finishes and new compiler source is stable.

**Original source evidence:** HLS-generated sources/Tcl/manifest SHA, original fresh CSim, reference numeric errors, CSynth whole-top worst cycles, requested target and uncertainty, estimated timing budget, LUT/FF/DSP/BRAM (and URAM where present), actual tool version/part, failure reasons, vendor wall time; independent read-only source audit. Objective reported as **HLS estimate `cycles × requested_clock_ns`**, conditional on qualification, never routed/real clock. Recording HLS resource/timing values is not a claim of physically implemented circuit.

**Primary comparisons:** controlled B0 vs B1 on matched networks, same original vendor tool and source. Report paired qualified absolute HLS ns only on the same task/clock and numeric format, resource changes, qualification coverage, compile cost and missing results. If both cannot be matched, report as separate illustrative realizations rather than a speedup.

## E6: Optional external reference flow (only if fair)

Use hls4ml or FINN only for a workload with identical mathematical model, weights, fixed-point semantics, target part and clock, and comparable HLS tool versions. Prior source snapshots can be inspected but not silently merged into prospective results. If equivalent configurations cannot be built, report capability comparison only; no quantified superiority claim.

## E7: RL-suitable IR — design/interface demonstration, **NO RL experiment**

**Question:** Can the same compiler IR provide a variable-topology, typed, legal-action API that a future optimizer *could* consume?

Implement a tiny CPU-only, no-training adapter or read-only serialization containing:
- graph operators, shapes, connectivity and type applicability masks;
- per-operator architecture choices and their scopes, resource/numeric constraints and global clock envelope if the API makes them explicit;
- compiler-derived *legal* candidate decisions, with rejected decisions separately diagnosable;
- deterministic seeded/scripted selection and independent compiler lowering/source-fingerprint round trip.

Test single- and two-operator supported shapes, clean schema stability and invalid-action rejection. **Evidence claim:** “IR supports a structured interface for prospective search/RL.” **Not allowed:** “RL learns faster due to IR”, “IR improves RL reward”, “trained shared GNN generalization”, “first RL-ready FPGA IR”. Those require future, separate matched learning studies and are frozen.

## Results ledger to populate only after execution

| Output | Population source | Status now |
| --- | --- | --- |
| Source capability/prior-art comparison | E0 | NOT RUN |
| Graph compatibility and negative-case table | E1 | NOT RUN |
| Restricted vs full IR expressiveness | E2 | NOT RUN |
| Independent lowering/numerical/mutation checks | E3 | NOT RUN |
| Effective mapping equivalence classes | E4 | NOT RUN |
| Original HLS cases and paired QoR | E5 | NOT RUN |
| Optional external baseline | E6 | NOT RUN |
| Typed graph search interface | E7 | NOT RUN |

## Stop/go and publication readiness

To claim the **new compiler-IR contribution**, E1 must establish a feature the existing per-op IR did not express and C2 must be *implemented and materialized*. E2 must demonstrate a verifiable expressiveness difference. E3 must validate the entire targeted lowering path. E5 supplies limited native HLS evidence (without guaranteeing faster QoR). Report which claims are E0/E1/E2/E3 rather than claiming E4/E5/E6 physical hardware without actual measurements.

Prior manuscript results (MLP/CNN precision, execution-policy, proposed detector pipeline) are **historical separate evidence**, not these experiments. They may be reused only after authentic source revision, target/numeric/tool consistency and independent attestation. Paper authorship, release rights and Radboud IP review occur before public redistribution of source or raw artifacts.

## E0 source-review checkpoint (2026-11-01)

[The source inventory](SOURCE_CAPABILITY_AUDIT.md) is complete as a **read-only code audit**, not as freshly rerun CPU tests or a finished prior-art comparison. Initial static-array boundary diagnostics and new compiler-only tests are committed; fresh local CPU test confirmation is pending. E1–E7 remain **NOT RUN**. The current source backend explicitly rejects mixed per-op numeric formats and multiple graph-top inputs/outputs; never count those cases as implementable without further work.

## Critical additional novelty experiment: coupled operator execution / tensor boundary identity

Use [CST-Model IR novelty specification](COMPOSITIONAL_SPACETIME_IR_NOVELTY.md). Target MatMul→MatMul with an intermediate shape M×N, M,N>1; compare canonical row-major storage against a **physically encoded and lowered** alternate logical-to-array layout/transfer. Derive producer-write-after-reduction and consumer-read element identity/partial-order obligations and independently verify actual compiled-host C++ addressing and final reference outputs, not merely source metadata. Prespecify differing PE/SIMD/reuse/tile choices as *operator-mapping variables*, not automatically interface incompatibility. Compare same mathematical model/clock/numeric/device/tool version; count legal, incompatible, source-correct, CSim-qualified and qualified original-HLS candidates. HLS speedups are hypotheses only. Allo/StreamTensor/HIDA/Stream-HLS provide closely overlapping prior art, so characterize the *joint per-iteration Phi and per-tensor-element cross-operator contract* rather than generic composition, streaming, or layouts. No evidence is complete until tests actually run.


## New source implementations available for prospective E1–E5 measurements

**Not yet run on workstation.** Both CPU-only campaigns are create-only and have zero vendor/agent calls.

| Study | CLI | Planned metrics |
| --- | --- | --- |
| A: row-major vs column-major source mapping | \`python -m fpgai_research.experiments.cst_ir_campaign --output build/compiler_ir/array_v1 --max-cases 96 --host-check-pairs 4\` | Source/Functional/model identity, layout mapping receipt, independent Phi exchange event hash, actual C++ numerical qualification on selected pairs. |
| B: whole-tensor vs per-row lifetime | \`python -m fpgai_research.experiments.cst_row_campaign --output build/compiler_ir/row_lifetime_v1 --max-pairs 18 --host-pairs 4\` | Same Functional graph and operator-local architecture, distinct emitted schedule, one-row scratch vs whole intermediate *in generated C++*, actual host numerical output. |
| C: correctness and temporal witness tests | \`python -m pytest -q tests/test_compositional_model_ir.py tests/test_cst_ir_campaign.py tests/test_cst_row_lifetime.py tests/test_cst_row_campaign.py\` | Typed interface and unsupported-case rejects; host-compiled numerical checks; generated C++ Phi iteration trace; per-row ready/consumed order. |

Study B is more causally informative than a layout-only comparison because it changes **inter-operator value availability and storage lifetime**. It is still a row-barrier program, **not a concurrent streaming HLS dataflow network**. The source memory-size ratio does not establish post-synthesis BRAM savings. No claims about Vitis clock, power or physical resource outcomes until actual native reports exist. Both A and B may have no latency benefit; publication must disclose that.

Any subsequent E5 native comparisons should use fixed same Functional graph/weights/numeric constraints/part/clock and matched per-op Phi; prespecify 12–24 total fresh cases, and independently verify original reports. Protect all historical experiment roots and frozen RL code.


## Executable original-HLS experiment implementation and limitation

The native source study is now prepared in source at \`experiments/cst_native_campaign.py\`, using the pre-existing, source-hashed \`write_vitis_package\`, \`run_vitis\` and independently attested \`summarize_case\` pipeline. It prepares 4 matched triplets (12 cases maximum by default): existing full intermediate tensor, explicit column-major intermediate tensor, and typed row-lifetime interleaving. Architecture/numerics/part/target period remain matched **within each triplet**. Preparation is CPU only; original Vitis runs require \`--execute-native\` and explicit bounded \`--max-calls\`. A read-only audit remains valid even after the study source revision ages; executing more native cases requires the original source revision.

Run commands in [RUNBOOK.md](RUNBOOK.md). New E0/E1 mutation tests \`test_cst_ir_mutations.py\` deliberately corrupt intermediate access or temporal event order in generated C++; \`cpp_row_trace.py\` checks actual host-executed row-ready/row-consumed order, and \`validate_host\` checks actual numerical effects. Test suite and full CPU study not executed in this environment. **No new E2/E3 case has been generated**; summary output never invents HLS results.

Before using native metrics for main-text speedup claims, add/execute a strict target-clock feasibility gate and resource budget checks using actual original Vitis reports. The current native auditor intentionally withholds that qualification.


## HLS scorecard and uncertainty discipline

The first native trial matrix alternates uniform and heterogeneous per-operator \`Phi\` decisions over four source-pinned matched triplets (12 original HLS sources by default). The reported source-level row-buffer size, operator parameters and hash differences are **not** measured hardware resource savings. The read-only \`cst_native_report.py\` exports only genuinely source-attributed HLS cases with CSim qualification plus bounded CSynth latency, and explicitly omits any pair missing those observations. Native clock/uncertainty request is constant within paired cases; achieved target timing remains unverified. Detailed commands in [RUNBOOK.md](RUNBOOK.md).

**Scientific comparison still missing:** published Allo, HIDA, StreamTensor implementations may express overlapping tensor-lifetime transformations. Therefore compiler-IR novelty requires a feature-level study and measurable independent advantage in expressiveness, diagnostics or transformation composition, beyond basic row-buffer reuse. No paper may state proven priority solely from this implementation.


## New E1 experiment — bounded multi-operator lifetime composition

**Research hypothesis:** For static MatMul chains with independent local Phi, a compositional row-lifetime exchange policy can preserve the complete graph function while replacing M×N materialized internal tensors with a reusable N-element source scratch per edge under explicitly verified nonoverlapping row epochs. This is a *source-level* representation/lowering hypothesis, not a claim that hardware memory automatically shrinks.

The \`cst_chain_campaign\` study targets 12 cases across 3- and 4-MatMul graphs, odd internal dimensions, and uniform/heterogeneous independent per-layer PE/SIMD/reuse. CPU E0 contracts plus generated HLS identity and code differences are checked in all cases; up to 4 are executed with GCC host-float numerical validation and up to 2 with actual generated C++ Phi and row-lifetime event traces. Failed/unsupported cases must remain visible. No fabricated or missing native hardware values are allowed.

The new \`cst_cpu_report\` verifies source revision and case counts of all three campaign CSVs before producing \`cpu_results.md\`. This prevents an E1-only row-buffer result from being accidentally promoted to a physical BRAM/latency benefit or combining different source revisions. The chain results are supporting compiler expressiveness/semantic evidence, **not enough alone to prove novelty versus Allo/StreamTensor**.

The GitHub CI workflow packages raw reports as \`compiler-ir-cpu-\${{ github.sha }}\` and retains them for a bounded duration; cite the source SHA and run URL in any paper table.


## Independent lifetime invariant experiment

For every row-chain sample, calculate \`2 × rows × Σ(intermediate_widths)\` producer/consumer **element events** and verify all per-tensor scratch addresses are released before reallocation to the next logical row. Hash the event stream to allow independent reproductions. Negative controls mutate typed physical capacity or finite proof budget and must reject rather than silently pass. Actual compiled host C++ row-ready/consumed event traces remain a **different evidence source**, and the generated source's \`Phi\` coordinates and numerical output are independently checked. A passing model-level certificate must not be promoted into an assertion of actual HLS cycle order or physical BRAM allocation.


## S3 nonlinear semantic preservation and trained assets (implementation 2026-11-02)

Controlled E1 experiment: keep the *complete* sequential 4-stage graph Flatten(axis=1) → Gemm with optional transposed constant weights and bias → ReLU → Gemm with bias. Vary row block B in legal {1,2,M}, M in {2,3,5}, and Gemm transB in {0,1}; preserve matched whole Functional SHA and resolved operator Phi. Test actual emitted C++ against the independent NumPy model, including data with positive and negative pre-activation values; record row traces. Synthetic weights establish source semantic conformance only, not trained accuracy.

Separate trained-model E1: original source-hashed MNIST MLP ONNX and fixed-shape examples, independently checked by ONNX Runtime and the NumPy Functional IR, followed by actual generated GCC host-float execution. Record sample/model SHA, full operator identities, tolerances, numerical errors and inability to extrapolate from selected samples. Do not infer fixed-point, CSim, physical or 10,000-image accuracy. Real asset directory is external/private and is NOT implicitly available in GitHub Actions.

Native E2/E3 source: prepare original-Vitis source/testbench/Tcl from actual original trained assets with explicit constant precision, part, clock/uncertainty, and classification labels. Run only on licensed Vitis environment. Batch-one (M=1) gives row source scratch ratio 1; hardware QoR benefits must be tested on separate legal multi-row network workloads.
