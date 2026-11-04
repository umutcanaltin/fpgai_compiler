# FPGAI compiler-first IR paper — implementation sprints

**2026-11-01 priority:** new FPGA neural-network research compiler paper, not a verification-only paper. **Plan/design only**, no completed-new-IR claim. Execute sprints after independently checking actual code/tests. Work directly on `main` per project owner's instruction, but **do not pull it into an in-progress source-pinned Vitis study** (L0 study started from `5221094af585bd501e842deca94d19c2289bb628`). RL development/new campaigns are frozen; existing L0 native run may finish without interruption. See [scope](README.md) and [experiments](EXPERIMENT_PLAN.md).

## Architectural engineering rules

- One Functional → Architectural → Lowered Model IR lineage, unchanged mathematical graph/tensor semantics and stable operator identity.
- Per-operator hardware choices are independent; no forced uniformity across layers. Explicit model/region/operator/loop scope resolution and effective vs requested provenance.
- Distinguish hardware architecture semantics from HLS directive spelling and vendor capability. HLS target is research backend only; avoid claims about VHDL-first five-layer commercial system.
- Only implement inter-operator interface conversions/adapters that are actually supported by typed compiler code and numerical tests. Interface precision/layout/clock/handshake assumptions must be explicit; no fake streaming claim.
- Fail closed on unsupported shapes, numeric conversions, mixed precision, adapters, temporal protocols or schedule properties; do not silently replace user-pinned decisions.
- CPU/source checks precede native experiments; native HLS opt-in; immutable case roots, source hashes, read-only independent replay, no post-route Fmax claim from HLS.
- Preserve original compiler functionality, older paper results, original RL receipts/checkpoints and source-pinned run histories; refactor with compatibility tests, not by deleting features.
- All data must retain true evidence levels E0 typed/refinement → E1 generated source/host trace → E2 CSim → E3 CSynth → E4 implementation → E5/6 executed validated board, if performed. No automatic evidence promotion.
- For this paper, RL is a *future compatible interface*, not a new optimizer, trained agent, or result.

## Prioritized sprint backlog

| Sprint | Work | Acceptance criteria | Status |
| --- | --- | --- | --- |
| IR-00 | Record compiler-paper RQs, prior art, existing source inventory, freeze boundary | Reproducible scope + paper/experiment docs; no code/HLS changes | **Documentation committed** |
| IR-01 | Audit current end-to-end compiler path: ONNX/tensor Model IR → `ArchitectureRequest` / `Phi` → effective architecture → Lowered IR → HLS; identify **actual** graph/edge interface model | Source-level feature matrix with real tests and missing/partial outcomes, no invented semantics | **Source inventory recorded** in `SOURCE_CAPABILITY_AUDIT.md`; fresh full CPU test still pending |
| IR-02 | Specify typed architectural interfaces across graph edges: tensor shape/dtype/numeric format, layout/transfer granularity, execution/stream/buffer contract, producer/consumer requirements and provenance | Minimal **precise** compatible/incompatible examples; explicit unsupported semantics, existing schemas backwards compatible | **In progress:** conservative E0 static-array interface inspector and source-only tests added; novel typed interface and HLS materialization still pending |
| IR-03 | Implement compatibility and compositional legality for a minimal fully materializable **two-MatMul/Gemm graph** | Individually legal but jointly incompatible examples must fail closed; supported valid graph remains compilable and numerically correct | Pending |
| IR-04 | Implement only necessary, fully specified interface legalization (adapter/cast/repack/transfer) or explicit blocker; no speculative hardware | Independent reference and generated source tests + adapters listed as supported; no implicit numerical approximation | Pending conditional: implement only where nontrivial justified |
| IR-05 | Connect graph-level effective interface decisions into lower IR and HLS emitter while preserving local `Phi` | Actual host-compiled C++ trace, graph-level output reference and operator-level refinement checks pass; no source-only “physical realization” claim | Pending |
| IR-06 | Create restricted-IR baseline: same compiler and HLS target but constrained uniform operator architectures/fixed or no graph interface decisions | Comparable legal inputs/precision/device/clock/testbench; define how unsupported cases are counted | Pending |
| IR-07 | Deterministic CPU semantic campaign: legal/incompatible two-op graphs, multi-op operators where implemented, odd sizes/tails, scoped local choices, negative tests, source-diff/trace | Complete manifests, statuses, reproducible seeds, independent oracle, costs; no Vitis invoked | Pending |
| IR-08 | Architecture expressiveness/equivalence analysis: requested/effective/source identities, with explicit equivalence scope and collision checks | Separate counts for legal requests, canonical effective mappings, generated source; do not claim physical equivalence | Pending |
| IR-09 | Small original Vitis CSim/CSynth sample with IR and restricted-IR controls and qualified baselines | Prespecified 12–24 **maximum target** (may be fewer if high compile cost), source pinned, all original reports/audits, honest no-feasible cases | Pending; licensed workstation only |
| IR-10 | Optional external hls4ml matched case (only if operator/numerical/part/settings truly comparable) | Explicit mismatch disclosure; no invented advantage; omit from headline if unmatched | Pending optional |
| IR-11 | RL-ready *interface-only* smoke: graph/edge observation, legal architecture menu, scripted proposal round-trip | CPU-only, zero training, no learning/performance claims or rewards; preserves graph-varied shapes | Pending |
| IR-12 | Paper drafting and independent evidence review, literature feature matrix, diagrams and reproducibility | Only audited quantities, clear contribution vs FINN/HIDA/ScaleHLS/CIRCT, limits and released-source/IP review | Pending |

## Critical-path order for fast scientific delivery

1. **First:** IR-01 + IR-02; establish whether compositional architectural interfaces actually differ from present implementation. If existing system already handles them, isolate precisely what is missing and do not claim invention.
2. **Minimal implementable contribution:** IR-03 + IR-05 on two-layer MatMul/Gemm. IR-04 only if compatible conversion semantics can be correctly implemented and fully tested.
3. **CPU evidence before HLS:** IR-06–08, then IR-11. Generate negative cases and independently check both requested mapping and actual generated source.
4. **Small native study:** IR-09. Keep tool and device stable; verify HLS timing only at HLS evidence level.
5. **Publication:** IR-10 as capacity allows, IR-12 after credible results exist.

**Stop/go criteria:**
- If graph composition adds no capability or verifiable invariant beyond existing systems, revise the scientific thesis before presenting a novelty claim.
- If numerical/structural correctness fails, no native campaign.
- If IR-09 yields only source-diff and no meaningful architecture diversity, report that limitation; do not assert superior hardware results.
- If external comparisons are unfair or cannot be reproduced, keep them out of performance claims.
- If claims require physical Fmax or implemented board numbers, schedule a separately justified E4/E5 campaign; no substitution of CSynth estimates.

## Paper-figure dependencies

- Fig. 1: Functional graph → operator-local architectures → typed interfaces → Lowered structures → HLS (IR-02/05).
- Fig. 2: One source graph, contrasting legal independently selected per-operator architectures, with traceable interface resolution (IR-03/05).
- Fig. 3: Rejection/compatibility and source-level conformance funnel (IR-07).
- Fig. 4: Distinct effective architectures vs requested configs (IR-08).
- Fig. 5: Qualified HLS latency/area and numerical trade-offs, where measured (IR-09).
- Table 1: Prior-art capability-specific comparison, with citations and no unsupported absence claims (IR-12).
- Table 2: Independent operator/graph composition legality matrix (IR-07).
- Table 3: Native original-HLS evidence matrix (IR-09).
- Small supplementary paragraph/figure: training-free RL-compatible typed interface (IR-11), no accuracy gain claim.

## RL freeze and native source-pinning

- **Allowed:** currently already-running L0 experiment finishes in its unmodified local checkout; retain immutable native receipts; after completion, run read-only audit/report for that exact source revision. This does not promote RL to an active research workstream.
- **Frozen:** implementing PPO/A2C/REINFORCE/SAC, new environment/agent code, new RL experiments, new training runs, RL-related paper claims for the compiler-IR contribution.
- **Git:** compiler-first source and documentation are committed to remote main; frozen RL modules are unchanged. An active native `gym_study` checks local `git rev-parse HEAD` per step; pushing remote docs is not an implicit local code update. **Do not `git pull` or modify tracked files in the active workstation checkout until L0 and its source audit are complete.** After that, pull `main` for the IR paper. The user explicitly requested main-branch planning; no feature branch is used.

## Completed first steps (compiler-only; no RL modifications)

- Git backup branch `backup/rl-environment-frozen-2026-11-01` points to exact pre-paper frozen source SHA `5221094af585bd501e842deca94d19c2289bb628`; it includes the complete `rl_environment/` tree and is distinct from active remote `main`.
- [Read-only source capability audit](SOURCE_CAPABILITY_AUDIT.md): current compiler **already supports** branched graphs, operator-specific P/PE/SIMD and host-source `Phi` tracing; new paper must not claim these as newly invented. Existing HLS emitter enforces one global numeric format and one model input/output.
- `src/fpgai_research/architecture/interfaces.py::inspect_array_interfaces`: prospective E0 source-boundary inventory for existing flat-tensor-array execution, with explicit blockers for mixed operator numerical formats and unsupported top arity. **This is not new streaming, adapter insertion or graph-interface legalization.**
- `tests/test_architecture_interfaces.py`: compiler-only regression cases committed. **Not yet run on user's workstation**; do not mark IR-02 or paper experiments finished.


## Higher-priority novelty gate for IR-02–IR-05 (supersedes any metadata-only interpretation)

New design source: [CST-Model IR novelty specification](COMPOSITIONAL_SPACETIME_IR_NOVELTY.md). An **architectural boundary report** by itself does not satisfy the first-paper novelty. Add a typed tensor **production/consumption event relation** that is derived from operator Phi/actual output ownership, records a symbolic availability order, and controls a genuine implemented intermediate storage/layout alternative in HLS. Use a nondegenerate M×N intermediate (M,N>1), retain exact logical element identity, and reject unimplemented streaming/casting/buffering. Evaluate against current row-major-array and restricted uniform-layout controls. Compiler invariants must be independently checked using actual generated-host-C++ execution, numeric agreement and small original-Vitis evidence. If no new executable graph-boundary behavior appears, change the novelty thesis instead of claiming a novel IR merely by adding fields. No RL training, agents, or environment modifications.


## IR-02–IR-08 implementation checkpoint — 2026-11-01

**Source committed directly to main; fresh user-workstation tests and all new native HLS runs are pending.** This is not a claim that the entire Model IR or paper is finished.

| Sprint | Narrow source-level result | Qualification still needed |
| --- | --- | --- |
| IR-02 | \`model_ir/composition.py\`: immutable \`TensorExchange\` / \`ComposedModelIR\`, per-operator Phi reference, consumer provenance, static 2D index bijection. \`model_ir/row_lifetime.py\`: a typed row-epoch lifetime contract with reduced scratch and whole-row availability semantics. | Independent executed host traces; general tensor/stream composition still unsupported. |
| IR-03 | Two MatMul operator composition, separate P/PE/SIMD/tile/fold requests; forbidden formats, unsupported layouts and invalid row schedule fail closed. | Tests must pass, generic/branched row-stream cases not supported. |
| IR-04 | Real two-layout producer-write/consumer-read rewriting and separate row-lifetime scratch lowering. **No cast, stream converter, FIFO or mixed-precision adapter**. | Full Vitis qualification; do not imply physical memory behavior. |
| IR-05 | \`codegen/hls/emitter.py\` now has an opt-in noncanonical intermediate tensor layout, preserving the no-override source; separate \`codegen/hls/row_interleaved.py\` emits an actual row-interleaved loop order and row scratch; \`validation/cpp_phi_trace.py\` accepts explicitly bound alternate emissions. | Host numerical/Phi/row-timeline tests committed, not yet run on workstation. |
| IR-06–08 | Source-only restricted vs alternate layout campaign \`experiments/cst_ir_campaign.py\`; whole-producer vs row-lifetime campaign \`experiments/cst_row_campaign.py\`; source digests and bounded element exchange witnesses. | Campaigns not run, no observed case counts/latencies. |
| IR-09 | Not started. | Controlled original Vitis CSim + CSynth required, after host/CPU gates and original L0 audit. |

**Highest-priority next checks:** execute all new tests in a separate clean compiler-only checkout (or once L0 audit completes), repair failures, run create-only CPU campaign, independently audit emitted C++ and numeric outputs, then source-pin a small prospective native campaign. [Runbook](RUNBOOK.md).

**Novelty gate:** A mere column-major layout is clearly not scientific novelty (Allo/StreamTensor precedents). The bounded row-lifetime mapping prototype adds **a real temporal storage aliasing rule with actual different loop order**, but novelty versus existing hierarchical/dataflow approaches is **not yet established**. [Prior-art and go/no-go assessment](PRIOR_ART_NOVELTY_GATE.md).


## Extended compiler implementation checkpoint — IR source and native evaluation integration

- \`compiler/cst_pipeline.py\`: opt-in one-entry-point compiler research API, explicit \`whole_tensor\`, \`column_major\`, or \`row_lifetime\` policies, preserving original \`compile_architecture\` and all source-pinned default code generation.
- \`validation/cpp_row_trace.py\`: actually executes host C++ and verifies row production/consumption event ordering rather than trusting Model IR metadata.
- \`tests/test_cst_ir_mutations.py\`: negative tests for wrong intermediate indexing and corrupted executed temporal events.
- \`experiments/cst_native_campaign.py\`: create-only matched triplet original-HLS package, explicitly bounded Vitis runner and read-only independent source-attested audit; \`tests/test_cst_native_campaign.py\` fail closed on tampering and unapproved vendor calls.
- Existing compiler-only CPU campaign modules remain separate; no RL modules have been changed.
- **Status:** source committed to \`main\`, but CPU/GCC tests and original licensed Vitis HLS remain **not run**. This checkpoint is not paper-ready, does not establish novelty precedence, and is not native timing closure.

**Immediate acceptance gates:** pass the full compiler regression and new mutation suite, run both create-only CPU campaigns, verify source-diff and numerical correctness, independently qualify native HLS cases. Only then update manuscript claims. For full source plus native command sequence use [RUNBOOK.md](RUNBOOK.md).


## Additional paper-scorecard and graph fan-out coverage

- Native comparison alternates **uniform** and independently configured **heterogeneous per-operator \`Phi\`** across the 12 default source-pinned cases (four matched 3-policy groups).
- \`test_compositional_model_ir.py\` includes a fan-out MatMul graph with two differently mapped consumers reading one alternatively laid-out intermediate tensor; actual host-output comparison pending.
- \`experiments/cst_native_report.py\` produces create-only paired HLS source/cycle/resource tables without fabricating missing metrics, with \`tests/test_cst_native_report.py\`.
- E2/E3/E4 native timing/resource results, complete CPU pass count, and definitive prior-art novelty **remain outstanding**.


## Three- and four-operator chain extension and experiment automation

- **IR-03/05 expanded:** \`model_ir/row_chain.py\` provides typed bounded row-lifetime composition for 2–8 sequential MatMul operators, one row scratch per internal tensor, and independent per-layer architectural Phi. \`codegen/hls/row_chain.py\` emits actual row-ordered generated HLS C++ rather than detached metadata; \`compiler/row_chain.py\` and \`compiler/cst_pipeline.py\` expose it opt-in as \`row_chain\`. Default HLS and existing two-op code remain available.
- **E1 semantic checks:** \`validation/cpp_chain_trace.py\` independently compiles and runs generated C++ and verifies row-ready/row-consumed events for each tensor boundary. \`tests/test_cst_row_chain.py\` compares generated source to NumPy reference and compiled C++ Phi coordinates on three- and four-operator networks, with negative chain/tiling tests. No native FPGA evidence is implied.
- **E1 CPU campaigns:** \`experiments/cst_chain_campaign.py\` enumerates variable 3–4 operator networks with uniform/heterogeneous local Phi; \`experiments/cst_cpu_report.py\` generates create-only source-pinned paper tables from actual array, row and chain CSV/JSON and rejects mismatched study revisions. Continuous CI archives every campaign and automatically prints their original summaries.
- **CI evidence:** [Reproducibility ledger](REPRODUCIBILITY_LEDGER.md) records confirmed initial 96/18-case source/GCC campaign and 808-test full regression checkpoints. Recent chain campaign and full regression counts are **not** asserted until the newest CI run is independently confirmed.
- **Remaining:** original licensed Vitis CSim/CSynth; numerical acceptance and real HLS timing/resource constraints; broaden beyond sequential MatMul only if essential; fair closest-prior-art functionality comparisons; finalized paper and Radboud release/IP review.


## IR-03 / IR-05 independent liveness check

- \`model_ir/row_chain_liveness.py\` adds a bounded source-independent finite check: exact producer/consumer element events, no write to a live scratch address, every read after the matching write, complete per-row release, typed storage capacity and source-specific SHA. The opt-in \`compiler/row_chain.py\` now requires \`EXHAUSTIVE_BOUNDED_LIVENESS_VERIFIED\` **before source generation**.
- \`tests/test_cst_row_chain.py\` includes positive and deliberately corrupted/too-large boundary constraints; \`experiments/cst_chain_campaign.py\` now records each successful E0 witness hash and coverage status.
- This is **not** E2 Vitis CSim, verified RTL or hardware RAM savings. All native/paper speedup claims remain blocked until original Vitis results are independently audited.

## Scientific validation checkpoint — actual results (2026-11-02)

**Evidence tied to immutable SHA `adb28108c7f13e52ae6c3b7ff15b87675762bf6d`.**

- Finished and verified E0/E1 research sprint: a first-class candidate row-block schedule and bounded, counterexample-producing tensor-lifetime legality check, actual B-slot C++ HLS lowering, temporal/iteration traces and scientific controlled interventions.
- **844 full tests passed, two skipped; 79 focused scientific tests passed**, all in the green GitHub Actions runs recorded in [the evidence ledger](REPRODUCIBILITY_LEDGER.md).
- Completed schedule study: 30/30 correct legal controls; 138/138 correct rejection of *five prespecified synthetic fault classes*; 6/6 selected additional host checks; 3/3 additional row traces. The physical claim remains explicitly **not measured**.
- Prepared two independently source-pinned original native HLS experiment families (12 tensor-interface cases; 6 row-block cases), zero vendor calls. Manual licensed-runner workflow supports both without changing frozen RL.
- Scientific publication work, not additional arbitrary features, is now gated by **E2/E3 original Vitis reports and an artifact-level comparison** to PACT'26 ST-Flow/Allo/StreamTensor. H1–H3 do not imply firstness or speedup. **No manuscript finalization until those gates close.**


## 2026-11-02 final source-completion sprint (new code, original execution pending)

Original user workstation **1182/1182 tests** and **71/71 focused**
passed at main `ad748e8b`; original `v6` physical input was
prepared as 12/12 source-matched `PREPARED_NOT_RUN` cases,
**zero** actual Vivado processes. This was a preparation milestone,
not a newly validated E4 result.

New code on main, pending workstation test: bounded case-by-case
foreground `cst_research_completion`, source-matched E3/E4
`cst_paper_gate` and genuine negative-friendly
`cst_phi_counterfactual_effects` on 24 original source-executed
two-reader case rows. Newly added tests check create-only behavior,
resume/no rerun, failing-case stop, source identity, clean/allowed
warning separation, fake owner-hash effects and unsupported no-op
Phi interventions. A successful syntactic import, source commit,
mock vendor log or full code unit suite is not physical proof.

**Next stage:** run focused/full CPU tests, inspect read-only v6 plan,
one original case_000, then continue precommitted physical cases
only when the independent auditor classifies the first case as
strict E4 or disclosed-warning exploratory route. Preserve all
original data. Run the actual source Phi counterfactual campaign,
then execute pinned upstream peer-compiler artifacts and real
trained-model tasks. Scientific firstness, external validity and
manuscript submission remain unqualified until evidence exists.

Exact commands: [FINAL_RESEARCH_EXECUTION.md](FINAL_RESEARCH_EXECUTION.md).


## 2026-11-03 final source orchestration checkpoint: C1/C2/C3

C1 mapping-conditioned CST Model IR and C2 certificate-backed
source legality stay PRIMARY; C3 policy-independent
RL-compatible state/action/graph/mask interface is SECONDARY.
Both source-only R3/R4 and Phi actual-last-use counterfactual
are now required in quick/paper `compiler_ir/series.py`
and original evidence hashes are cross-stage verified.
Unit tests were added for the series steps and scientific
evidence-index release gates; all subsequent tests remain
to be executed on the user's workstation. The latest
earlier 1221-pass original transcript is **not** a test
result for this new revision.

No claim of improved RL training performance, hardware
resource reduction or firstness over ST-Flow/Allo/HIDA/
StreamTensor is authorized from compiler interface alone.
The definitive experiment commands are in
`FINAL_RESEARCH_EXECUTION.md`.
