# FPGAI compiler paper — precise novelty thesis and proof obligations

**Decision date:** 2026-11-02. **Status: a specific, falsifiable proposed scientific contribution, NOT demonstrated research priority.** Scientific scope: \`fpgai_research\` compiler paper; RL/Gym remains frozen. We will *not* present the existence of an IR, FPGA acceleration, HLS C++ generation, per-layer PE/SIMD, streaming, memory reuse, or type checking by itself as novel.

## One-sentence thesis (paper target)

> **A Compositional Space–Time Model IR can make independently chosen operator-level space/time mappings composable by deriving explicit tensor-value availability and storage-lifetime obligations, synthesizing legal physical exchange plans, and attaching independently checkable source-level refinement certificates to each inter-operator boundary.**

The **potential contribution** is the *mapping-conditioned, certificate-carrying operator-boundary refinement*, NOT the particular row buffer, its size, or the name CST-Model IR. An exact first-of-kind claim requires artifact-level related-work analysis.

### Candidate title (not final)

**CST-Model IR: Mapping-Conditioned Tensor Lifetime Contracts for Verifiable FPGA Accelerator Composition**

Scientific focus: compiler IR semantics, architecture-conditioned legality, memory planning and compositional source-level refinement. *Not* AI search, multi-vendor product engineering, general RTL formal verification, learned RL, or blanket performance superiority.

## Why the old “novelty” statement is insufficient

Allo (PLDI 2024) already combines compute, memory, communication and datatype customizations across functions with type-safe composition. HIDA (ASPLOS 2024) already has hierarchical dataflow IRs and cross-kernel optimization. StreamTensor (MICRO 2025) already has a first-class iterative tensor type for streaming layouts/fusion/buffering. Dato (2025) already encodes task streams and data sharding as types. ST-Flow (PACT 2026) already implements spatial/temporal/hybrid FPGA mappings, reuse-aware sharing, stream layout conversion and Vitis simulation. Calyx/Piezo already distinguish static/dynamic accelerator scheduling. **Wavelet/Let It Flow (PLDI 2026) already develops verified dataflow compilation with a capability type system and Lean proofs.** “We are the first compiler to verify dataflow,” “we introduce spatial-temporal HLS,” or “we type tensor interfaces” would be false or at least unjustified.

Our narrower testable difference is whether **per-operator iteration mapping \`Phi\` and tensor value availability/reuse are represented in ONE compositional contract and checked AGAINST independently emitted source semantics** for multiple legal mapping choices. This needs direct experimental comparison, not just terminology.

## Formal research object

Given a Functional tensor/operator DAG \`G=(V,E)\`, a logical iteration domain \`I_v\` and independently selected operator mapping \`Phi_v : I_v -> (space_v, logical_order_v, numeric_v)\`:

- \`R_v(x) subset I_v\`: all reduction/contribution iterations required to produce logical tensor element \`x\`. For MatMul \`Y[m,n]=sum_k A[m,k]*W[k,n]\`, \`R_v(m,n)={(m,n,k):0<=k<K}\`. The result is **not available until all contributions have completed**, even when some local PE has executed partial work.
- \`ready_e(x)\`: *logical* producer-commit event, constrained to occur after every \`Phi_u(i)\` event with \`i in R_u(x)\`. This is a partial-order event, not a measured FPGA cycle.
- \`read_e(x,j)\`: each consuming event of \`x\` by the \`j\`th reader, derived from \`Phi_v\` and the consumer's declared tensor access map. Fan-out can produce multiple reads; don't identify a tensor with a single consumer.
- \`release_e(x)\`: the event/condition after **every** required reader has completed. A producer cannot reclaim the storage slot from just one reader while another reader remains.
- \`addr_e(x)=(bank,offset)\`: concrete logical-element to physical source address mapping, with an implemented layout or buffer-reuse policy.
- \`N_e\`: exact numeric preservation obligations under the declared fixed-point arithmetic/rounding/cast policy. Do not silently assume host float equals \`ap_fixed\`.

**Proposed first-class tensor edge certificate:**

\`\`\`
C_e = (source_tensor_identity, producer_Phi_projection,
       consumer_Phi_projections, reduction_completion,
       ready, reads, last_use_or_release,
       bank_and_address_map, numeric_policy,
       source_refinement_witness, evidence_level)
\`\`\`

An operator's \`Phi\` is **not** merely recorded as a SHA digest; its iteration-space/ownership and *logical completion relation* must actually constrain \`C_e\`.

### Exact semantic obligations

For every required logical element \`x\` and each consumer \`j\`:

1. **Ownership/completeness:** the correct unique logical value is produced, all its reductions precede \`ready_e(x)\`, and the producer's actual generated-loop writes match the expected value owner.
2. **Dependence:** \`ready_e(x) ≺ read_e(x,j)\`; the consumer reads precisely the logical tensor element it requests, including layouts and tails.
3. **No stale alias:** if \`addr_e(x)==addr_e(y)\` for two different live values, their intervals from \`ready\` through final \`release\` **must not overlap** in the declared execution order.
4. **Fan-out:** release is safe only once **all** consumers have finished; a second consumer must prevent premature reuse.
5. **Numeric/shape semantics:** the declared operator graph, shape, numeric format and conversions are preserved, or the candidate is explicitly unsupported.
6. **Refinement:** the generated HLS *source* write/read/address/event traces conform to the contract. This is neither a theorem about arbitrary C++ nor proof of RTL schedule, timing closure or physical BRAM.

The intended verifier returns either a bounded **certificate** (complete source/IR mapping identities, checked events, safe memory assignment) or the first concrete failure (tensor/element/reduction, producer/consumer event, physical address and live owner). Bounds, unsupported operations and unverified large cases are never marked valid.

### Proposed bounded soundness proposition (TO PROVE, not established)

**Given** individually semantically correct operator lowerings, a complete and correctly extracted local logical event order from \`Phi\`, exact logical-to-physical source-address refinement, a schedule that honors every certified edge and numeric policy, and a finite static graph, **then** no emitted logical consumer read observes an unproduced/stale value or an overwritten still-live tensor element; therefore the graph's evaluation matches the Functional IR under the *same arithmetic model*.

Proof plan: induction on accepted logical events; operator base cases from local reduction-completion/output-ownership checks; inter-operator step from ready-before-read and non-overlap of live physical addresses; fan-out from last-reader release. This is an abstract/source-level theorem unless the generated C++ refinement relation is also proved. **No formal verification of RTL is currently claimed.**

### Candidate algorithmic result

For a *fixed valid total logical event schedule*, derive the live intervals of all intermediate tensor values. For a homogeneous scratch pool with arbitrary legal address assignment, minimizing slots is the classical interval-coloring problem: minimum slot count equals maximum simultaneously live intervals. That identity is **not itself novel**. The research algorithm must instead handle the *extra constraints induced by mapping-specific banks, reductions, multiple readers, source tiling and actual emitter memory addressing*, and either synthesize a legal mapping or return a concrete failed obligation. Compare with source-matched whole-tensor/row-block contracts and existing layout/buffer allocation tools.

Do not claim exact optimality for a more general bank/port-constrained problem without a proof or verified search. A baseline \`B=1,2,M\` chosen by hand is not an automatic algorithm.

## What the repo actually implements NOW (independent code audit)

| Component | Actual current status | Why it does **not** yet prove our main thesis |
| --- | --- | --- |
| \`model_ir/phi_refinement.py\` and operator ownership | Real bounded per-op HLS loop-to-Phi coordinate/refinement checks for supported compute operators | Each operator is verified locally; no value-level cross-operator certificate is derived from Phi |
| \`model_ir/composition.py::TensorExchange\` | Typed per-boundary identities, layout, producer/consumer **Phi SHA hashes**, whole-producer barrier and verified static 2D address bijections; alternate real column-major C++ lowering | Phi is currently attached mostly as metadata; actual consumer availability is not computed from producer/consumer iteration semantics |
| \`model_ir/row_chain.py\` | Real row-buffer IR and explicit sequential chain constraints for MatMul/Gemm/Relu/leading Flatten | Restricts graphs to single-reader linear chain; no fan-out and no general reduction-completion relation in IR |
| \`model_ir/temporal_schedule.py::check_schedule\` | Real finite independent proposed **(row,stage)** checker, slot aliases and concrete counterexamples | The checker uses atomic **whole-row stage completion** and fixed modulus slots; **it does not inspect local Phi coordinates** or prove partial-row/reduction completion |
| \`validation/schedule_reference_oracle.py\` | Separate constructive oracle, 3,600 exhaustive six-event permutations agreeing with checker | Independent *algorithm*, but same restricted atomic-stage specification; not a general graph proof |
| \`codegen/hls/row_chain.py\` and \`validation/cpp_chain_trace.py\` | Actual B-row C++ buffers and executed row-ready/consumed traces | Trace resolution is whole row, not arbitrary logical element/PE/tile/last consumer |
| HLS native experiment series | Source-pinned and CPU validated, with opt-in original Vitis | Native matched CSim/CSynth not yet run; source scratch reduction is not measured BRAM reduction |

**Crucial finding:** current code is a **good scientific prototype**, but it **does NOT yet implement the proposed primary novelty**: an operational \`Phi -> readiness/last-use/physical-address\` derivation for independent operators plus a fully source-refined certificate. This is a technical blocker, not a documentation issue.

## Distinguishing experiments and *falsification*

1. **Mapping-conditioned counterexample (required).** Hold Functional IR and boundary storage request constant. Change **only** a legal local operator mapping that changes logical output completion/consumer access order, then propose the *same early-consumption* plan. An actual Phi-aware contract must classify appropriately, citing a missing reduction or data-availability event; the existing row-atomic checker cannot distinguish these cases. Reject the novelty claim if nothing beyond a Phi hash participates.
2. **Independent producer/reduction ownership (required).** Show at least one nontrivial tiled/reduced MatMul where partial sums are not ready, a legal complete-reduction schedule and a deliberately premature-read mutant. Source trace must prove that actual memory commits occur after all required reductions.
3. **Last-consumer lifetime and address reuse (required).** Use a real branch with two consumers of the same intermediate; check one illegal early overwrite after reader 1 but before reader 2, and a valid reclaim after reader 2. No fake fan-out metadata without executable source.
4. **Automatic memory-plan comparison (required for algorithmic claim).** For fixed local mappings and constrained schedules, derive and **actually emit** more than one legal address/buffer plan, together with concrete witnesses or failure certificates. Compare inferred capacity with brute-force bounded optimal solutions and prespecified baseline memory allocation; separate source size from original HLS BRAM.
5. **Numerical and source-refinement tests.** Independently evaluate original Functional IR, actual generated GCC C++ with exact address/event tracing, and original Vitis fixed-point CSim when available. Change only one dimension at a time (local Phi, schedule, bank capacity, layout) with matched model, numeric policy and source revision.
6. **Original HLS matched QoR.** Run source-hashed, matched original Vitis CSim and CSynth on small and medium supported graphs; preserve failures, cycles, LUT, FF, DSP, BRAM, requested clock and timing report. Do not claim routing or physical Fmax from HLS estimates. No a priori assertion that our approach is faster.
7. **External novelty falsification.** Pin Allo, HIDA, StreamTensor, Dato, ST-Flow, Piezo/Calyx and Wavelet paper/artifact versions. Inspect whether they already offer a **functionally equivalent** *mapping-conditioned, value-level ownership/lifetime/refinement certificate* and native source realization. Any equivalent implementation reduces or voids our claimed novelty, even if it uses different names. Their support/absence must be verified, not inferred from README keywords.

### Essential ablations

- **A0:** original local Phi + conservative whole-tensor boundaries, same backend.
- **A1:** independently selected local Phi + typed layouts/contracts without value-dependent timing.
- **A2:** existing row-atomic schedule checking with fixed B (current state).
- **A3:** proposed Phi-derived tensor-value availability/last-use checking, **fixed** buffer assignment.
- **A4:** proposed full certificate + source-refined bank allocation under mapping/port constraints.

Report novel executable architecture coverage, illegal-candidate detection, checker/certificate cost, inferred safe capacity vs bounded optimum, numeric correctness, original HLS QoR when measured. No arbitrary grouping of tests into an invented performance benchmark.

## Scientific kill conditions

**Do not submit with “novel Model IR” as the central claim if:**

- the composed boundary still contains only a producer/consumer Phi **hash** rather than a semantic projection of mapped iterations,
- the checker still relies only on atomic whole-row stages while our claims involve element/tile readiness,
- the only new physical code is manually selected B-row scratch arrays or column-major storage already representable in previous systems,
- a single-consumer chain is presented as proof of general DAG composition,
- correctness is inferred from assertions written by the same emitter rather than independent reference/actual executed source traces,
- source scratch capacity is described as real BRAM, or requested clock is described as achieved Fmax,
- related work implements the same exact relation and equivalent acceptance behavior.

If these conditions persist, retitle the work as a **bounded row-lifetime HLS compilation/validation study**, without claiming a broadly new IR.

## Source references to be evaluated against the exact claim

- Allo, PLDI 2024: https://arxiv.org/abs/2404.04815 and https://github.com/cornell-zhang/allo
- HIDA, ASPLOS 2024: https://doi.org/10.1145/3617232.3624850
- StreamTensor, MICRO 2025: https://arxiv.org/abs/2509.13694
- Dato, 2025: https://arxiv.org/abs/2509.06794
- ST-Flow, PACT 2026: https://github.com/UCLA-VAST/ST-Flow (artifact tag PACT26; validated 2024.2 HLS)
- Calyx static scheduling: https://docs.calyxir.org/lang/static.html
- Piezo, OOPSLA 2024: https://doi.org/10.1145/3689790
- Wavelet/Let It Flow, PLDI 2026: https://doi.org/10.1145/3808263 and https://github.com/plum-umd/wavelet

This document establishes the **research hypothesis and explicit next-code gates now**, not novelty priority or results that have not been measured.

## Second proposed contribution: RL-ready but policy-independent certified IR (2026-11-02)

In addition to the primary mapping-conditioned tensor-lifetime compiler semantics, the IR **must expose those same semantic obligations as a canonical, machine-checkable optimization-state and transition interface**. The intended contribution is not an RL framework or a GNN architecture: it is the **single shared certificate relation** that both (a) validates compiled tensor exchange and (b) admits/rejects proposed architecture edits and produces sound legal action masks for RL, classical search and human edits.

See [RL_READY_IR_NOVELTY_AND_EXPERIMENTS.md](RL_READY_IR_NOVELTY_AND_EXPERIMENTS.md) for typed state/action design, variable-sized graph observations, source qualification, R1–R6 experimental controls, real-HLS reward separation and prior-art kill criteria. CompilerGym (CGO 2022), IronMan (GLSVLSI 2021), RL4ReAl (CC 2023), DRiLLS (ASP-DAC 2020) and CASCADE (ISCAS 2026) already show various compiler/RL/GNN/hardware workflows. Therefore **RL compatibility by itself is not a novelty claim**. Only a demonstrated, correctly derived Phi-conditioned *certificate-backed* action transition that is equivalent to actual compiler/source legality could qualify as a distinct representation result.

**Implementation status:** not yet operational. Current `TensorExchange` holds Phi hashes and the row checker does not derive its action legality from Phi coordinates. No RL policy will be trained or modified for the compiler-first paper; the frozen RL environment remains unchanged. Search/RL-readiness can be evaluated with policy-independent exhaustive/random/greedy action sequences and numerical source conformance before later training-based experiments.

## RL-ready compiler interface implementation checkpoint

The policy-independent state/action architecture is now represented in `model_ir/decision_interface.py`; candidate actions are actually source compiled through the same existing `resolve` + `compile_cst` gates that a manual edit would use. It emits immutable, deterministic source-bound receipts, exact finite masks, and weight-private graph observations. A restricted independent experiment over PE/SIMD/row-block choices is source-prepared and incorporated in the one-command E0/E1 research suite.

**Important nuance:** for this initial subset, the mask is *source admissibility with whole-row-stage checking*, **not the proposed mapped-`Phi` tensor-element completion and last-reader release predicate**. Thus our precise primary novelty is still a research hypothesis; the new interface is an operational substrate, not proof of firstness or RL efficiency. See [CERTIFIED_DECISION_IR_SPRINTS.md](CERTIFIED_DECISION_IR_SPRINTS.md) for complete L0–L9 gates.

## 2026-11-02 L5e check against the proposed thesis — three independent Phi contracts

The **new verified bounded evidence** is not a new streaming operator or an independently invented interval allocator: one static five-op Functional DAG `MatMul(X,Wp)->H; MatMul(H,Wc0)->A; MatMul(H,Wc1)->B; Add(A,B)->G; Relu(G)` has **three independently resolved MatMul Phi mappings** (producer and both readers). The new `model_ir/branch_matmul_contract.py` derives every required read token `(tensor H[m,k], consumer operator, consumer output[m,q])`: **2MQN** logical source reads, compared to `MN` produced H values, and accepts storage release only after every individual token was observed. Reductions/commits are separately checked against source-loop mapping coordinates. Producer/reader0/reader1 independently emit the same shared implementation-lowering MatMul loop template; `validation/cpp_branch_matmul_trace.py` checks actual executed GCC reductions, read owners, source order, commits and alias reuse.

**Original scientific evidence anchor:** immutable compiler-series source `425772d6be0f26f036ff48564f5b6fb571b94186`; [GitHub Actions original run 41442961276](https://github.com/umutcanaltin/fpgai_research/actions/runs/41442961276), **26/26 quick and 26/26 paper-profile stages passed** (zero blocked/failed). Prespecified source campaign: three shapes × four independent reader mapping policies × two layouts = **24/24 E1 host-float numerical and executed source-trace checks**, with 12/12 legal disjoint/reuse source-storage comparisons. Actual Vitis calls: zero in this new campaign.

**Correct interpretation:** a narrow *mapping-conditioned, read-cardinality and last-use certificate* with independent source refinement, satisfying a small instance of our proposed composition relation. The scalar ACK abstraction is provably insufficient for MatMul consumers requiring multiple per-output contributions. No hardware schedule, bank-port certification, per-element arbitrary-DAG source compiler, learned search, formal RTL proof, hardware performance superiority or priority relative to Allo/HIDA/ST-Flow/StreamTensor has been established. Exact contiguous-base reuse in this sequential schedule is not by itself a new algorithm.

**Falsification/next work:** (i) vary K/Q and producer/reader mappings independently, including source-changing tiling and reduction order, (ii) construct explicit mismatching local Phi owners and early-last-use reads, (iii) qualify arbitrary multi-reader graph certificates with true bank/port constraints and `UNKNOWN` rather than false legal states, (iv) run original Vitis CSim/CSynth and compare source-matched ablations and external pinned prior-art artifacts. RL interface must share this verifier; RL learning remains frozen.

## New experimentally tested implications (2026-11-02), not a novelty verdict

At immutable source `a8e03eafd68262c9414e6e9d579c33ac163c67d2`, the [29/29 quick and paper CPU/GCC evidence series](https://github.com/umutcanaltin/fpgai_research/actions/runs/41446968133) adds: a finite exact source-array base allocator under mapped multi-reader individual reads, a seven-case obligation ablation demonstrating that Phi-owner and complete multi-read completion catch failures omitted by deliberately weaker internal proxies, and declared abstract bank-port models with mandatory UNKNOWN for absent cycle schedules. The full certificate reports no false accepts **against its own finite spec**, not an independent formal theorem for arbitrary C++/RTL. Original Vitis tool calls: 0.

This is a useful bounded *proof-obligation refinement and falsifiability* result. It does **not** establish firstness over Allo/HIDA/ST-Flow/StreamTensor/Wavelet, general DAG source correctness, physical bank/port eligibility, or measured hardware QoR. Explicit inspected public artifact pointers and limitations are in [PRIOR_ART_PINNED_ARTIFACTS.md](PRIOR_ART_PINNED_ARTIFACTS.md).

## L6a new bounded falsifiable support for compositionality — 2026-11-02

A verified [source-pinned 30-stage original E0/E1 run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41450443712) at code SHA `c1298ea35bf526b743f39c910615aa72f29afadf` now includes graph-wide typed element/edge certificates and a real emitted/GCC-executed C++ refinement pathway on **three non-diamond 6/7/8 operator graph families**. The new `GraphEdgeWitness` carries concrete value readiness, every consumer-mapped read owner and index, global last use across readers, actual source array address, and independently source-attested witness identity. Not merely a producer/consumer `Phi` hash: ownership constrains per-element reads and a forged/missing owner is rejected. Complete 12/12 source GCC numerical and trace experiments are retained.

**Do not overclaim:** this is a **bounded static rank-two subset**, no general Conv/Concat/broadcast or graph formal theorem, no actual Vitis CSim/CSynth, qualified hardware RAM ports/clock, measured resource/PPA advantage, or external matched prior-art execution. Both this small graph research object and its sources may be covered by existing composition/dataflow work. The core thesis is contingent on full comparative tests and a sound generalized refinement account; [precise scope](GRAPH_PHI_BOUNDARY_CERTIFICATES.md).

## New source-verified nonidentity source access and independently falsified verifier, 2026-11-02

At immutable source `dbf6aade8228cbbe703037fb5108715cd0dd64f9`, [original 32-stage quick/paper series](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625112) includes 16 actual GCC/NumPy source-refined Add-broadcast/Concat graph cases and 8 independent NumPy semantic-check cases. Four intentionally synchronized faulty event-producer/lifetime-verifier access mappings were **all** accepted by that deliberately weaker correlated checker but **none** by an independently formulated NumPy index-grid semantic oracle. This establishes an actual finite *independence-of-reference* failure mode, **not** novelty relative to published compiler systems or formal hardware verification. See [proof scope](NONIDENTITY_GRAPH_ACCESS_AND_ORACLE.md).

The target scientific claim is a *single compositional model IR contract across independent per-operator Phi plus semantically correct tensor input access maps and source-checkable readiness/last-use*; the reference oracle must not simply consume the same formula as codegen. We have not verified general operator coverage or achieved physical performance, and no external peer artifacts were executed.

## NCHW Conv E0 / actual reader-LOAD E1 scientific evidence — 2026-11-02

The [source-pinned 33-stage full experiment series](https://github.com/umutcanaltin/fpgai_research/actions/runs/41456356235) at commit `6af10f9e2ec71810181b51560d227d8dba133303` now includes an eight-case bounded NCHW Conv branching experiment with independent kernel padding/stride source indexing, per-operator requested Phi read owners, logical per-value last reader, actual GCC-generated Conv reader kernel LOAD indices and NumPy numerical equivalence (all **8/8** qualified). This is a supported **bounded extension** of the CST compositional boundary candidate, not a full hardware semantic proof or originality result. Source COMMIT order, non-Conv reader loads, real shared scratch reuse, actual Vitis fixed-point/hardware ports and executed outside prior art remain open. Do not present an ordinary Conv lowering, tensor liveness or static kernel indexing as a scientific first. [Proof scope](CONV2D_NCHW_BOUNDARY_PROOF.md).

## NCHW graph-wide compositional scope (bounded) — original 34-stage evidence

Verified source `fcb1f621bb6c0eaadae8f2206e823f59bd2bf21c`, [34/34 quick/paper run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339211): a typed **general topological** NCHW graph subset now binds operator-local Conv `Phi`, nonlinear source-access geometry for Conv/MaxPool/broadcast Add/channel Concat, per-value multi-reader lifetime and exact *logical* scratch addresses. An independent NumPy semantic oracle and eight genuinely executed GCC sources independently check numeric outputs/actual Conv intermediate kernel-load maps; all 16 rows retain their provenance. This is stronger evidence of a *compositional boundary representation* than a fixed hand-written diamond, **not proof of novelty over ST-Flow/HIDA/Allo/StreamTensor/Wavelet**, and not an RTL proof. Complete real source event schedules, materialized aliasing, Vitis physical truth and external comparable baselines are still missing. [Bounded proof definition](GENERAL_NCHW_CST_EDGE_CERTIFICATES.md).

## Original per-intermediate value attestation and materially shared C++ scratch — 2026-11-02

The [source-pinned original 36-stage experiment](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917065) at `319ab4e94a57019d0249dbe2254a925e748e4ace` establishes **source-observed post-operator tensor numerical refinement** in eight generated NCHW graph instances (50 snapshots/1,764 elements) and **actual generated source-level C++ scratch array reuse** in 8 matched disjoint/reuse pairs (16/16 numerical and 16/16 Conv load checks; 32 GCC runs, min 8 elements lower local C++ scratch capacity). This is a stronger bounded source refinement claim than merely computing abstract last-use offsets, but is **not** new optimal memory allocation, proof of complete actual load/store time semantics, native synthesized BRAM/clock/energy advantage, or demonstrated scientific priority over existing compiler/dataflow work. See [scientific acceptance and evidence levels](ACTUAL_NCHW_INTERMEDIATE_AND_SCRATCH_EVIDENCE.md).

## Source-observed compositionality strengthened; peer novelty NOT established

At source `77fe5dfcd96ad81ff8d71f36fe57cb88a47411e4`, [original 37-stage run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41480551778) now observes genuine GCC-produced NCHW source-array READ and WRITE events, not only E0 planned event schedules. The independent, input-sensitive NumPy access oracle accounts for ReLU's conditional double read and MaxPool's additional first-tap read. Twelve original GCC variants preserve correct access multiplicity, global causality and real scratch reuse. The native study [38-stage source-only preparation](https://github.com/umutcanaltin/fpgai_research/actions/runs/41481941961) pins 12 Vitis model/testbench/Tcl packages **without** executing CSim or CSynth.

This is evidence supporting a bounded, mapping-conditioned and independently checked *compositional source refinement contract*, **not a novel-firstness finding, formal arbitrary C++/RTL theorem, synthesized FPGA memory saving, or improvement over external prior art**. Comparator artifact matching, actual Vitis fixed-point/timing/resource, bank/port data, physical implementation and held-out trained models remain open. [Complete evidence gate](NCHW_SOURCE_ACCESS_AND_NATIVE_HLS_HANDOFF.md).
