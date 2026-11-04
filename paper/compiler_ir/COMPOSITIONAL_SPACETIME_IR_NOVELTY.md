# Proposed central novelty: Compositional Space–Time Model IR (CST-Model IR)

**Research design / not implemented:** 2026-11-01. This is a *falsifiable candidate contribution* for the compiler-first FPGAI paper, not an established research priority claim or measured result. Preserve existing Functional → Architectural → Lowered Model IR; add semantics only within/alongside the Architectural and Lowered levels. RL agent/environment remains frozen and backed up at `backup/rl-environment-frozen-2026-11-01`. Make compiler source updates on `main` as instructed, but never pull new `main` into the already-started source-pinned L0 local checkout until that study and its original-revision read-only audit finish.

## Why a stronger contribution is needed

**Established competitors:**
- **HIDA (ASPLOS 2024)** has Functional and Structural dataflow IRs, explicit buffer/stream/control representations, inter-task optimization.
- **Allo (PLDI 2024)** composes compute, memory, communication and datatype customization across functions with type safety.
- **StreamTensor (MICRO 2025)** defines an iterative tensor type system encoding streaming layouts and does fusion/layout conversion/buffer optimization.
- **Stream-HLS (2025)** performs multi-kernel dataflow architecture and global scheduling.
- **Calyx** separates structure/control and has static scheduling; Timeloop/MAESTRO/AutoSA formalize spatial/temporal mapping.

Thus “new three-stage IR”, “per-layer PE/SIMD”, “hardware-aware graph IR”, “typed tensor interfaces”, “stream layout conversion”, “architecture provenance”, “graph-level scheduling”, or “compiler-ready for RL” **alone** are NOT adequate novelty claims. Focus must be the **particular compositional relation linking per-logical-iteration operator resource/time coordinates to tensor value production, consumption and implemented exchange**, with machine-checkable refinement obligations.

Research claim to test: **CST-Model IR jointly captures operator-local computation mapping and tensor-boundary value availability/ownership, with an explicit compositional preservation invariant that remains checkable after HLS source lowering.** Do not claim precedence or a complete formal machine proof over C++/RTL; test incremental capability against a restricted-IR baseline and closest published systems.

## Formal objects (proposal)

Keep the immutable Functional Model IR `G=(V,E)` and per-op logical iteration domain `I_v`.

Existing operator mapping:

```
Phi_v(i) = (q_v(i), kappa_v(i), r_v(i), p_v(i),
            ell_v(i), f_v(i), g_v(i))
```

where numeric, tile, replica, PE, lane, folding/reuse and pipeline are preserved. `g` is an **architecture scheduling relation/request**, not measured issue cycles.

**New first-class tensor boundary contract for each functional edge `e=(u,v,tensor)`:**

```
B_e = (TensorIdentity, ProducerRelation, ConsumerRelation,
       AvailabilityOrder, PhysicalExchangePolicy,
       StorageIdentityOrLayout, NumericContract, EvidenceLevel)
```

- `ProducerRelation` maps the *logical output elements* of `u` to their owner/production events, **after** completing all required reductions. This is not equal to every MAC iteration. Map producer execution coordinates through `Phi_u` where defined.
- `ConsumerRelation` maps each input element used by `v` to the corresponding *logical element identity* and required consumption events, referencing `Phi_v` where meaningful. Supports fan-out and repeated reads.
- `AvailabilityOrder` is a **symbolic partial order**, not a fabricated HLS cycle number: a consumer read may occur only after its producing value is available in the specified storage/transfer mechanism. The current sequential-array HLS flow uses a conservative whole-operator completion barrier.
- `PhysicalExchangePolicy` is a typed choice with `array_materialized` supported today; `stream`, `repacked`, `buffered`, `mixed_precision_cast` are **unsupported options until there is actual matching emitter+verification code**. Do not infer need for adapter from PE/SIMD mismatch alone.
- `StorageIdentityOrLayout` must specify a real layout/access relation actually lowered to C++; adding a metadata-only `layout` field is **not** an architectural advance.
- `NumericContract` records equality or explicitly implemented conversion with tolerance/rounding/saturation obligations. Current emitter globally uses one `ap_fixed<W,I,AP_RND,AP_SAT>`; mixed per-op precision remains blocked.
- `EvidenceLevel` separates structural/requested E0, generated/host-executed E1, Vitis CSim E2, CSynth E3 and actual implementation/runtime evidence E4+.

## Proposed soundness obligations (for supported bounded finite cases)

1. **Tensor conservation:** For every required consumer read, exactly the correct logical producer/input/constant value is identified; fan-out may have multiple legal consumers.
2. **Reduction completion:** An output cannot be deemed available before all required reduction contributions have been accounted for; avoid conflating the operator iteration domain with output-write ownership.
3. **Temporal consistency:** The symbolic producer-available event precedes each corresponding consumer read under the selected exchange policy; do not derive exact cycles from HLS pragmas.
4. **Storage/layout fidelity:** The generated HLS source actually uses the declared layout and data-access transformation. Unsupported layout pairs produce a blocker; no silent conversion.
5. **Numeric preservation:** No implicit type change; implemented precision conversion must have independently checked numerical behavior.
6. **Refinement witness:** Per-op `Phi` and boundary contracts are associated with concrete generated source identities and independently checked host traces. Requested and realized/potentially unverified properties remain separately labeled.

This is a *contract specification*, not a theorem until a separately stated verifier and implementation satisfy it.

## What the existing repo already supports

- `model_ir/contracts.py`: graph tensors/operators and `ArchitecturalModel(functional, points)` with no explicit boundary relation.
- `architecture/resolver.py`, `architecture/scopes.py`: legal, independently chosen operator mapping `Phi`.
- `model_ir/phi_refinement.py`, `architecture/materialization_oracle.py`, `validation/cpp_phi_trace.py`: per-op spatial/fold source checks (already existing; **not new**).
- `model_ir/output_ownership.py`: output writer ownership enumeration (foundation for producer events).
- `codegen/hls/emitter.py`: ordered one-top-function source with flat C++ tensor arrays and one global `ap_fixed` type; different PE/SIMD across layers are already legal; no streaming or layout converter. 
- `architecture/interfaces.py`: new **E0 inspection only**, not a new first-class composition IR or realized converter.
- `experiments/branch_ir_study.py`: existing branched Conv→Add→Relu→pool path; graph composition/branching itself is **not** newly invented.

## Minimal novel *implemented* capability (research critical path)

**Experiment target:** A two-MatMul chain, e.g. `MatMul(1×3,3×5) → MatMul(1×5,5×2)` (plus controlled fan-out). MatMul left/right have distinct PE/SIMD/tile/reuse choices. Compare two fully implemented intermediate tensor exchange/layout alternatives for the same original network and numerical policy:

- **Baseline:** one canonical row-major flat array with a whole-producer completion barrier (already supported).
- **New IR:** explicitly assign a *different real tensor storage/layout/access relation*, e.g. an output tile-major or transposed intermediate when the type/shape allows, carry its producer/consumer index relation in `B_e`, and lower both correct write and read addressing into real generated HLS C++ (or materialize an explicit verified layout conversion). This is a prospective example, not a statement that the new layout always improves HLS. For shape 1×N, a transpose can be degenerate; choose a nondegenerate intermediate of shape **M×N with M,N>1**, e.g. chained MatMul with batch M=3, intermediate width N=5.
- Verify output element identities, boundary partial order, physical source indexing and end-to-end numeric output on odd/tail dimensions; negative test malformed or incompatible layout and omitted/inaccurate layout conversion.
- **Go/no-go:** If the new B_e merely summarizes already implicit array use and produces no new valid executable HLS alternative, it is insufficient as the scientific novelty. Do not publish it as such; implement a real lowered change or revise the research thesis.

**Not in initial paper:** arbitrary dynamic shapes, asynchronous per-edge streams, hardware deadlock proof, formal RTL equivalence, mixed-precision converters, multi-clock systems/CDC, cross-device or full heterogeneous hardware deployment.

## Architecture-preserving lowering algorithm (proposal)

1. Validate Functional IR, operators and tensors; independently resolve each operator-local `Phi_v`.
2. Derive producer output ownership/event and consumer use relations for every graph edge `B_e`; classify model inputs/constants separately.
3. Solve a small, deterministic compatibility/choice problem for actual supported storage/layout policies. Reject nonimplementable requests, never invent adapters or silent fallback.
4. Produce Lowered Model IR containing an explicit graph-edge storage/address/lifetime scheme and its provenance; keep the operator `Phi_v` unchanged.
5. Emit actual C++ reads/writes according to B_e, compile/run host trace + independent numerical reference, and optionally run source-pinned original Vitis CSim/CSynth.
6. Report E0/E1/E2/E3 evidence separately; do not infer post-route Fmax.

## Experimental falsification and baselines

- **Restricted-IR ablation:** same emitter and operator library, but only canonical intermediate row-major tensor exchange; compare against first-class boundary contracts for *identical* Functional graph, PE/SIMD/precision, clock, target and Vitis version. If a layout option is inapplicable for a workload, report excluded/unsupported, not speedup.
- **CPU semantic cases:** target ~100–200 source cases: varied MatMul chain shapes (including nondivisible tile tails), one branched graph where supported; requested variants; invariant checks; mutant producer/consumer permutations; source generated-C++ traces and numerical oracles.
- **Runtime checks:** E0 legality, independently executed host source, structural source identities, E0/1 verification overhead. Do not claim bit-exactness based on host float.
- **Original HLS sample:** only ~12–24 fresh, prespecified candidates after CPU gate; real original CSim/CSynth, exact numerical tolerance, resource/timing constraints and independently audited source/provenance. Report HLS-estimated latency (= whole-top worst cycles × requested period), not physically realized clock or hardware runtime.
- **Statistical/causal results:** show representational increase over restricted baseline; show whether boundary behavior changes as intended; report absent gains. Pareto/QoR improvements are desirable but not necessary for semantic novelty **if** the new implemented capability is substantive and overhead is acceptable.
- **Prior-art matrix:** specifically distinguish from Allo's type-safe hardware customization composition; StreamTensor's stream layout type and conversion; HIDA/Stream-HLS graph dataflow and scheduling; Calyx static scheduling; prior mapping IRs. Do NOT assert features are absent from these systems without supported evidence.

## Future RL compatibility wording (no RL work)

Permitted in compiler paper: “FPGAI's typed, graph-structured Model IR exposes independently valid architecture requests and explicit feasibility/evidence boundaries, providing a potential interface for subsequent learning-based search.” Show only a zero-training serialization/action round-trip if desired. **Not permitted:** any claim that IR's structure improves RL sample efficiency, convergence, transfer, latency optimization or baselines absent a separate matched learning study. No RL files or checkpoint receipts are modified.

## Paper-ready claim (conditional)

“We present FPGAI, a neural-network-to-FPGA research compiler with a compositional space–time Model IR that relates operator-local architectural realization to explicit tensor production and consumption contracts. The compiler retains these mappings through executable HLS lowering, enabling per-operator heterogeneity and independently checked graph-boundary realization. Evaluation measures the new representation's expressiveness, semantic preservation and qualified hardware trade-offs.”

This statement is **prospective** and cannot be used as a completed results claim until a real nontrivial B_e layout/exchange alternative and independent experiments exist.


## Source implementation update: row-epoch temporal aliasing (prospective)

The opt-in row-lifetime subset is now implemented in \`model_ir/row_lifetime.py\`, \`codegen/hls/row_interleaved.py\` and \`compiler/row_interleaved.py\`, with source/host/temporal tests and a deterministic CPU experiment. For a supported two-MatMul chain, logical \`M×N\` intermediate values use an \`N\`-element physical scratch row at one time; row \`r+1\` overwrites addresses only after downstream computation has consumed row \`r\`. The typed contract records the distinct temporal epochs and refers to both operator Phi identities. The generated source interleaves complete producer/consumer rows, unlike the default whole-tensor barrier. This is **row-barrier source HLS, not automatically simultaneous streaming hardware**.

**No actual measurement or successful test is implied by this code commit.** The scientific question is whether explicit joint computation-space/time and value lifetime contracts enable reproducibly different, correct HLS realizations with clear legality obligations and less compilation burden than ad hoc templates, *not* whether FIFO storage or memory reuse itself was invented. Allo and StreamTensor have overlapping buffer/lifetime work; the full-paper feature comparison is essential.


## Implemented stronger cross-operator invariant: bounded lifetime safety

The opt-in 2–8 MatMul row-chain compiler now uses \`model_ir/row_chain_liveness.py::verify_row_chain_liveness\`. For the supported sequential row-epoch schedule, the verifier enumerates **each logical intermediate tensor element's write and downstream consumption**, using the real typed edge width and disjoint row epochs. Before emitting HLS, it requires every produced value to have a matching consumer, forbids overwriting a still-live physical scratch address, enforces physical per-edge capacity and complete per-row release, and returns an explicit E0 boundary event digest. It fails closed when the finite proof budget is exceeded or the typed exchange contract is inconsistent.

It is paired with \`validation/cpp_chain_trace.py\`, which **compiles and executes generated C++** and checks actual boundary-ready/consumed row ordering, and with the existing generated-C++ \`Phi\` iteration checker and independent NumPy reference. This is a more substantive architecture-to-value-lifetime coupling than a layout-only hash. Nonetheless, this is a bounded proof **only of the specified row-chain semantic model**, not a general HLS translation proof or verified physical schedule. Allo/StreamTensor prior-art novelty distinction remains unresolved.


## Status update — 2026-11-02

**This 2026-11-01 document is historical design rationale.** Since then, the repository implemented static layout contracts and bounded row-stage lifetime checking, but those do NOT semantically couple local Phi coordinates to edge-value availability. The revised, publication-critical formal novelty thesis and current gap audit live in [NOVELTY_THESIS_AND_PROOF_OBLIGATIONS.md](NOVELTY_THESIS_AND_PROOF_OBLIGATIONS.md). Refer to that document for the current falsifiable target and rejection criteria. RL remains frozen.
