# Scientific novelty gate — CST Model IR vs closest published research

**Date:** 2026-11-01. **Status: critical literature audit and implementation checkpoint, not an accepted priority claim.** This review is intentionally skeptical: the current column-major compositional HLS prototype is a **mechanism demonstrator**, not by itself a publishable new IR. No new original Vitis or end-to-end CPU campaign results have been reported.

## Closest published systems

| Work | Closest demonstrated semantics | What FPGAI cannot claim as invented | Proposed narrower differentiator, pending evidence |
| --- | --- | --- | --- |
| Allo, Chen et al., PLDI 2024, DOI 10.1145/3656401 | Decoupled compute, memory, communication, datatype scheduling, type-safe composition across accelerators; inference of memory layouts and producer/consumer rate/buffer relationships. | Composable hardware customizations, typed layout/memory interface, inter-kernel communication, type-directed layout inference, correctness-by-composition as a generic idea. | Joint per-*logical-iteration* operator Phi coordinates and per-*tensor-element* value-availability/consumption relation carried through actual NN Model IR and HLS with independent witnesses. Must compare to Allo actual documented abilities. |
| HIDA, Ye/Jun/Chen, ASPLOS 2024, DOI 10.1145/3617232.3624850 | Hierarchical Functional/Structural dataflow IR, node/stream/buffer/control, graph-level and local optimization, real FPGA NN results. | Multi-level Functional/Structural IR, graph compilation, buffers/streams, global scheduling, hardware accelerator optimizations. | Typed refinement and independently selectable operator-local Phi coupled to explicit graph-boundary value identity; show a capability or guarantee HIDA does not already encode in the same form. |
| StreamTensor, Ye/Chen, MICRO 2025, DOI 10.1145/3725843.3762817, arXiv:2509.13694 | First-class iterative tensor type encodes stream layout via iteration space/affine map; type-checked kernel fusion, stream-layout converters, buffer management. | First-class stream/tensor layout typing, producer/consumer index map, automatic layout conversion, efficient FIFO sizing, streaming kernel fusion. | **Not** another layout-type-only claim. If FPGAI supplies only layout permutations, its research novelty is weak. Must demonstrate precise coupling to spatial PE/SIMD/folding and its implications for legal lowering/comparison. |
| STeP, Streaming Tensor Programs, YARCH 2024 | Stream-centric tensor computation, dynamic tensor graph/shape handling. | Tensor stream programs and tokenized tensor dependency representation. | The current FPGAI prototype is restricted to static arrays; no credible superiority or stream innovation claim yet. |
| Timeloop/MAESTRO/AutoSA | Spatial/temporal mapping of accelerator computations, tiling, reuse and time transformation. | Phi-style spatial/temporal coordinates, reuse/tiling, PE/SIMD concepts, polyhedral space-time mapping. | Explicit graph-level *tensor-element* production/consumption obligations composable with NN operator Phi and executable source backend. |
| Calyx/CIRCT/ScaleHLS | Explicit structural/control IR, scheduling, HLS lowering and legality checking. | Three-level IR, scheduled hardware control, constraint validation, IR provenance by itself. | Precise operator-to-tensor contract, domain-restricted independent checker, observable implementation consequences. |

Primary references:
- https://doi.org/10.1145/3656401 (Allo)
- https://doi.org/10.1145/3617232.3624850 (HIDA)
- https://doi.org/10.1145/3725843.3762817 and https://arxiv.org/abs/2509.13694 (StreamTensor)
- https://ppl.stanford.edu/papers/YARCH24_STEP.pdf (STeP)
- https://arxiv.org/abs/2404.04815 (Allo preprint)
- https://arxiv.org/abs/2311.03379 (HIDA preprint)

This is a focused literature check, not a guarantee no related work exists. Before a strong novelty statement, examine the relevant full papers/appendices/artifacts and identify what their IR cannot express or verify under matching assumptions; avoid unsubstantiated “first” statements.

## New code at this checkpoint

- \`model_ir/composition.py\`: typed \`TensorExchange\` plus \`ComposedModelIR\`, alternate physical array-address bijection, operator-local Phi source-digest dependency, explicit whole-producer-complete barrier. Unsupported layouts/formats fail closed.
- \`model_ir/exchange_witness.py\`: bounded per-output-element producer owner and per-consumer logical reduction-read enumeration with physical addresses. Explicit E0 model-level limitations.
- \`codegen/hls/emitter.py\`: optional 2D internal column-major layout changes **both** MatMul/Gemm intermediate producer write index and consumer lhs read index. Default empty layout keeps old generated C++/manifest behavior. This is real generated source, not metadata-only.
- \`compiler/composition.py\`: opt-in compiler facade attaching source-hash/provenance and the E0 element-exchange witness; old pipeline intact.
- \`tests/test_compositional_model_ir.py\`, \`tests/test_cst_ir_campaign.py\`: source and host-numerical tests/negative cases. **Not yet run on user's workstation.**
- \`experiments/cst_ir_campaign.py\`: optional bounded/create-only CPU E0/E1 comparison and limited actual GCC host checks; **no Vitis or RL**.

## What is scientifically shown only if tests succeed

1. The **same** functional graph and per-operator Phi choices can be compiled using two different actual intermediate source layouts with matching producer/consumer address mapping.
2. Host-executed generated C++ retains numerical semantics within the declared host-float tolerance for a tested small NN graph.
3. An independent bounded E0 checker can enumerate how completed producer values relate to actual downstream logical reduction reads and physical array addresses.
4. The original Functional Model IR and default HLS emitter path remain unchanged on the source-visible test matrix.

This is valuable compiler engineering and can contribute to an IR expressiveness study; it **does not** establish that this joint relationship is unrepresented in Allo/StreamTensor, or that synthesized hardware contains independent storage structures.

## Required *next* novelty criterion before claiming major IR invention

**N1 — Representation:** Provide a nontrivial first-class event/data contract that is not just a renamed affine tensor layout or an unverified detached timing field, and that connects every relevant operator Phi to production/consumption semantics.

**N2 — Constructive lowering:** Materialize and independently validate a new cross-operator *schedule or data exchange* alternative that differs in actual generated HLS, e.g. a correct per-row producer/consumer availability schedule vs the existing conservative whole-producer-completion barrier. The current column-major layout is **only a first step** and does not satisfy this stronger condition.

**N3 — Independent checking:** Validate true generated source execution (and numeric semantics) and inject negative producer/consumer event or address mutations to demonstrate checking beyond existing per-op Phi oracles.

**N4 — Scientific comparison:** Run the same-compiler restricted-IR ablation and characterize what can/cannot be represented. Compare the specific new semantics against published Allo/StreamTensor capabilities, not falsely describe them as missing features.

**N5 — Original native HLS:** Qualify a small prespecified set with Vitis CSim/CSynth; do not claim a change in physical timing without E4+.

**Go/no-go:** If we only demonstrate row-major vs column-major arrays or a checker over sequential generated loops, we should publish an honest compiler engineering/IR case study or deepen implementation, **not** claim an entirely novel space–time IR at a top-tier compiler venue.

## RL preservation policy

The entire source snapshot at \`5221094af585bd501e842deca94d19c2289bb628\` is preserved at branch \`backup/rl-environment-frozen-2026-11-01\`. The experiment already running locally must remain at its original source until complete and audited. Do not modify \`rl_environment\`, the DRL agents or environment; new native RL experiments remain frozen.


## Urgent 2026 related-work correction: PACT'26 ST-Flow, ISCA'26 CODO, ASPLOS'26 FuseFlow (checked 2026-11-02)

A fresh review found **ST-Flow**, Suhail Basalama, Stéphane Pouget and Jason Cong, *A Hardware Compiler for Automating Spatial-Temporal Dataflow Acceleration*, PACT 2026, DOI **10.1145/3838684.3846865**. Its public paper artifact is https://github.com/UCLA-VAST/ST-Flow and https://doi.org/10.5281/zenodo.21881993. ST-Flow already describes PyTorch/MLIR-to-FPGA compilation, per-design spatial/temporal/hybrid mappings, cross-kernel time-multiplexed resource reuse, structurally compatible dataflow operators, nondivisor tiling, MINLP/heuristic DSE, and Vitis HLS C-simulation. **Do not claim FPGAI is the first neural-network compiler with spatial/temporal operator mapping, cross-operator reuse or a multilevel dataflow IR.** ST-Flow is an essential closest comparator. Full semantic/certificate feature comparison against its artifact is pending.

Other overlapping recent compilers:
- **CODO**, ISCA 2026, https://github.com/sjtu-zhao-lab/codo-artifact: reports automated dataflow-violation detection, data movement optimization, and schedule generation. Do not claim automated violation detection or cross-operator scheduling itself is new.
- **FuseFlow**, ASPLOS 2026, DOI https://doi.org/10.1145/3779212.3790165: fusion-centric compilation for sparse deep learning on streaming dataflow hardware. Not directly the same dense static MatMul-only subset but relevant to any broad graph-level fusion claim.
- **Wavelet**, *Let it Flow: A Formally Verified Compilation Framework for Asynchronous Dataflow*, 2026, DOI https://doi.org/10.1145/3808263: illustrates that compiler memory/scheduling type systems and verified dataflow transforms are also prior art. FPGAI has **not** machine-proved full HLS correctness; its E0 finite liveness witness and E1 host-C++ traces must be described accurately.

**Defensible position (if experiments succeed):** a research compiler case study introducing an *explicit per-operator Phi / per-boundary tensor-element lifetime relation*, with independently inspectable **bounded certificates** bound to an actual HLS implementation. This is a **narrower, implementation-specific contribution** and its distinctness from ST-Flow/Allo/StreamTensor remains **a question to answer experimentally and by artifact inspection**, not an established first claim.

**Required comparator experiment before claiming IR novelty:** check whether ST-Flow, Allo, and StreamTensor can represent the same MatMul chain with independent per-operator spatial choices and row-epoch scratch aliasing; record exact commands, source revisions, generated HLS, legality diagnostics and independent source/host evidence. Missing setup or unsupported external baseline is **unknown** rather than a win for FPGAI. A comparison to the *restricted FPGAI compiler itself* establishes an ablation, **not superiority to prior research**.

## Source pin audit pointer (2026-11-02)

Public reference revisions and outstanding **unexecuted** comparison requirements are enumerated in [PRIOR_ART_PINNED_ARTIFACTS.md](PRIOR_ART_PINNED_ARTIFACTS.md). In particular ST-Flow's annotated PACT26 tag was resolved to its referenced commit; Allo/HIDA/Wavelet are currently pinned **source snapshots, not confirmed paper evaluation artifacts**. No external equivalence, functionality, QoR, or novelty comparison is yet qualified merely by finding these sources.


## Artifact-grounded external verification checkpoint — 2026-11-02

The published literature already creates substantial direct overlap
with broad accelerator-IR novelty claims; *do not claim these features
as absent without an executable source-level control*.

- **Allo**, Chen et al., PLDI 2024, `10.1145/3656401`:
  paper and https://arxiv.org/abs/2404.04815 document hierarchical,
  type-safe compute/memory/communication customization and composition.
- **StreamTensor**, Ye and Chen, MICRO 2025,
  `10.1145/3725843.3762817` and
  https://arxiv.org/abs/2509.13694: iterative tensor types for stream
  layout, fusion, resource and buffer optimizations.
- **ST-Flow**, Basalama/Pouget/Cong, PACT 2026,
  `10.1145/3838684.3846865`,
  https://github.com/UCLA-VAST/ST-Flow: its authors identify the
  pinned `PACT26` artifact and describe spatial/temporal/hybrid
  mapping, compatible-kernel time multiplexing, memory sharing and
  source CSim validation. This is a high-priority direct comparator.
- **Wavelet / Let It Flow**, Lin/Cai/Surbatovich, PLDI 2026,
  `10.1145/3808263`: formalized dataflow compiler correctness
  and determinacy, with a published Lean/Rust artifact
  https://github.com/plum-umd/wavelet. Our finite source-trace
  checker is NOT a stronger whole-compiler theorem.

These publicly described capabilities have been researched, but
**no matched full peer artifact has been executed within the current
FPGAI CST experiment**. Peer capability and functional parity are
UNKNOWN until reproduced, with exact versions, toolchains, test inputs,
IR scope and source-level counterexample controls. For paper comparison,
ask whether each competing implementation can instantiate the SAME
mapping-conditioned complete-reduction/last-reader physical address
counterexample, not whether its marketing terminology contains `Phi`.
Compiler failure is evidence of unsupported behavior only for the
reproduced version/task; do not generalize it to other versions.

The exact prospective novelty acceptance test and fully prespecified
E0/E1/E3/E4 experiment matrix are in
`SCIENTIFIC_STUDY_PROTOCOL.md`. Physical v4 source evidence remains
unqualified; no baseline is attributed a zero BRAM or zero runtime.
