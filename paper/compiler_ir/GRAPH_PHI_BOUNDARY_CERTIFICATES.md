# L6a bounded graph-wide CST tensor edge certificates

**Research status, 2026-11-02:** an implemented and source-executed **bounded** instance of the CST-Model IR mapping-conditioned composition thesis. This is not a scientific firstness verdict or a general graph/RTL proof.

## Scientific object and claimed scope

The research target remains a compositional tensor boundary relation conditioned on independently requested operator mappings `Phi_v`: for every produced tensor value, identify *all* producer reduction contributions, readiness/COMMIT, each consumer's input-element access and spatial Phi owner, final release, and the source address. A candidate is admissible only when the exact declared source semantics and numeric constraints can be preserved.

The new `src/fpgai_research/model_ir/graph_phi_contract.py` admits topological, 2D, **2–12 operator** graphs using **MatMul with independent constant RHS, scalar ReLU and scalar same-shape Add**; one runtime input and final model output. Each MatMul retains its own requested PE, SIMD, serialization and tile factors. Unsupported operators, broadcasting, reused RHS naming, attributes, mixed numeric formats, unlowered scalar Phi and dead intermediate values are rejected rather than silently scalarized. This subset permits nested branching and several readers at multiple tensor boundaries, beyond the earlier specialized five-op diamond.

For a produced tensor `t`, logical element `e`, and a MatMul consumer `v` computing `Y[m,n]=sum_k X[m,k]W[k,n]`, the read relation contains a separate required token per output `(v,m*N+n,port0,k)`; elementwise operators require one token per source input and output position. Every required token must be read exactly once **after** its source COMMIT. `last_use(t,e)` is the final read across *all* consumers, including an explicit external sink for the final model output. Scratch aliasing `addr(t,e)=base(t)+e` is rejected whenever a newly committed value collides with any still-live value. This is source-order logical safety, **not** inference of a hardware clock schedule.

## First-class evidence objects

`src/fpgai_research/model_ir/graph_edge_witness.py` materializes immutable per-edge `GraphEdgeWitness` and per-element `GraphValueWitness` records. Each edge contains a producer identity and Phi projection digest, consumer identity/port and Phi projection digest, plus a stable source-event hash. Each element contains a source address, COMMIT event index, **global** last-reader release event index, every consumer-specific read event index, and individual Phi-read ownership coordinates. The complete witness is hashed in the source emission manifest and independently recomputed from the **observed** executed GCC trace. A witness is never issued for an invalid lifetime schedule.

`src/fpgai_research/codegen/hls/graph_phi.py` reuses the existing shared MatMul lowering template for every MatMul, and emits actual sequential graph source for supported ReLU/Add. Optional source-read instrumentation in `row_interleaved.py` uses the **same C++ index variable** for actual MatMul scratch loads and observed read indices. The source refinement checker `src/fpgai_research/validation/cpp_graph_phi_trace.py` independently validates iteration Phi owners, reduction completion before COMMIT, all source READ/COMMIT/sink events, hashed first-class edge witness, and source scratch last-use. A same-shape, in-bounds **actual load-index mutation** is rejected in tests. Original host-float numerical outputs are independently compared with NumPy.

**Important caveat:** ordinary GCC instrumentation and static source-address control are stronger than a detached IR hash, but are **not machine-checked formal semantics for arbitrary C++**, and do not establish original Vitis fixed-point, hardware scheduling/II, bank/port feasibility, RTL behavior or placed/routed QoR.

## Original verified experiment

- **Immutable research code revision:** `c1298ea35bf526b743f39c910615aa72f29afadf`.
- **Original full one-command series:** [GitHub Actions run 41450443712](https://github.com/umutcanaltin/fpgai_research/actions/runs/41450443712) — **30/30 quick** and **30/30 paper** stages passed, 0 failed/blocked. [Full raw series artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41450443712/artifacts/13144403380).
- **Original full CPU regression:** [run 41450443849](https://github.com/umutcanaltin/fpgai_research/actions/runs/41450443849) — **1,033 passed**, **5 skipped**.
- **Prespecified L6a campaign:** 3 graph families `mixed` (6 operators), `triple_reader` (7 operators), `nested` (8 operators) × 2 mapping policies (serial and independent MatMul Phi) × 2 source scratch layouts (disjoint vs post-last-reader reuse) = **12/12 original GCC numerical** and **12/12 actual source-trace** checks, with 6/6 paired safe source-array reductions. All original HLS/Vitis calls: **0**.
- `experiments/compiler_ir/series.py` integrates `graph_phi_e1` into both quick and paper profiles. Workflow push filters now include graph/branch sources and tests so scientific regressions rerun when implementations change.

## Falsifiable next gates (NOT completed)

1. **General graph operator access maps:** broadcasting, Concat, Conv2D, pooling, Flatten, mixed shapes and multiple external outputs, with per-reduction/output source addresses; current rank-two subset must not be misrepresented as arbitrary DAG support.
2. **Independent source semantics and proof:** a second graph-level small-domain oracle for readiness, complete reader multiplicity, and address aliasing; exhaustive topology/order mutants; formalize local reduction and global edge induction. Current witness is derived only for schedules accepted by the same E0 checker.
3. **Source/hardware bridging:** qualified Vitis CSim, synthesized inferred RAM banks/ports, original schedule cycles and explicit UNKNOWN whenever unavailable; source scratch array size is not measured BRAM.
4. **Independent comparison:** execute pinned Allo/HIDA/ST-Flow/StreamTensor/Wavelet published artifacts with overlapping graph capabilities and matched numeric/schedule constraints; a source pin is not a baseline experiment.
5. **Learning-ready decision semantics:** tie action masks, `observe/actions/transition` and source guarantees to the exact graph-edge checker; train no policy or RL while the agreed RL study remains frozen.

**Publication gate remains OPEN.** The candidate contribution is mapping-conditioned source-checkable *composition across multiple independently mapped operator boundaries*, not creating an IR, using PE/SIMD, ordinary memory reuse, or generated HLS source.
