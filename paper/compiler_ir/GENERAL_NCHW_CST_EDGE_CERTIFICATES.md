# General topological NCHW Phi-conditioned value-edge contract — L6a

**Original verified research checkpoint, 2026-11-02.** Source SHA `fcb1f621bb6c0eaadae8f2206e823f59bd2bf21c`. [34/34 quick and 34/34 paper experiments passed](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339211), zero failed or blocked, [original unmodified archived artifacts](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339211/artifacts/13156047647); [full CPU regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339307): **1,070 passed, five skipped**.

## Precise scientific object

The research target is a *compositional, independently checkable CST Model IR* connecting local per-operator `Phi`, mathematical operand-access maps, per-value producer readiness, **every** required consumer read (including reduction/tap multiplicity), last-use across all readers and declared logical source-address ownership. A legal source-schedule or storage action needs an exact, checkable event certificate; unqualified hardware performance/port actions must remain UNKNOWN. This research compiler is a standalone scientific subset, **not** a replacement for the separate five-layer Morfics production architecture or the frozen RL/Gym study.

## New general NCHW implementation

`src/fpgai_research/model_ir/nchw_graph_contract.py` checks a bounded **2–12 operator**, one-input/one-output, **topological NCHW four-dimensional** graph rather than a hard-coded Conv diamond. Operator subset: `Conv` (distinct constant kernel, group=1, dilation=1, explicit pads/strides), `Relu`, ONNX-style rank-4 broadcast `Add`, channel-axis `Concat`, and unpadded 2×2 stride-2 `MaxPool`. All Conv stages retain independently requested PE, SIMD, parallelism, serialization and eligible tile mechanisms; non-Conv mappings remain restricted to supported scalar lowering. Invalid shapes, missing edges, unsupported ops/axes, unlowered mapping and mixed numeric formats fail closed.

The `nchw_graph_events` event relation is **mathematical producer-major**. A Conv output element `(n,oc,oy,ox)` requires one read per valid `(ic,ky,kx)` source coordinate, with padding taps outside the tensor excluded. Each read carries an output index, source element, explicit operand port, reduction identity `(ic*Kh+ky)*Kw+kx`, and resolved consumer Phi owner. `MaxPool` similarly creates four logical reduction reads per output; `Add` derives its source indices from input broadcast shapes; `Concat` maps to a unique source port/channel and preserves per-channel spatial positions; `Relu` reads one source element. Every produced element is committed exactly once. Final sink reads are explicit.

`verify_nchw_graph` demands all required reads exactly once after COMMIT and releases logical source addresses **only after the global last read across every consumer**; source-base overlaps that would overwrite live values are rejected with concrete counterexamples. `nchw_edge_witnesses` produces first-class per-edge/per-element source addresses, readiness indices, actual per-consumer logical read indices, Phi owners and global release events. The witness digests are stable over source-pinned logical inputs.

**This is an E0 bounded logical value schedule, NOT emitted concurrent hardware timing and NOT a formal proof of arbitrary C++/RTL.**

## Independent semantic and original C++ evidence

`src/fpgai_research/validation/nchw_access_oracle.py` independently derives expected source-element/read-count multisets from NumPy padded sliding-window Conv identities, NumPy 2×2 pool views, `broadcast_to` index maps, concatenated channel/port grids and plain elementwise identity; it does **not** use the production `nchw_graph_contract._source_reads` helper.

`src/fpgai_research/validation/cpp_nchw_conv_graph_trace.py` GCC-compiles and executes the **original canonical HLS generated C++** under opt-in `trace_conv_access=True`. It checks real **intermediate Conv input-load** addresses (the same index variable drives the actual C++ load and the emitted trace) and per-contribution resolved `Phi` for **all Conv consumer stages**, then checks their combined logical multiset against the independent NumPy oracle. Host-float model outputs are separately evaluated against NumPy via `validation.host.validate_host`.

**E1 scope limit:** Real GCC trace attests **intermediate Conv loads only**; other ReLU/Add/Concat/MaxPool/sink READs and all COMMIT event ordering are still planned E0 events. The canonical source has its own intermediate arrays; post-last-use logical scratch reuse is **not emitted as shared HLS memory**, and source scratch element reductions are **not BRAM/resource savings**.

## Exact prespecified original experiment

`src/fpgai_research/experiments/cst_nchw_graph_campaign.py`, integrated as the 34th stage `nchw_graph_e0_partial_e1` in both quick and paper one-command profiles.

- Three genuinely distinct graph families: `branch_concat` (7 operators with Conv/Relu/Concat/second Conv/Add), `pool_fork` (5 operators with MaxPool and stride-2 Conv branches), and `broadcast_fork` (6 operators with a 1×1-spatial broadcast skip). Four graph/shape cases, including an additional 5×5 Concat graph.
- Two independent per-Conv Phi mapping policies, `scalar` and `independent_conv_phi`, and two **logical E0** scratch layouts, disjoint and post-last-use. 4×2×2 = **16 source-pinned result rows**, eight generated-source GCC executions and eight matched logical-layout pairs.
- All **16/16 E0 source/read/last-use graph checks** and **16/16 independent NumPy semantic checks** passed; all **16/16 recorded original GCC numerical and Conv-source-load checks** passed, representing eight unique executed C++ sources reused in the paired logical-layout rows.
- All eight layout pairs have identical Functional/architecture/source C++ hashes and E0 event hashes, with lower **logical** scratch element counts after the final release. The results explicitly record `physical_alias_emitted=false`, `real_source_alias_emission_cases=0`, `original_vitis_calls=0`.
- Negative tests include forged Phi owner, changed source index, duplicate/missing reduction token, premature overwrite of an actively read tensor, unsupported Conv configuration/operator, and unsafe event budget.
- Raw `cases.csv` and `summary.json` in `series_paper/nchw_graph_e0_partial_e1/` record exact SHA and evidence. [Original experiment artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339211/artifacts/13156047647).

## Publication-critical gates still OPEN

1. A **single source-executed generalized NCHW schedule** attesting all READ/COMMIT events, not a modeled E0 scalar trace mixed with actual Conv loads; actual address-aliased storage emission and real refinements with separate source and independent reference.
2. Independent **temporal**, numerical and finite topology oracles beyond shared expected-source-token maps, a compositional proof argument, a wider relevant operator/DAG/shape domain (including mixed MatMul/NCHW conversions and grouped/dilated conv when properly qualified).
3. **Original Vitis fixed-point CSim and CSynth**, qualified RAM/port inference, actual schedules/timing/II/BRAM/DSP and hardware QoR under pinned part/clock; absent native evidence means UNKNOWN, not source-certified hardware.
4. Paired trained-model studies and **executed pinned** Allo/HIDA/ST-Flow/StreamTensor/Wavelet overlaps; source pins alone cannot establish novelty, parity or superiority.
5. A stable action/observation/mask API referencing the **same** semantic lifetime/physical qualification checker. RL *training remains frozen*; no historical RL evidence is modified.

**Do not treat this bounded E0 + partial E1 result as publication completion, hardware performance or scientific-firstness.**
