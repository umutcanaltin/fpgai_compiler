# L6a/L7 — non-identity source access maps and independent semantic oracle

**Source-pinned qualified status, 2026-11-02:** `dbf6aade8228cbbe703037fb5108715cd0dd64f9`; [original compiler-paper quick/paper series](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625112) **32/32 stages passed in both profiles** with zero failures/blocks; [raw archived reports](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625112/artifacts/13146491650); [full CPU regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625106): **1,048 passed, five skipped**. No original Vitis execution occurred in the two new stages.

## Research question

Does the graph-wide CST tensor boundary relation preserve the *actual source-element identity* and read multiplicity when tensor edges are not same-index one-to-one? Can an independently constructed oracle expose failures of a producer and lifetime verifier that depend on the **same incorrect semantic read map**?

**Implemented finite scope:** static rank-two **ONNX-compatible Add broadcasting** and **axis=1 Concat**, alongside the existing MatMul(with distinct constant RHS) and scalar Relu, all within a bounded 2–12 operator, single-input/single-output source pipeline. Independent MatMul operator-local `Phi` remains supported and unchanged. Unsupported rank/axes/operators/mixed numeric formats or unlowered scalar mappings fail closed.

## Explicit indexed-edge access relations

For `Add(A,B)[r,c]`, an input with static shape `(R,C)` reads flat element `(0 if R==1 else r)*C+(0 if C==1 else c)`. A `(m,1)` column can therefore have **n source reads** per produced value when broadcasting into `(m,n)`; it is NOT released after one reader acknowledgment.

For `Concat(A,B,...,axis=1)[r,c]`, find the unique port `p` covering the cumulative channel width `c` and read `r*width_p + (c - offset_p)`. Concat's mixed widths require a port-specific address and a different output shape from either input. The new `model_ir/graph_access_maps.py` drives expected per-consumer `GraphEvent` obligations and the graph-lifetime proof; the emitted C++ independently calculates these source indices and logs the **same index variable used for the physical scratch load**. The full edge-witness digest is recomputed from observed GCC C++ source-trace events, rather than trusting only source metadata.

## Independent NumPy semantic reference

`validation/graph_access_oracle.py` builds expected read multisets using **NumPy broadcasted index grids** and **concatenated index/port grids**, plus explicit MatMul reduction loops; it deliberately does not call `graph_access_maps.scalar_output_accesses`. The actual GCC graph trace is checked against BOTH the production lifetime relation and the independent semantic relation. Neither is an original RTL verification.

The controlled `graph_semantic_oracle_e0` study injects in-bounds, bijective wrong input indices. It temporarily substitutes the SAME faulty read-map function for both event generation and the lifetime verifier (restoring it immediately afterward), while the independent NumPy oracle continues to use its own semantics. This is an explicitly **synthetic correlated-implementation failure**, not evidence of an upstream compiler bug.

## Source-pinned original experiment outcomes

- `graph_access_e1`: 4 distinct graph/shape settings (2 broadcast, 2 Concat) × 2 independently requested MatMul architecture policies × 2 scratch layouts = **16 original GCC host-float numerical passes and 16 actual source-trace passes**, with 8/8 safe, source-matched disjoint vs post-last-reader source-storage pairs.
- `graph_semantic_oracle_e0`: **8 total logical semantic cases** (4 positive controls, 4 artificially corrupted access-map permutations). **4/4** corrupted cases were falsely accepted **by the deliberately correlated lifetime-only reference**; **0/4** were falsely accepted by the independent NumPy semantic oracle. The positive 4/4 cases passed the independent oracle. This is *relative to NumPy logical index semantics*, not an assertion of hardware safety/unsafety or statistical superiority.
- Actual new native Vitis calls **0**. No inferred RAM bank ports, hardware cycles, latency, BRAM, DSP, energy or physical implementation evidence.

Raw matched CSVs, JSON source receipts, exact source SHA and full-stage denominators are under `series_paper/graph_access_e1/` and `series_paper/graph_semantic_oracle_e0/` in the [original archived artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625112/artifacts/13146491650).

## Scientific contribution and remaining falsification gates

The candidate contribution is not simply an IR, resource optimization, ordinary buffer reuse, NumPy evaluation, or a new RL algorithm. It is a **source-checkable compositional tensor-boundary contract**, where each operator-local `Phi` and **per-operand semantic access map** generate first-class per-value readiness, producer reduction, all consumption, last-use and storage obligations; the actual source is audited against an **independent** semantic oracle.

**Still OPEN before publication:** Conv2D/grouped/dilated and pooling edge access, arbitrary broadcast/axis/rank variants, general multi-input/multi-output DAGs, a complete independent scheduling/time oracle and formal refinement reasoning, qualified original Vitis fixed-point CSim/CSynth and actual hardware RAM-port/cycle attribution, matched trained-model cases, external execution/comparison of pinned prior-art systems including Allo/HIDA/ST-Flow/StreamTensor/Wavelet, and evidence for originality and any hardware performance claim.

**RL policy training remains frozen.** An eventual `observe/actions/transition/mask` must share the same **full** legality/semantic oracle and represent missing vendor/hardware evidence as `UNKNOWN`. CPU/GCC observations must never be promoted to hardware-qualified masks.
