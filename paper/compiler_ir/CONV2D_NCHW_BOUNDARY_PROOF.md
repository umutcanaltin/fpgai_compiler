# Bounded Conv2D Phi-read and source-traced NCHW fan-out (L6a)

**Research checkpoint, 2026-11-02.** Immutable original code SHA `6af10f9e2ec71810181b51560d227d8dba133303`, [33/33 original quick and 33/33 paper E0/E1 experiment stages](https://github.com/umutcanaltin/fpgai_research/actions/runs/41456356235), zero failed/blocked; [raw source-pinned artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41456356235/artifacts/13146878908); [full CPU regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41456356220): **1,061 passed, five skipped**. This is a **bounded Conv2D fan-out scientific experiment**, not complete graph Conv2D lowering, general hardware validity or a firstness claim.

## Exact proposed research contribution

The research thesis remains **CST-Model IR: compositional space-time tensor-boundary contracts**, where operator-local mapping `Phi` and an *independently checked semantic input access map* jointly determine the identity/multiplicity of every required read, producer readiness, last-use, legal contiguous source address and source-execution obligations. Storage optimization, PE/SIMD factors and ordinary streaming/dataflow are not claimed as novel on their own. No RL policy training has been restarted.

This sprint introduces a **static five-operator NCHW Conv -> (Relu,Conv) -> Add -> Relu** bridge with independently mapped producer and reader convolutions. Restrictions: group=1, dilation=(1,1), explicit pads and strides, static constant RHS weights and all intermediate/output shapes equal; inference-only. Unsupported grouped/dilated/auto-padding, arbitrary 4D DAGs, mixed precision and unsupported scalar mappings fail closed.

For an output `Y[n,oc,oy,ox]` of a Conv with kernel `K[oc,ic,ky,kx]`, the semantic source coordinate is `X[n,ic,oy*stride_y+ky-pad_top,ox*stride_x+kx-pad_left]`. Out-of-bound padded taps do **not** read the tensor. Every valid source read has an output identity, reduction coordinate `(ic*Kh+ky)*Kw+kx` and consumer `Phi` owner `(replica,PE,SIMD-lane,fold)`. Source-array value `X[n,ic,iy,ix]` remains live until *all* required branch reads finish (including both reader operators and external sink for final output). Earlier overwrites are explicitly rejected. This is an **E0 logical producer-major schedule** and must not be confused with actual HLS loop order or scheduled hardware cycles.

## Independent oracle and actual source

- `model_ir/conv_boundary.py` provides bounded typed per-kernel E0 events and finite source lifetime checks with concrete counterexamples, including wrong source element, duplicated/missing taps, forged Phi ownership, early alias, and invalid geometry.
- `validation/conv_access_oracle.py` independently constructs padded NCHW NumPy input-id arrays and uses sliding-window extraction to derive the **expected read multiset** without calling the production Conv source-access helper.
- The existing canonical `codegen/hls/emitter.py` now supports optional `trace_conv_access=True`. The default source is **unchanged**. In opt-in GCC host-float mode, the actual Conv reader load uses the same C++ `__cst_source_index` variable reported by instrumentation. `validation/cpp_conv_trace.py` compares every observed Conv reader reduction tap/input address to the E0 contract, independent NumPy reference and resolved Phi; actual in-bounds C++ load-index mutants are rejected. This newly qualifies **actual Conv reader per-kernel load tracing** but NOT the complete graph scalar READ/COMMIT source-order timeline.
- The full standard HLS emitter is used for actual generated source; `validation.host.validate_host` compiles and executes original GCC C++ and compares host float numeric results with NumPy. Neither original fixed-point Vitis CSim nor synthesis/place-route was run.

## Prespecified original experiment

`src/fpgai_research/experiments/cst_conv_boundary_campaign.py`, integrated into the 33-stage one-command quick and paper profiles as `conv_phi_boundary_e0_e1_numeric`, uses four geometries:
- 4×4, kernel 3, padding 1, stride 1
- 5×6, kernel 3, padding 0, stride 1
- 5×6, kernel 3, padding 1, stride 2
- 3×4, kernel 1, padding 0, stride 1

Two per-operator mapping policies (scalar and independently chosen producer/reader Phi) give **eight prespecified cases**. Original archived results are **8/8 independent E0 Conv read-map checks, 8/8 actual GCC generated-source numerical passes, 8/8 actual GCC Conv reader kernel-load/Phi traces, and 8/8 E0 logical last-reader reuse plus premature-overwrite rejection controls**. Full case CSV records SHA, geometry, numeric error and source trace hash. No cases were silently excluded; new original Vitis calls: zero.

**Evidence ceiling:** Logical early/late source scratch alias legality is not implemented as physical reuse in the stock HLS emitter, so it does not demonstrate measured BRAM savings. The actual source trace covers **Conv reader kernel loads only**, not materialized source COMMIT and non-Conv scalar reads. Independent NumPy verifies source-element identity, not original C++/RTL formal equivalence. No vendor-clock/resource/II measurements, general Conv graph composition, external peer artifact reproduction, or scientific-firstness outcome.

## Publication-critical OPEN gates

1. General graph-wide 4D source refinement, multiple Conv branches, Conv/Relu/Add/Concat/pooling combinations, grouped/dilated cases when semantically qualified, and precise first-class cross-operator edge witnesses and proofs.
2. Actual GCC source execution trace of **all** producer COMMIT and scalar/Conv READ events, source-ordered multi-reader last-use and real source-address reuse; independent finite topology/time oracle.
3. Native original Vitis CSim/CSynth with source/part/clock/fixed-point/accuracy and actual memory-bank/port/clock receipts. No hardware-valid mask from an E0/GCC trace.
4. Executed, pinned and honest external Allo/HIDA/ST-Flow/StreamTensor/Wavelet comparisons and trained model benchmarks, with equal numerical acceptance and transparent unsupported-capability denominators.
5. Sound general `observe/actions/transition/mask` preserving the **same** validator and `UNKNOWN` hardware provenance, prior to optional future RL experiments. RL training remains frozen.

**Do not call this paper finished or publishable solely on E0/E1 Conv success.**
