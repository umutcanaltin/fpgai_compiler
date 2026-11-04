# Original GCC all-intermediate refinement and materialized NCHW source scratch

**Research evidence checkpoint, 2026-11-02.** Verified original source revision `319ab4e94a57019d0249dbe2254a925e748e4ace`. [34→36 stage source-pinned experiment series](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917065) passed **36/36 quick** and **36/36 paper** stages, zero failed/blocked; [immutable original artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917065/artifacts/13157444356); [full CPU regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917137) passed **1,083 tests, five skipped**.

## Scientific goal and precise scope

The proposed CST-Model IR contribution is a compositional *operator-local Phi + semantic input access + readiness + all-reader last-use + source-storage* contract, not memory sharing, tensor dataflow, ordinary interval allocation, PE/SIMD selection, the IR name, or an RL policy alone. Existing earlier E0 source event proof checks bounded topological static NCHW graphs over Conv, ReLU, Add, channel Concat, and 2×2 MaxPool. Earlier partial E1 evidence source-attested actual GCC Conv input LOAD indices and numerical final outputs; exact source READ/COMMIT timeline for all scalar/Conv/Pool/Concat accesses is still not qualified.

**This increment adds two distinct original execution checks.** Each is based on the original generated HLS C++ arithmetic, with GCC host-float, and remains independent of native Vitis fixed-point CSim and synthesized hardware.

## L6a: Every original intermediate tensor is numerically attested (E1)

`src/fpgai_research/validation/cpp_nchw_intermediate_trace.py` compiles and executes the exact emitted C++ under existing opt-in `FPGAI_TRACE_TENSORS` and independently recomputes every intermediate using `validation.reference.evaluate(..., return_all=True)` (NumPy). For every operator in declared topological order, the verifier demands exactly the expected number of observed tensor elements, no duplicates/gaps/missing stages, finite values, correct source, Functional and per-operator requested effective-Phi receipts, and elementwise numerical agreement with the independent NumPy tensor. It also checks the external C++ output against the last observed output tensor. It retains SHA-256 receipts and per-stage max absolute errors. A mutant changing an actual Conv weight in the generated C++ (with a recomputed source hash) is rejected at its first semantically corrupted intermediate; missing tensor observations, source hash mismatch and unknown event-size budgets fail closed.

**The observed post-operator tensor snapshots are NOT individual C++ STORE/COMMIT events or proof of all consumer READ ordering.** They do, however, check actual intermediate numerical data at every operator boundary rather than assuming a correct output entails correct internal refinement.

Prespecified `nchw_all_intermediates_e1`: four graph/shape settings × two independent Conv architecture policies = **eight genuinely executed GCC cases**, all passed; **50 observed operator snapshots and 1,764 intermediate tensor elements** independently checked, with every case in original CSV/JSON. Native Vitis calls: **zero**.

## L6a: Materialize conservative shared source memory (E1)

`src/fpgai_research/codegen/hls/nchw_shared_scratch.py` creates a local C++ `data_t __cst_scratch[S]` and binds every intermediate tensor to a concrete, contiguous source scratch slice using explicit pointers. This is **not** a proposed source-address-only layout; the generated original arithmetic/Conv loops actually access the aliasing C++ arrays. The top-level model output remains an external output pointer, not an aliased local array. The opt-in emitter's normal, uninstrumented default codegen is unchanged.

The acceptance gate combines two distinct checks:

1. Conservative **whole-operator producer-to-last-consumer intervals**, with all endpoints inclusive. Two intermediate tensor buffers may share an address only if their ENTIRE live operator-stage intervals are disjoint; this avoids relying on an unobserved within-operator loop schedule. The planner uses a deterministic bounded first-fit contiguous placement, **not an exact optimal allocator**.
2. Existing E0 `nchw_graph_contract` checks the complete tensor source-value READ/COMMIT/last-use relation and proposed actual source array bases. Invalid or source-overlapping live tensors, missing input/arrays and search/capacity bounds are rejected. The source hash, architecture identity, offsets, interval map and original unshared HLS hash are retained in the manifest.

`src/fpgai_research/experiments/cst_nchw_shared_scratch_campaign.py`: four original graph settings × two Phi policies × two ACTUALLY DIFFERENT generated C++ source layouts (disjoint local scratch vs conservatively reused scratch). **16/16 original GCC intermediate numerical checks and 16/16 original actual Conv input LOAD/Phi-owner checks** passed, executed as **32 GCC runs** because every layout has an intermediate and Conv load audit. **Eight of eight paired cases have smaller local C++ scratch allocations** under reuse; the minimum saving is eight source elements, total across pair differences 408 elements. Each variant has a different C++ source hash; each pair shares the same Functional graph, Phi and E0 logical source-event identity. An invalid prematurely overlapping source alias is rejected. Complete original source CSV retains the denominator, memory source-element differences and all evidence receipts.

**Critically:** these are **C++ scratch array capacities in the emitted source**, not measured LUT/BRAM/DSP savings, inferred FPGA primitive banking/ports, vendor timing, II, placed/routed PPA, or real FPGA measurements. This use of conservative first-fit is not claimed as a newly invented memory allocation algorithm.

## Reproducibility

Immutable original source SHA `319ab4e94a57019d0249dbe2254a925e748e4ace`; [original 36-stage run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917065), [raw source-pinned artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917065/artifacts/13157444356), [1,083-pass CPU regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917137). The original artifact includes `series_paper/nchw_all_intermediates_e1/{summary.json,cases.csv,operator_receipts.json}` and `series_paper/nchw_real_source_scratch_e1/{summary.json,cases.csv}`. All 36 stage statuses passed in both profiles, zero failed and blocked; source revisions match the immutable original commit.

The regular compiler-IR one-command series is still `bash experiments/compiler_ir/run_all.sh --profile paper --output <fresh-create-only-folder>`. Both new stages run GCC and NumPy, with **zero original Vitis invocations**. Source artifacts are create-only; never rewrite archived studies or RL evidence.

## Falsifiable next publication gates

- **Full source-event refinement:** actually trace every scalar, Concat, pool, and Conv consumer LOAD and every producer STORE/COMMIT at its original C++ execution site, reconcile the redundant first-tap read in the existing MaxPool kernel, and independently replay the *observed global event order*. E0 producer-major ordering is not equivalent to original compiled loop scheduling.
- **Hardware truth:** native Vitis fixed-point CSim and CSynth of *both* source variants with exact source/clock/part provenance, numerical acceptance and actual memory primitive/bank/port/clock evidence; if unavailable return UNKNOWN. Native HLS may reject pointer aliases or infer a different physical memory hierarchy.
- **Robustness/generalization:** independent complete finite interleaving and resource/time oracles, multiple train/test held-out topologies, mixed 2D/4D boundaries, complete precision policies and workload-specific closed-loop search semantics.
- **Independent outside novelty evidence:** actually run comparable pinned artifact versions of Allo, HIDA, ST-Flow, StreamTensor and Wavelet where supported, with matched numerical constraints and transparent unavailable/unsupported combinations. Source pins alone cannot establish priority or hardware superiority.
- **RL-ready typed mask:** preserve SAME semantic source and hardware provenance in the action mask, never label an abstract/E1 GCC proof as achieved FPGA legality. RL/Gym *training remains frozen*.

**The paper is not finished and no novel-firstness, competitor superiority, Vitis performance, or hardware savings is established by this source-only GCC increment.**
