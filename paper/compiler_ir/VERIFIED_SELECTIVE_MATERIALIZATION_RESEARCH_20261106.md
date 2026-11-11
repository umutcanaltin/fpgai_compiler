# Proposed first-paper improvement: verified selective intermediate materialization

**Design protocol, 6 November 2026. No new performance experiment has been executed.**
This is a prospective, falsifiable research direction, NOT a measured speedup or a claim of unique novelty. Preserve all original frozen studies and the RL feature branch. Start implementation only after auditing existing modules and deciding which source backend is the canonical one.

## 1. Why the existing result is insufficient

Historical R2s: 102/126 architecture changes reduced *committed-value source scratch*, but the maximum was 12 scalar slots. For the largest original tensor configuration, the saving was only 0.389% of **all declared mutable source array elements**. The current R2h generated C++ uses separate `scratch[]`, `acc[][]`, and `counter[][]`. It is therefore unsurprising that 47/57 qualified original factorial HLS contrasts reported tied BRAM estimates. R2x already found observable generated RTL memory-name structure changes while all 14 qualified *address-mode* HLS comparisons tied in BRAM. Do not repeat that result as new science.

**New target:** change the *materialization decision* and actual generated memory organization, not just the allocation of an already small scratch pool.

## 2. Hypothesis

For a bounded statically scheduled network with one produced intermediate tensor and multiple independently mapped consumers, can a contract-aware compiler select between:

1. **Whole-tensor materialization**: separate stored intermediate tensor, read by consumers.
2. **Element lifetime reuse**: current certified scratch slot aliasing, same overall computation.
3. **Certified forward/no-spill**: forward each produced value directly to all its required consumers, retaining it in bounded registers or FIFOs only as long as the operator-local issue schedules require.
4. **Certified selective spill**: forward some values while materializing others when consumer timing/rate mismatches exceed bounded forwarding storage.

The important question is not whether streaming or fusion exists already. It does. The potentially distinguishing mechanism is **a single typed cross-operator obligation record**, linking per-operator Phi issue coordinates, producer completeness, all-reader ownership, chosen delivery mode, generated C++ source access evidence, and source-vs-HLS-RTL observables.

Reject any mapping whose scheduling, fanout/read multiplicity or bounded storage ownership cannot be established. UNKNOWN is not a proof of feasibility. This is source E0/E1 only until vendor E2/E3/E4 evidence is obtained.

## 3. Proposed typed contract

For each producer value `v`, track:

- `produce(v)` after all required arithmetic contributions;
- `deliver(v,c,k)` for each enumerated consumer/use ordinal `(c,k)`, preserving operator-local Phi issue semantics;
- `retain(v)` and `release(v)` for local storage, or `spill(v,address)` and `reload(v,address,c,k)` if required;
- an obligation that every required consumer read obtains the right live value exactly once, and every physical source memory access is justified by the contract.

For a FIFO-like channel `c` with known static producer/consumer issue schedule, the provisional occupancy bound is

    B_c >= max_t (Produced_c(t) - Consumed_c(t))

subject to prefix non-negativity, ordered tokens and consistent timing units. It is a source-schedule bound, **not a proved FPGA cycle bound**. Count every consumer use: for the current MatMul fanout, a producer H element may be used by every reader across several output columns. A one-time delivery to a consumer is not automatically enough.

Bank/port conflicts, HLS operator overlap, pipeline initiation intervals, bit-exact arithmetic differences and clocked hardware scheduling are separate obligations. They must be modeled or marked UNKNOWN. Do not write a safety theorem that silently assumes all scheduled source events happen in separate hardware cycles.

## 4. Controlled arms for a new experiment

Keep one fixed network, weights, numeric format, part, source issue schedule where applicable, clock request, uncertainty, and numerical reference for each pair.

| Arm | Source behavior | Purpose |
|---|---|---|
| A — disjoint storage | Existing R2h whole intermediate scratch | Conservative baseline, already implemented |
| B — element reuse | Existing R2h element-exact scratch | Reproduce existing slot optimization, not a new novelty claim |
| C — certified forwarding | Proposed new emitter; output of producer consumed while live without full intermediate scratch H storage | Test whether storage elimination really reduces generated memory while preserving all intended consumers |
| D — selective spill | Proposed extension after C; scheduled buffering when consumer maps differ | Test nontrivial heterogeneous Phi and temporal decoupling |

**Critical control:** C and D must pass the *same* complete graph input/output and intermediate numeric conformance. Compare total declared scratch+accumulator+counter/register or FIFO storage accounting, not scratch alone. HLS resource differences must be attributed to one changed materialization mechanism; source changes to PE/SIMD/precision are separate factorial dimensions.

Suggested preregistered dimensions to span storage granularity at 16-bit elements: producer shape `m*n` of 16x16, 32x32, 64x32 and 64x64, with `k=8`, `q=8` or `16`, and reader counts `R=2` and `R=4`. Retain all planned source cases, including failures. These are **proposed** and must be resource-screened on target before native execution, with screening criteria and excluded cases reported. Do not choose only sizes that already produce positive HLS results.

The original 32-case R2u HLS cohort is **historical** and uses a different source population. Treat this prospective experiment as a separate study. If compatible, predeclare one FPGA target and fixed clock/uncertainty before reviewing outputs; do not tune these after observing failures to manufacture E3 passes.

## 5. Acceptance gates and outcome measures

**G0 architecture semantics:** Valid selected mappings produce a consistent graph and full producer/read ownership contracts; invalid schedules fail with counterexamples. Require both valid and invalid controls: mismatched reader order, premature overwrite, missing fanout reader, duplicate consumption, unsupported buffer depth, and bank/port conflict where modeled.

**G1 actual generated source:** Compile and execute original generated C++ over multiple numerically distinct deterministic inputs, compare all intermediate and final values to an independent NumPy implementation, and instrument real scratch/FIFO/forwarding operations. Mutate **actual operations** while leaving metadata untouched, and report whether numerical outputs remain unchanged for the negative controls.

**G2 source cost:** Report total stored bits including accumulators, counters, registers and FIFOs, the maximum simultaneous live-value demand, source emitter complexity and checker runtime. A decrease in scalar scratch alone is insufficient.

**G3 native HLS:** For every preregistered pair, report E2 CSim numeric error, strict E3 clock qualification, BRAM18K/URAM/LUTRAM where reported, LUT/FF/DSP, worst cycles and estimated clock. No favorable deltas from timing-failed arms. Report both absolute and paired values, including regressions and unknowns.

**G4 physical implementation:** For selected *predeclared* HLS-qualified pairs, synthesize/place/route with identical legitimate constraints; require clean DRC, and report mapped BRAM/LUTRAM/URAM, WNS, area and throughput. The original six DRC-warned routes do not qualify this result.

**G5 scientific comparison:** Compare a clearly defined contract obligation with executable Allo verification, and with related memory/scheduling work. Do not claim that Allo, Dahlia, Filament, CODO, ST-Flow or PoCo lack streaming, memory safety or equivalence checking. Measure exact coverage, author-visible counterexamples, runtime and HLS resource outcomes where comparable.

A positive result requires a repeatable **qualified hardware** advantage on an identifiable subset of the prespecified workloads without hidden numerical/clock regressions, plus a demonstrable additional verification capability. A negative result is publishable as a boundary/mechanism finding if the experimental design is sufficiently independent, complete and reproducible. There is no target percentage to tune the benchmark toward.

## 6. Novelty/overlap checklist before writing priority claims

- **Allo (PLDI 2024):** composable accelerator scheduling, memory customization and source transformation equivalence checking. https://doi.org/10.1145/3656401
- **Dahlia (PLDI 2020):** time-sensitive affine constraints on memory resource use. https://doi.org/10.1145/3385412.3385974
- **Filament (PLDI 2023):** modular timeline-based timing and structural constraints. https://doi.org/10.1145/3591234
- **CODO (ISCA 2026):** FPGA dataflow analysis, memory movement and scheduling optimizations. https://arxiv.org/abs/2604.12618
- **ST-Flow (PACT 2026):** spatial-temporal dataflow mapping and compiler implementation. https://github.com/UCLA-VAST/ST-Flow
- **PoCo (TRETS 2025):** shared multi-producer/multi-consumer on-chip buffer implementation and resource optimization. https://doi.org/10.1145/3771938

**Proposed distinctive claim to test, NOT assert now:** source-certificate-guided choice between materialization, forwarding and selective spill for heterogeneous operator-local Phi maps with real generated-source access witnesses, together with HLS/physical evidence of which choices survive lowering. Source memory minimization, static lifetime coloring, streaming, compositional typing and backend resource improvements are not new by themselves.

**Implemented bounded first step (new experiment, not observed QoR):** [R4a row-retained materialization source/GCC protocol](R4A_ROW_RETAINED_MATERIALIZATION_EXPERIMENT_20261106.md). This new opt-in source path reuses canonical R2g arithmetic and holds each H producer row until all mapped fanout readers finish. It is NOT general zero-spill forwarding or proven physical BRAM savings. R4b provides an explicit guarded native Vitis E2/E3 qualification path. The R4a source/GCC E1 study was subsequently executed (quick 3/3 and paper 5/5; [observed terminal results](R4A_WORKSTATION_SOURCE_RESULTS_20261106.md)). The R4b Vitis E2/E3 native and physical results remain unexecuted; no hardware saving is established.

## 7. Engineering integration and no-duplication rule

Reuse the existing `src/fpgai_research/model_ir/` architecture/contract framework and `src/fpgai_research/codegen/hls/` canonical emitter. Do **not** create an alternate neural network compiler, second independent source AST, second set of HLS vendor toolchains, or separate ad hoc research-only architecture graph.

Initial implementation should be a typed **materialization decision** and a compatible contract extension, with feature flag defaulting to the original materialized behavior. Extend one existing bounded MatMul fanout emitter first, then incrementally extend the existing `neural_chain` mixed Conv2D/Dense path. Keep the mature original studies and tests unchanged. Reject unsupported cycles, branches or latency schedules rather than guessing.

**Start with this check before coding:** inspect `r2h_allocated_fanout_cpp.py`, `fanout_contract.py`, `source_allocation_baselines.py`, `neural_chain.py`, and the canonical HLS packaging layer. Write one integration test that compares generated C++ intermediate values and rejects an intentionally premature last-reader release. Only then add candidate source lowering and HLS pilot cases.

**No original native cohort is to be regenerated or overwritten. The paused RL branch remains untouched.**