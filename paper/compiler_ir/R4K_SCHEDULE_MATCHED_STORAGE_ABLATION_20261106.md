# R4j verified original three-site RTL evidence and R4k source-schedule-controlled storage experiment

**6 November 2026 — R4j and R4k original E1 now WORKSTATION VERIFIED.** The author reported 40 focused tests passing, followed by a successfully completed new create-only R4k source-only E1 experiment with status `MATCHED_ROW_SOURCE_SCHEDULE_REAL_HOST_E1_ACCEPTED`, original source revision `776e3a7a4b599fd114718f6a03ab73e06be17f5c`, and zero native Vitis/Vivado calls. The [R4l source-pinned two-arm native protocol](R4L_SCHEDULE_MATCHED_STORAGE_NATIVE_GATE_20261106.md) is newly implemented but NOT yet workstation-tested or run through Vitis. These are distinct experiments. No original study or native HLS files may be altered.

## R4j original corrected evidence

The user's focused corrected R4j/R4i/R4h regressions passed **34/34**. The original source-attested 32/6 read-only CLI returned `SOURCE_SCRATCH_SITES_PARSED_NOT_ELABORATED` for exactly **3** matched Verilog source declaration sites per arm, each with explicit literal named parameters `DataWidth=32`, `AddressWidth=8`, and `AddressRange=214` (full-H) vs `134` (row-H). Each source declaration pair differs by **80 entries**. This was previously incorrectly shown as UNKNOWN due to a singleton aggregation gate, since fixed and tested. The R4j original immutable first report was not overwritten; corrected CLI execution was stdout-only.

Across these three original source sites, the numerical sum is `3*(214-134)=240` **source-declared entry-slots**, a numerical match to the earlier R4a compiler source-level saving `(m-1)*n = (16-1)*16 = 240` scalar scratch elements. The comparison **does not** establish an E1-to-RTL element-identity mapping, three elaborated memory instances, physical on-chip bits saved, BRAM packing cause, or power/throughput. Original 32/6 HLS resources remain estimated BRAM18K **13 vs 13**, DSP **23 vs 19**, LUT **4990 vs 4755**, FF **1798 vs 1647**, worst cycles **22192 vs 22602**. The original E3 comparison also confounds memory layout with the producer/consumer source-loop schedule.

## R4k scientific question and controlled contrast

**Question:** With the **same row-local producer/consumer source issue schedule**, can a change ONLY to the full-H versus row-H scratch **materialization address layout/capacity** pass a strict independent source E1 check and subsequently produce an HLS hardware resource difference?

The old R4a/R4d original full-H emitter executes all producer rows before the global read phase. The old row-H emitter interleaves producer and consumer activity row by row. Therefore their existing HLS resource and latency deltas are **not** a clean causal estimate of storage-only optimization.

R4k creates two **new, source-matched** arms with the exact same row-local loop/ordinal schedule and the same graph, weights, numerical format 32/6, model, target part, clock 12 ns, clock uncertainty 1.5 ns, and absolute CSim tolerance 0.02:

| New E1 arm | Source schedule | Scratch storage | Purpose |
|---|---|---|---|
| `full_storage` | R4a row-local | full H: original unaliased H index | Storage-only control |
| `row_storage` | R4a row-local | one retained H row, aliased between rows | Storage-only intervention |

The `full_storage` source is derived from the already canonical row-H emitter by EXACT checked substitutions of scratch array capacity, H store/load address expression and the non-H scratch base offsets. Both arms reuse the existing arithmetic loops and Phi ordinal decoder. Any unexpected emitter template change is a hard error. The disjoint full-H slot map is certified using existing R2f `verify_allocation`; the row-H arm uses the original R4a row-retention certificate. Original full H and original R4d native studies are NOT rewritten.

## Scientific source gates

**E0:** same original source event order SHA, same original architecture and FunctionalModel SHA, successful independent Phi issue and per-tensor lifetime/source address map. Exact event sequences must be the same between the two new arms.

**E1:** build and execute each emitted C++ arm with real g++ (`FPGAI_HOST_FLOAT`, `FPGAI_TRACE_R2G`); decode original actual source READ/COMMIT address witnesses with the existing strict decoder; compare ALL intermediate values and final output with the independent FunctionalModel NumPy evaluator; additionally require identical source event sequences, every intermediate value, final output and logical trace with address witnesses removed. An intentionally mutated H array load must fail independent numerical checking even if the trace address witness variable is unchanged.

**Original study provenance:** before new output allocation, reuse the R4e original read-only native audit, require the original **both-arm 32/6 E2 and E3 qualifications**, exact previously observed row-H source CPP SHA, original Vitis Tcl SHA, and stable original part/clock/tolerance. Original source+native evidence is not rerun.

**Packaging only:** after E1, create two new Vitis HLS packages, each with an unchanged, identical testbench and exact matching 12ns/1.5ns Tcl. Emit a create-only original source+host report directory with SHA-256 artifact inventory. This first stage does **not** execute Vitis, Vivado, CSim or synthesis. If E1 fails, retain a new failure record rather than claiming any result.

**E2/E3/physical pending:** only after E1 acceptance and independent source/Tcl hash checks should new original **paired** Vitis CSim/HLS executions be considered. Their resource changes will be compared ONLY against each other; historical full-H vs row-H HLS numbers are a separate mixed-schedule contrast. Physical E4 requires a later independent Vivado gate with real synthesized/placed/routed evidence and clean DRC. Do not infer hardware cycles from the host issue trace.

## New files and effects

- `src/fpgai_research/codegen/hls/r4k_full_storage_row_schedule_cpp.py`: alternative source emitter + exact event/slot proof; imports and leaves R4a row source emitter unchanged.
- `src/fpgai_research/validation/r4k_matched_storage_gcc.py`: independent two-arm GCC E1 conformance, numerical oracle and original-address witness verification.
- `src/fpgai_research/experiments/cst_r4k_matched_row_schedule_storage.py`: CLI plan, frozen R4e evidence preflight, create-only two-arm E1 experiment/source packages.
- `tests/test_cst_r4k_matched_row_schedule_storage.py`: small/medium source matrix, matched GCC, original provenance failure, real create-only package test and negative mutated H-load witness.
- `paper/compiler_ir/R4K_SCHEDULE_MATCHED_STORAGE_ABLATION_20261106.md`: this paper protocol and original evidence record.

Engineering rules: no GitHub Actions, no changes to the canonical R2g/R4a emitters, no modifications to the R4a/R4d original input/result paths, no implicit native execution, no claim of sourced E3 hardware improvement from source E1, no unreported file additions.

## Next safe local workstation sequence

Run source-only plan first:

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m fpgai_research.experiments.cst_r4k_matched_row_schedule_storage
~~~

Then real GCC E1 and no-native tests:

~~~bash
python -m pytest -q \
  tests/test_cst_r4k_matched_row_schedule_storage.py \
  tests/test_cst_r4j_memory_instance_source_sites.py \
  tests/test_row_retained_fanout_cpp.py
~~~

Finally, ONLY if those tests pass, create a new independent E1/source-package study:

~~~bash
python -m fpgai_research.experiments.cst_r4k_matched_row_schedule_storage \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --output build/r4k_source_schedule_matched_storage_20261106_01
~~~

All commands are vendor-free; Python plan does not run GCC, while tests/study do run actual local GCC. **Do not rerun** either original 32/6 Vitis HLS case. Do not reuse an existing output folder. PR remains draft and paper remains **MAJOR REVISION / NOT SUBMISSION READY**.


**R4k workstation follow-up:** the author executed the E1 study at `build/r4k_source_schedule_matched_storage_20261106_01` after 40 focused tests passed. It is original source-only evidence; the subsequent R4l native experiment has not yet generated measurements. Original R4k source and packages remain unchanged except for future explicitly authorized, source-pinned native vendor outputs in R4l.
