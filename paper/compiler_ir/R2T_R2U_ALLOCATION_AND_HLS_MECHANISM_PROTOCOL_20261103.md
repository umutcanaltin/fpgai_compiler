# R2t and R2u — Exact tensor comparator and controlled HLS mechanism study

**Date:** November 3, 2026  
**Status:** R2t and R2u original experiments **WORKSTATION VERIFIED AND FROZEN** at clean source revision `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`. Original focused tests **29/29**, full pytest **1,565/1,565**. R2t global tensor optimum **36/36 proved**, **12/36 better than feasible greedy**, **36/36 equal tensor peak lower bound**, **36/36 element-exact source layouts better than tensor optimum**, **0 budget unknowns**, independent read-only source recomputation audit agreed. R2u **32/32 packages source-audited**, **32/32 native Vitis HLS processes completed** with **32/32 CSim E2**, **30/32 CSynth E3**, and **two E3 clock-budget failures at case_004 and case_012**. Physical E4 and routed memory remain UNKNOWN. The new R2v **read-only** 64-contrast analysis is implemented but not yet workstation verified. Earlier original studies remain unchanged. All original R2i/R2j/R2l/R2m/R2n/R2o/R2p/R2q/R2r/R2s source and vendor studies remain frozen, source-pinned and unmodified. The paper remains **MAJOR REVISION**.

## Completed original R2t/R2u evidence: exact outcomes and limitations

**R2t original commands (already executed; NEVER rerun original v1):** `build/r2t_tensor_exact_36_original_v1/` and `build/r2t_tensor_exact_36_readonly_audit_v1.json`. The 36-case summary reported **36 OPTIMAL_PROVED, 0 UNKNOWN_BUDGET**, **12** globally optimal contiguous tensor placements better than the greedy first-fit baseline, **36** optimal placements equal to the tensor peak lower bound, and **36** source element-exact allocations strictly smaller than the proved contiguous tensor optimum. This is the **committed-value source E0 memory model**; it is not physical BRAM, an emitted fused in-place design or a novel algorithm.

**R2u original commands (already executed; NEVER rerun original v1):** `build/r2u_hls_factorial_32_original_v1/` contains all 32 source packages and all original Vitis HLS invocations under revision `0c6d85f...`, 8 ns requested clock and 1.5 ns uncertainty. The user reported **32 E2/CSim-qualified, 30 E3/CSynth-qualified**, **case_004 and case_012** failed the effective clock-budget gate despite successful Vitis processes. All other 30 yielded source-attested HLS estimated resources. This is **not 32 E3 passes** and not a Vivado E4 campaign. There are **64 prespecified conditional factor pairs**, but 7 are adjacent to E3-unqualified arms (logical topology expectation, pending independent read-only analysis). **No additional vendor run is required**: use the new [R2v read-only original study analysis](R2V_READONLY_NATIVE_FACTORS_PROTOCOL_20261103.md) to verify the exact matched contrast population, per-factor effects and original CSynth XML hashes. Keep every R2t/R2u original result folder and invocation immutably.

**Commands below are preserved as historical experiment instructions**. Do not re-run the original v1 source/HLS preparation, Vitis HLS execution, or original R2t calculation against those same output paths. The R2v postprocessing is a separate NEW, create-only read-only analysis.

## Original evidence retained

R2s original revision `f38a98cb96a96097174204773462a53b8e6c02e2` passed **1,543/1,543 full regression** and completed **147 original source E0 cases** with 126 scalar-controlled Phi interventions. Committed-value source capacity: **102 reductions, 24 ties, 0 increases**, maximum 12 scalar slots. Hypothetical in-place first-MAC accounting: **0 reductions, 126 ties**. The reviewer-proposed bound `S_Phi <= R(w−1)` held on all **126 tested** interventions, but is **not a proof**. The largest-shape maximum saving is **0.389% of all declared mutable source arrays**, and the maximum absolute fraction across the finite study is **5.882%**. The separate read-only source recomputation audit agreed. Zero original GCC/Vitis/Vivado/learned-RL calls. These are E0 source accounting results, not physical FPGA outcomes.

R2r original revision `fbdd34ea363647dbea700223f923d0dad8547732`: **1,531 full regression tests**, six unmutated generated-C++ GCC baselines, **216/216 source-level mutants detected (192 shadow, 24 R2h checker)**, independently SHA/receipt audited without GCC replay. The historic native 72-case original study and selected six OOC routed comparisons remain **six BRAM ties**, mixed LUT/FF/DSP/WNS and **0 strictly clean DRC E4-qualified pairs** under requested **32 ns**. We must not reinterpret their physical observations from source-only evidence.

## R2t: Global **optimal feasible whole-tensor** reuse

The old R2q analysis compared committed-value scalar scratch allocation against disjoint storage, a **greedy feasible whole-tensor first fit**, and a tensor concurrency **lower bound**. The lower bound was not automatically realizable with contiguous tensor addresses. R2t implements a stronger, exact comparator:

- Each tensor has one contiguous block sized by the number of elements, live **from first element COMMIT to last element READ**, inclusive. **No splitting across elements.**
- Two tensors with overlapping source lifetimes cannot have overlapping physical block intervals; tensors that never overlap in time may reuse addresses.
- An exhaustive search enumerates **all global spatial orderings of the blocks**, placing each block at the earliest nonconflicting address under its spatial predecessors. Every feasible integer-base layout has a spatial ordering; left-justifying each block in that ordering cannot increase the peak memory span. Therefore **completed** search and matching source replay certify a globally feasible minimum for this exact, bounded whole-tensor problem.
- The exact result must lie between **the tensor concurrency lower bound and greedy first-fit**. The existing committed-value element-optimal slot count is at most the whole-tensor optimum. A source-event replay checks every actual COMMIT/READ against the returned whole-tensor address map.
- This is a classical fixed-lifetime layout search, **not a newly invented allocation algorithm**. It does not model FPGA RAM banking, port counts, source reader accumulators or in-place first-MAC storage.
- A finite `max_search_nodes` protects the reference study from exponential blowup. If exhausted before proving the optimum, it explicitly returns `UNKNOWN_BUDGET` and **NO certified optimal capacity**, rather than promoting the best feasible placement.

Files:
`src/fpgai_research/model_ir/tensor_exact_allocation.py`, `src/fpgai_research/experiments/cst_r2t_tensor_optimum.py`, `tests/test_cst_r2t_tensor_exact_allocation.py`. The tests include a **standalone brute-force base-enumeration oracle**, malformed-block controls, and a concrete temporal-block counterexample where **greedy=15** but **globally feasible optimum=14** source slots.

The experiment includes **36/36** original R2q graph/Phi configurations, but is **a separate NEW source revision**. Saved cases contain the original disjoint and element-exact capacities, greedy capacity, tensor concurrency lower bound, proven exact minimum or UNKNOWN, per-tensor physical bases, source-event digest and search budget. A create-only `cases.json`, `summary.json`, `comparisons.csv` and read-only same-revision source recomputation audit supply complete denominators. No GCC/vendor/RL run during this step.

## R2u: Controlled native Vitis HLS mechanism factorial

**Reason:** Previously, six selected pairs had lower Vitis HLS-estimated BRAM but **no routed Vivado BRAM difference**. C++ scratch element counts do not themselves prove HLS RAM binding, and address-table logic may distort estimates and latency. R2u explicitly varies source-memory and address-generation mechanisms using the same actual **R2h generated C++** and the canonical **Vitis HLS 2023.2** package writer/runner.

**Prespecified source population:** two graphs, each with all readers PE2: `R=2,(m,k,n,q)=(2,3,5,4)` and `R=4,(3,3,4,4)`. Within each graph the **exact functional model, weights, input, output reference, requested clock, fixed-point format, testbench and Tcl are matched across all arms**. Four binary factors, **16 arms per graph, 32 original HLS packages**:

| Factor | Control | Intervention | Interpretation |
|---|---|---|---|
| Scratch layout | Disjoint | Certified element-exact | Committed-value source scratch aliasing |
| Address expression | Static `r2h_slot_tensor[idx]` table | Equivalent source-level `switch(idx)` case lookup | Tests generated address-decoding structure; may change LUT/control cost, not just address count |
| Storage binding | Tool inference | `#pragma HLS bind_storage variable=scratch type=ram_2p impl=bram` | Requests BRAM dual-port; **does not prove tool honors directive** |
| Array declaration | Exact source capacity | Round declaration up to next multiple of 64 | Tests inferred storage quantization; unused padding may be optimized away |

The storage pragma is an AMD Vitis HLS **2023.2 UG1399** supported source directive; see <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/set_directive_bind_storage>. The source study uses a more discriminating requested **8.0 ns** clock, **1.5 ns** uncertainty, and **0.01 absolute** CSim tolerance rather than the old **32 ns / 0.1** conditions. All tool failures or nonqualified numerical outcomes remain explicitly **UNKNOWN/FAILED**, never silently scored as successful or 0 BRAM. The exact bit-level fixed-point comparison remains a separate obligation; 0.01 is still a tolerance, not proof of bit identity.

**Source preparation:** `cst_r2u_hls_mechanism_prepare` emits each `case_000`…`case_031` with source `model.cpp`, `tb.cpp`, `run_hls.tcl`, `manifest.json`, and a cohort-level SHA manifest. It fails closed if the source revision is dirty, a factorial factor is unsupported, a source SHA differs, or two variants accidentally emit identical C++ within the graph. It preflights all 32 source transforms before creating an original output directory. Read-only `--audit` rederives the exact compiler source for every arm, verifies Tcl/testbench/source/manifests and asserts complete matched controls.

**Numerical source controls:** `tests/test_cst_r2u_hls_mechanism.py` executes selected transformed C++ variants under real host GCC and the original R2h actual address/NumPy verifier **with the unchanged original certified logical scratch map**. A padded declaration cannot expand legal observable source addresses; printed witnesses must match actual source events. This is **host E1**, not HLS CSim/CSynth E3, physical resource measurement or proof of directive inference.

**Native execution and results:** `cst_r2u_hls_mechanism_execute` runs no vendor tools by default. It requires **BOTH** `--execute-native` and `--confirm-isolated-native`, source package verification and an original Vitis installation. Selected cases must have **no previous invocation/log/project artifacts**, and all are preflight-checked before executing the first tool process. The existing source-attested Vitis vendor runner, `summarize_case`, and native numerical/timing/resource qualification gate are reused unchanged. It records **32 case statuses** and **64 planned matched two-arm factor contrasts** (16 for each factor). Compare only where **both arms are E3-qualified**. Otherwise, HLS BRAM/LUT/FF/DSP/URAM/cycle deltas remain `null` and `UNKNOWN`, **not ties**.

All measurements will be **HLS estimates**, not routed BRAM, achieved clock, area, power or a physical E4 claim. Even an accepted BRAM source directive must be confirmed through the generated RTL/RAM binding report before stating its physical implementation. A factorial effect is only conditional on these **two static graphs** and their compiler code. No native data have yet been observed.

## Exact workstation verification commands

Use the existing clean `~/Desktop/github_projects/fpgai_research` checkout. **No GitHub Actions or tokens.**

```bash
cd ~/Desktop/github_projects/fpgai_research
git status --short
git pull --ff-only origin main
git rev-parse --short HEAD
python -m compileall -q src/fpgai_research experiments/compiler_ir tests

python -m pytest -q \
  tests/test_cst_r2t_tensor_exact_allocation.py \
  tests/test_cst_r2u_hls_mechanism.py \
  tests/test_compiler_ir_series.py

python -m pytest -q
```

Only after these tests pass, begin **new create-only source studies**, not any frozen original results:

```bash
python -m fpgai_research.experiments.cst_r2t_tensor_optimum \
  --output build/r2t_tensor_exact_36_original_v1 \
  --max-search-nodes 250000

python -m fpgai_research.experiments.cst_r2t_tensor_optimum \
  --study build/r2t_tensor_exact_36_original_v1 \
  --audit-output build/r2t_tensor_exact_36_readonly_audit_v1.json

python -m fpgai_research.experiments.cst_r2u_hls_mechanism_prepare \
  --output build/r2u_hls_factorial_32_original_v1

python -m fpgai_research.experiments.cst_r2u_hls_mechanism_prepare \
  --audit build/r2u_hls_factorial_32_original_v1

python -m fpgai_research.experiments.cst_r2u_hls_mechanism_execute \
  --study build/r2u_hls_factorial_32_original_v1 \
  --audit-native
```

The last command is **read-only** and should report **0 native original invocations, 0 E3 qualified and 64 UNKNOWN contrasts** before Vitis is run.

**Native HLS, explicitly opt-in:** after source preparation/attestation and confirming the actual installed `vitis_hls` toolchain, use:

```bash
python -m fpgai_research.experiments.cst_r2u_hls_mechanism_execute \
  --study build/r2u_hls_factorial_32_original_v1 \
  --execute-native --confirm-isolated-native \
  --start-case 0 --max-cases 1
```

Then inspect its native evidence and testbench/RTL warnings. Continue the remaining cases with **new, nonoverlapping** `--start-case N --max-cases M` ranges; do not rerun a case with an existing original invocation. Once all 32 have original results, run the **read-only** `--audit-native` command again. Do not force complete E3 qualification: measured unknowns and failures matter for causal interpretation. No Vivado is run by this R2u study.

The `experiments/compiler_ir/series.py` source-only quick and paper profiles include `tensor_exact_comparator_e0` and `hls_mechanism_factorial_source_e1` **without any `--execute-native`**. Do not launch the entire older research series simply to start these standalone experiments.

## Paper interpretation and remaining publication gates

**Pass of all new source tests is necessary, not sufficient.** R2t's exact tensor comparator may show the committed-value element packing advantage over the strongest feasible whole-tensor baseline, but that still follows standard interval-coloring and **does not establish a novel memory allocator**. R2u may explain some source/HLS memory-binding mechanism, but **cannot retrospectively explain all six physical BRAM ties without actual independently attested RTL/memory instance evidence**; no unmeasured data can be invented.

A complete publication-ready paper still needs: validated RTL/physical memory and timing evidence or an explicitly scoped negative translation case study; independent mechanistic comparison against Allo/StreamTensor/ST-Flow/CODO and related compiler artifacts; stronger source-level semantics and proof formulation distinguishing total event order from hardware cycles; and a transparent numeric precision test (the older HLS 0.1 tolerance is roughly 102 LSB at `ap_fixed<16,6>`). RL training remains frozen, and C3 action-interface compatibility is a **secondary engineering property**, not proof of a new RL method.

**Proposed narrowed scientific contribution:** a bounded compiler case study of *source-address refinement and tensor-value lifetime certificates for independently mapped accelerator operators*, with honest negative source-to-HLS-to-routing observations, rather than claims of universally novel Phi mapping, memory savings or physical performance.
