> **LATEST ORIGINAL EVIDENCE — 3 November 2026, superseding older 1/6 and 3/6 status notes below.** The author's workstation at code `6e87a4c42aaa30f6ad9e9be1ecccd7218c6ede99` passed **30 focused and 1,649/1,649 full pytest**. The original source cohort `f665b62ca81ee181a1447e3eddd23d7eba578186` has **6/6 original Vitis attempts, 6/6 numerically qualified CSim E2, 4/6 strictly E3-qualified CSynth**, and **2/3 originally prespecified complete matched E3 pairs**. **`case_002` and `case_003`** both failed their unchanged **9 ns effective HLS clock budget** with separate source-attested **9.181 ns** estimated clock reports; no E3 resource/latency comparison is valid for that pair. The two qualified independent-Phi interventions had **HLS BRAM18K Δ0 in both**, and *increased* worst HLS cycles: **624→911**, **2365→5436**; DSP, FF, LUT estimates also increased. All observations are **HLS**, not physical E4, proven general improvement, trained RL, or bit-exact all-input semantics. The new [R3c full-cohort publication result exporter/protocol](R3C_COMPLETE_CNN_HLS_PUBLICATION_RESULTS_20261103.md) is committed and **NOT yet workstation-tested**; it performs no vendor calls. Paper decision remains **MAJOR REVISION / NOT SUBMISSION-READY**; prior status notes below are historical snapshots.

> **Original workstation checkpoint — 3 November 2026 (later than earlier notes below).** On code `fb6747af88e09a68a763a85e66845042029c4967`, **45/45 focused and 1,633/1,633 full tests passed**. Original source cohort `f665b62ca81ee181a1447e3eddd23d7eba578186` remains 6/6 source packages SHA-attested and 12/12 GCC E1. Original native Vitis runs now cover **three** cases: `case_000` and `case_001` E2/E3-qualified; `case_002` E2-qualified but **FAILED_HLS_EFFECTIVE_CLOCK_BUDGET** at E3. The first matched `cnn_pool_dense` Phi contrast is qualified: estimated HLS BRAM18K Δ0, DSP +48, FF +1796, LUT +9146, worst cycles +287 (independent mapping vs scalar). This is **2/6 E3 and 1/3 complete two-arm HLS comparisons**; cases `003–005` unattempted. No new strict E4, Fmax or physical BRAM result. The new R3b narrow continuation past fully attested E3 clock failures and read-only rejected-clock timing diagnostics are implemented on `main` **but have NOT yet been tested on the workstation**; no new native run is claimed. The paper remains **MAJOR REVISION**, with novelty/baseline and physical evidence unresolved. [Original follow-up protocol](R3B_PUBLICATION_NATIVE_COMPLETION_AND_EVIDENCE_20261103.md).

> **Verified author's workstation checkpoint (3 Nov 2026; original source `f665b62ca81ee181a1447e3eddd23d7eba578186`):** 38 focused and 1,626 complete pytest passed; six original mixed CNN/Dense source packages and 12/12 actual GCC intermediate-value E1 checks were completed; the six original source package SHA audit passed. One original Vitis case, `case_000`, then completed in 31 seconds and qualified CSim E2 and CSynth E3, with **HLS-estimated** BRAM18K = 1. The five other R3a native cases and all three full matched E3 contrasts remain UNKNOWN. No R3a physical E4 measurement exists; earlier R2y DRC-warned route is not strict E4. **Publication decision remains MAJOR REVISION.** The new [R3b safe completion/evidence handoff](R3B_PUBLICATION_NATIVE_COMPLETION_AND_EVIDENCE_20261103.md) is committed but has **not** yet been workstation-tested. Older below-dated prospective notes are historical and should not supersede this verified checkpoint.

## 2026-11-03 — Paper publication gate: R2z source/ONNX CNN→Dense and R3a six-case original native HLS ready for workstation validation (UNVERIFIED)

**New source-only compiler/publication features committed, no workstation tests or native runs yet:** `compiler.compile_onnx_neural_chain(path,...)` imports a strict static ONNX Conv→pool/activation→Flatten→multi-Dense model through the existing ONNX frontend, canonical architecture resolver and canonical source emitter. It does not invent kernels or claim arbitrary ONNX/streaming support; pure MLPs still use the separately implemented `compile_row_chain`. Existing R2z mixed source compiler and original 3-network × 2-Phi source campaign remain pending original workstation verification. New [R3a original Vitis native protocol](R3A_CNN_DENSE_ORIGINAL_HLS_NATIVE_PROTOCOL_20261103.md), `src/fpgai_research/experiments/cst_r3a_conv_dense_native_hls.py`, and `tests/test_cst_r3a_conv_dense_native_hls.py` reuse the existing original `summarize_case`, exact E2 numeric and E3 effective HLS clock/resource `qualify_case` and source-attributed vendor runner, with 6 original CNN/Dense cases and 3 within-network scalar-vs-independent-Phi pairs. All attempted/missing/failed cases remain UNKNOWN unless actual vendor E2/E3 qualified. The native runner defaults DRY RUN; original tool execution requires BOTH explicit `--execute-native --confirm-isolated-native`, refuses reattempts. No Vitis/Vivado jobs launched by coding.

**One-command publication source gate added:** `bash scripts/run_compiler_ir_publication_source_gate.sh` compiles files, runs R2z/R3a and R2y focused tests, full pytest, prepares original six-case R2z study if absent, executes 12 *planned* full generated-C++ per-stage GCC traces and audits their original SHA/denominator, and runs R3a E2/E3 read-only report + native DRY RUN. It never runs native vendor tools and preserves any existing R2z original source folder for read-only audit. **DO NOT report 6/6 or 12/12 as completed until the user's console confirms those actual original checks passed.** Only then may the new opt-in R3a native HLS CSim/CSynth pilot begin.

**Current original R2y still 1/9 routed, first `case_000` DRC-warned NOT clean E4**; no physical positive result. Older R2i–R2x and frozen RL/CI unchanged. The newly checked [Nov 2026 public prior-art matrix](RELATED_WORK_COMPILER_IR_NOV2026_VERIFIED_MATRIX.md) explicitly acknowledges Allo/StreamTensor/Filament/CODO/SODA/hls4ml and confirms that standard mixed CNN/Dense model compilation is **engineering support, not novel by itself**. Independent prior-art artifact comparison, qualified native results, theorem boundaries and paper scientific novelty remain **publication blockers**. Paper status stays **MAJOR REVISION**. No submission, DOI or accepted venue yet.

## 2026-11-03 — R2z mixed Conv2D→NN/Dense end-to-end support IMPLEMENTED, workstation tests and original study PENDING

**User-requested feature implementation:** Existing research compiler already had separate Conv2D, MatMul/Gemm, activation/Flatten/MaxPool reference/codegen paths and a row-chain MLP, but no explicit source-attested unified mixed **CNN→Flatten→multi-Dense sequential compiler**. New `src/fpgai_research/model_ir/neural_chain.py`, `src/fpgai_research/compiler/neural_chain.py`, `src/fpgai_research/validation/neural_chain_host.py`, and public `compiler.compile_neural_chain` now reuse the SAME canonical `FunctionalModel/resolve/lower/verify_refinement/emit_hls` paths with independent Conv and Dense operator mapping. Static NCHW Conv2D (optional constant bias, grouped/dilation unsupported), ReLU/MaxPool/GlobalAveragePool, Flatten(axis1), MatMul, Gemm (bias/transB), Sigmoid/Softmax are admitted in bounded sequential networks only, with complete per-stage full-tensor contracts and fail-closed unsupported configurations. No independent second kernel emitter, physical memory/streaming/in-place fusion or unknown hardware QoR claim.

**New source-only actual generated C++ checks and source study, not yet run on workstation:** `tests/test_cst_r2z_conv_dense_neural_network.py`, `tests/test_cst_r2z_neural_campaign.py`, `experiments/cst_r2z_neural_network_fixtures.py`, `experiments/cst_r2z_conv_dense_network_campaign.py`. Three deterministic CNN/multilayer NN families × two scalar/independent Phi mapping policies = **six** complete network source packages, **two** deterministic samples each = **twelve planned original GCC E1 real emitted-C++ checks of EVERY intermediate stage** versus independent NumPy; existing `write_vitis_package` produces six source-only Vitis Tcl/testbench packages (10 ns, 1 ns uncertainty, 0.01 absolute reference criterion). Source receipts and create-only output receive a separate read-only same-source audit with **zero new GCC/Vitis/Vivado calls**. No original R2z case counts have been measured yet; the software source and tests are committed but unverified on the workstation. ONNX Conv→Relu→Flatten→Gemm integration is included in pytest if ONNX is installed. See [R2z source/network protocol](R2Z_MIXED_CONV_DENSE_NEURAL_NETWORK_PROTOCOL_20261103.md).

**Original R2y physical pilot partially executed (verified user terminal):** source preparation at `303209deaf660334a798337c187cff8dbaae6e27` passed **30/30 focused**, **1,595/1,595 full tests**, prepared and audited **9 original Vivado OOC packages** and 7 matched contrasts; dry run made zero calls. First original opt-in Vivado `case_000` ran with exit 0, source-attributed routing and timing accepted **only as `ROUTED_WITH_DISCLOSED_DRC_NOT_E4`**, strictly `E4=false`. Remaining 8/9 Vivado original physical cases not attempted, **0 clean DRC E4 cases/pairs**. Exact raw DRC rules/resource/WNS not yet supplied; do not invent them. **Never rerun original R2y case_000 or overwrite the case's native report/receipt**. Previous original R2i–R2x source/vendor studies are frozen; R2x original 1,588/1,588, 32/32 static type graphs and 64/64 pairs; earlier 6 routed BRAM ties with 0 strictly clean E4 still separate. RL environment and GitHub Actions unchanged.

**Publication status: MAJOR REVISION.** R2z closes bounded mixed-network source coverage but not trained-model accuracy, bit-exact fixed-point, stream/port legality, physical memory, complete neural topologies or scientific novelty. Next gates: local R2z focused/full tests and six-source original study; independent exact in-place source implementation; native fixed-point HLS and clean DRC routed matched pairs; and verified competing prior-art capabilities. None of these unexecuted gates may be described as passed.

## 2026-11-03 — R2x ORIGINAL static hierarchy VERIFIED/FROZEN; prospective R2y OOC Vivado pilot IMPLEMENTED but NOT TESTED

**R2x original read-only source evidence** at clean analysis revision `e5ea6a14d279be786492f338bdf1e51ba684ba73`: **23/23 focused and 1,588/1,588 full pytest passed**. The create-only `build/r2x_original_r2u_static_hierarchy_v1/` completed, preserving the original R2u Vitis source revision `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`: **32/32 original RTL source SHA attested**, **32/32 static top/type graphs known**, **64/64 matched RTL source pairs type-graph analyzable**, **0 new Vitis/Vivado calls**. Recognized source-level memory-named reachable module types / recognized type instances changed from **10/10 table to 6/6 switch** in two-reader graphs and **16/16 to 8/8** in four-reader graphs. This is static Verilog subset/type-level reachability, **NOT actual elaborated or physically mapped RAM cells**. Original 14 qualified address source-lookup HLS BRAM estimates remained all ties. R2x original v1 is FROZEN.

**New R2y strictly source-matched Vivado OOC physical pilot code/tests/protocol are COMMITTED, but NOT workstation-tested and NOT executed.** `src/fpgai_research/experiments/cst_r2y_controlled_physical_pilot.py` / `tests/test_cst_r2y_controlled_physical_pilot.py` with [R2y predeclared original Vivado pilot protocol](R2Y_NINE_CASE_STRICT_E4_PHYSICAL_PILOT_20261103.md). Reuses canonical original HLS source/RTL/ROM SHA audit and existing Vivado OOC project runner; prepared original source cohort of **9 E3-qualified original 8ns Vitis arms** `case_000,001,002,008,016,017,018,020,024`, **7 prespecified matched pairs** holding graph/Phi/input/TB/Tcl and all but one source factor fixed. Includes graph 1 storage `016→024` whose original HLS **BRAM_18K estimate decreased by 1**, plus positive/negative control binding/padding/address pairs. Original `case_004`/`case_012` E3 clock failures are **excluded**, never promoted to a valid physical comparator. Vivado packaging remains **SOURCE-ONLY, CREATE-ONLY**, fresh `build/r2y_vivado_pilot_9_original_v1/` outside original frozen folders. Actual vendor use is **opt-in** requiring both `--execute-native` and `--confirm-isolated-native`; missing/attempted cases cannot be rerun. Strict E4 requires source/ROM/report SHA integrity, complete route, WNS≥0 and **verified ZERO DRC violations/critical warnings** for both arms. DRC-warned observations must never be declared strict E4. All prospective R2y physical BRAM/LUT/FF/DSP/WNS outcomes are **UNKNOWN** until local native execution; no board bitstream/Fmax/power results. The six older routed R2m BRAM ties and 0 clean-DRC E4 remain independently frozen at a different 32ns HLS protocol and must not be pooled.

Paper remains **MAJOR REVISION** pending workstation verification and original clean E4 or explicit negative physical translation study, bit-exact numeric/actual source-to-RTL proof and external prior-art artifact novelty. Freeze R2i..R2x original data, RL environment and workflow settings; NO new GitHub Actions or vendor calls made by assistant.

## 2026-11-03 — R2w ORIGINAL 32/64 RTL SHA FORENSICS VERIFIED; R2x STATIC HIERARCHY CODE UNVERIFIED

**R2w original source-frozen read-only experiment completed at analysis revision** `bb98983a438efe9e65c0204041895353730f964e`: **22 focused/22** and **1,580 full/1,580 pytest passed**, then the new create-only `build/r2w_original_r2u_rtl_memory_forensics_v1/` finished against the original R2u Vitis source revision `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`. **All 32/32 original generated RTL and ROM case SHA inventories matched their original Vitis invocation receipts; all 64/64 source-matched RTL pairs were attested**, even though **57/64** HLS-estimated resource comparisons E3-qualified and 7 remained E3 UNKNOWN. No additional Vitis/Vivado calls.

The original per-case R2w lexical file inventory split by address expression: two-reader graph **table 23 RTL files / 10 memory-named / 7 scratch-named**, switch **24 / 6 / 7**; four-reader graph **table 30 / 16 / 9**, switch **32 / 8 / 9**. These are **original HDL source filename/token counts, not actual RAM instances or physical memory counts**. The 14 eligible table/switch HLS BRAM estimate comparisons tied even though their RTL file organization differs. Preserve original R2u/R2v/R2w study directories; never overwrite or rerun any original v1 path.

**New R2x source-only static RTL hierarchy program, tests and protocol COMMITTED but NOT WORKSTATION TESTED/EXECUTED:** `src/fpgai_research/experiments/cst_r2x_rtl_hierarchy_analysis.py`, `tests/test_cst_r2x_rtl_hierarchy_analysis.py`, [R2x protocol](R2X_STATIC_HLS_VERILOG_HIERARCHY_PROTOCOL_20261103.md). It reuses read-only original R2w SHA checks, scans known `module` definitions and a subset of named type instances, tracks *type-level* reachability from the expected HLS `fpgai_model` top, counts reachable scratch/memory-named modules and limited literal reg-array declarations, and compares all 64 original source factor pairs. Unsupported/ambiguous Verilog yields **UNKNOWN**, not false zero-memory conclusion. No Vitis, Vivado, RL or GitHub Actions. **Static module types are not elaborated hardware instances, physical BRAM cells or proof of memory port legality.** R2x new create-only output: `build/r2x_original_r2u_static_hierarchy_v1/`, only after focused/full pytest pass. Paper remains **MAJOR REVISION** pending native E4 or a carefully scoped negative physical translation, peer-artifact capability comparison and emitted in-place control.

## 2026-11-03 — Original R2v 64 HLS factor pairs VERIFIED/FROZEN; R2w RTL forensics IMPLEMENTED but NOT TESTED

**Original workstation evidence:** R2v analyst revision `23e998a0553e3eec6f05c5d3cd09a9b4a81af83e` passed **26/26 focused and 1,572/1,572 full regression tests**. Its **create-only and now frozen** `build/r2v_r2u_native_factorial_review_v1/` read-only output was computed from the preserved original R2u vendor source revision `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`, performing **0 new Vitis/Vivado calls**. Exactly **32/32 original CSynth XML files had matching original invocation SHA receipts**; **32/32 CSim E2**, **30/32 effective-clock-qualified CSynth E3**, **2 failed E3 clock budget** (`case_004` and `case_012`, both switch/address-auto/exact on graph 0). Of **64 planned matched HLS factor contrasts**, **57 were E3-qualified and 7 UNKNOWN**, never counted as ties.

**Verified HLS-estimated BRAM_18K signs (intervention minus control):**

| Factor | Reduced | Tied | Increased | E3 UNKNOWN |
|---|---:|---:|---:|---:|
| Source scratch layout disjoint→element-exact | **2** | **13** | 0 | 1 |
| Address static table→switch | 0 | **14** | 0 | 2 |
| Binding automatic→requested BRAM `ram_2p` | 0 | **10** | **4** | 2 |
| Padding exact declaration→next multiple of 64 | 0 | **10** | **4** | 2 |
| **Total** | **2** | **47** | **8** | **7** |

The two source-layout BRAM-estimate reductions occurred **only** in the four-reader graph (2/8 qualified storage pairs); the two-reader graph had 0/7. Binding/padding increased HLS-estimated BRAM in 4/14 qualified comparisons each. **No physical BRAM, routing, Fmax or power was measured**. The older original six warning-disclosed routed BRAM ties / 0 strict clean-DRC E4 remain unchanged and are not explained by these HLS estimates.

**New R2w code/tests/protocol are COMMITTED but NOT yet workstation verified or executed:** `src/fpgai_research/experiments/cst_r2w_rtl_memory_forensics.py`, `tests/test_cst_r2w_rtl_memory_forensics.py`, [R2w source-attested original HDL/ROM forensic protocol](R2W_SOURCE_ATTESTED_RTL_MEMORY_FORENSICS_20261103.md). This explicitly read-only, **no vendor call** next experiment checks the original 32 Vitis `rtl_output_sha256` and `rom_output_sha256` receipt maps, hashes original CSynth text reports where available, and provides lexical scratch/memory module-name/primitive-token inventory plus all 64 matched RTL evidence pairs. Missing/unsupported/symlinked/altered cases cannot be promoted to physical evidence; comments or module names containing RAMB identifiers are **not** physical netlist RAM cells. Source R2t/R2u/R2v and all prior original studies stay frozen. **Do not rerun any R2u Vitis case or original R2v analysis v1 path.** The next original output, if tests pass, must be **new create-only** `build/r2w_original_r2u_rtl_memory_forensics_v1/`.

Remaining publication blockers: actual inferred RAM and memory-port connectivity/RTL semantics, independently qualified physical E4 or explicitly scoped negative translation results, proper generated fixed-point numeric tests, emitted first-MAC in-place control and independent capability comparison to current related work. No trained RL improvement or full product completion claim. Paper still **MAJOR REVISION**.

## 2026-11-03 — Original R2t/R2u VERIFIED and FROZEN; R2v READ-ONLY FACTOR REVIEW PENDING

**Clean original source revision:** `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`. Workstation local **29/29 focused** and **1,565/1,565 full pytest** passed. R2t exact tensor original **36/36 OPTIMAL_PROVED**, **0 UNKNOWN_BUDGET**, **12/36 exact contiguous-tensor optima better than feasible greedy first-fit**, **36/36 exact tensor optima equal the whole-tensor concurrency lower bound**, **36/36 element-exact source allocations strictly better than the new proven optimal contiguous tensor assignments**. The separate read-only original E0 recomputation audit agreed. This establishes only bounded source-event memory theory, not FPGA physical BRAM, novel interval coloring or an emitted first-MAC in-place accumulator.

R2u original prospective **32-case 2-graph × 2^4-factor source-verified Vitis packages**, all 32 original native vendor invocations **exited 0** and all **32/32 original Vitis CSim E2 qualified**. **30/32 E3 qualified** under the strict effective 8 ns clock (1.5 ns uncertainty), 0.01 numerical tolerance and complete HLS estimates; **case_004** and **case_012** FAILED_HLS_EFFECTIVE_CLOCK_BUDGET despite successful CSim. They are **not E3-qualified**, and their resource changes are **UNKNOWN rather than ties**. Original native physical BRAM/Fmax remain `null`: **no new Vivado E4**. These are HLS estimated resources, not realized RAMB primitives or measured power. The earlier six original routed BRAM ties with zero strict clean-DRC E4 pairs remain separately frozen and must not be conflated with new 8 ns R2u HLS estimates.

**New R2v read-only post-HLS analysis code/tests committed, NOT yet workstation-verified.** `src/fpgai_research/experiments/cst_r2v_hls_factorial_review.py` and `tests/test_cst_r2v_hls_factorial_review.py` re-audit original source hashes, original source/Tcl/testbench/CSynth/receipt lineage, and ALL **64 preplanned controlled factor contrasts** at the original pinned revision **without launching any vendor tool**. The two timing-failed arms produce **7 logically impacted conditional contrasts**, leaving **57 expected E3-qualified contrasts**, pending actual R2v audited results. Per-factor/per-graph resource sign and missing-data denominators will be retained. The output is a **new, create-only** `build/r2v_r2u_native_factorial_review_v1/` directory; NEVER rewrite `build/r2u_hls_factorial_32_original_v1/`. See [R2v read-only protocol](R2V_READONLY_NATIVE_FACTORS_PROTOCOL_20261103.md) and [updated R2t/R2u original protocol](R2T_R2U_ALLOCATION_AND_HLS_MECHANISM_PROTOCOL_20261103.md).

**Full paper remains MAJOR REVISION** pending verified original R2v factorial causal interpretation, HLS RAM/RTL evidence rather than source pragma speculation, stronger all-input/fixed-point/physical checks and external peer-source novelty comparisons. The original R2i/R2j/R2l/R2m/R2n/R2o/R2p/R2q/R2r/R2s and R2t/R2u **source and native artifacts are frozen**. The `rl_environment/` remains frozen; **no GitHub Actions or native rerun**.

## 2026-11-03 — Multi-sprint R2t/R2u implemented, **WORKSTATION TESTS & ORIGINAL MEASUREMENTS PENDING**

**All previously original evidence remains frozen:** R2s source revision `f38a98cb96a96097174204773462a53b8e6c02e2`, **1,543/1,543 full pytest**, **147 source configurations**, 126 paired Phi contrasts **102/24/0 committed-value reduction/tie/increase**, **0/126 hypothetical first-MAC in-place reductions**, **0/126 observed violations** of the reviewer-proposed unproved `S_Phi<=R(w-1)`, source-only read-only recomputation audit. R2r `fbdd34ea363647dbea700223f923d0dad8547732`, **216/216 targeted C++ mutants detected** (192 source-injected dynamic shadow, 24 original R2h checker; six unmutated GCC passes). Original R2i/R2j/R2l/R2m six selected routed BRAM ties and mixed LUT/FF/DSP/WNS at 32 ns remain **0 strict clean-DRC E4**. Do not rerun/rewrite originals; `rl_environment/` remains frozen.

**R2t new source exact tensor baseline, NOT YET TESTED/EXECUTED on workstation:** `src/fpgai_research/model_ir/tensor_exact_allocation.py`, `src/fpgai_research/experiments/cst_r2t_tensor_optimum.py` and `tests/test_cst_r2t_tensor_exact_allocation.py`. Exact exhaustive search of global physical spatial order of tensor blocks with contiguous addresses under the same fixed COMMIT→last-READ source intervals. Independent direct brute-force base oracle for small examples (reference greedy **15** vs exact **14**), explicit finite search budget that returns `UNKNOWN_BUDGET` not falsely optimal, replay of every actual source event. Prospective full **36 R2q-matched graph/Phi source cases** with new source revision, create-only CSV/JSON and read-only exact same-source recomputation audit. **No original optimum counts until executed.** This is a classical feasible whole-tensor comparator, not a novel allocation algorithm, in-place memory, E3 or hardware BRAM measurement.

**R2u new controlled HLS mechanism factorial, NOT YET TESTED/EXECUTED on workstation:** `src/fpgai_research/experiments/cst_r2u_hls_mechanism_prepare.py`, `src/fpgai_research/experiments/cst_r2u_hls_mechanism_execute.py`, `tests/test_cst_r2u_hls_mechanism.py`. Prespecified **32 source HLS packages** = two static fanout graphs × 2 disjoint/element-exact layouts × 2 table/switch-case address expressions × 2 automatic/ram_2p-BRAM binding requests × 2 exact/padded-to-64 array declarations. All 16 arms per graph share exact weights/model/Phi/input/testbench/Tcl. New **8 ns requested clock**, **1.5 ns uncertainty**, **0.01 numerical tolerance**. The original R2h E0 source slot map and real GCC numeric/address checker are reused in regression controls, native measurement separate. Source-attested, create-only packaging and read-only re-derivation audit; native Vitis HLS opt-in **only with BOTH `--execute-native` and `--confirm-isolated-native`**, refuses existing original outputs. The read-only E2/E3 gate preserves **64 matched binary factor contrasts** and all failed/UNKNOWN arms; HLS resource/cycle estimates must **never** be presented as routed physical BRAM or achieved Fmax. No new HLS measured results or physical data currently exist. The tool may optimize unused padding or ignore a binding request—report truthfully.

**Cross-project safety:** R2t/R2u source-only stages are in the existing quick and paper research-series plans but **NO implicit native execution** was added. No original vendor/source studies, RL environment, or GitHub Actions modified. Source-level mathematical assumptions and classical allocation proof are stated in [formal E0 source-event scope](FORMAL_SOURCE_EVENT_REFINEMENT_AND_TENSOR_BASELINE_20261103.md). Exact workstation test, original source studies, opt-in native execution and full scientific limitations: [R2t/R2u protocol](R2T_R2U_ALLOCATION_AND_HLS_MECHANISM_PROTOCOL_20261103.md). The paper remains **MAJOR REVISION** pending genuine local tests, HLS results, empirical mechanisms, independent prior-art capability gaps and/or stronger qualified physical results. **Do not claim the full product/paper is finished.**

## 2026-11-03 — R2s ORIGINAL SOURCE SCALING 147/147 VERIFIED and FROZEN (supersedes earlier prospective entries)

Clean original source revision **`f38a98cb96a96097174204773462a53b8e6c02e2`**, following schema fixes, passed **19/19 focused tests** and **1,543/1,543 full pytest**. The original **create-only** `build/r2s_phi_scaling_147_source_v1/` and separate **read-only** `build/r2s_phi_scaling_147_source_audit_v1.json` completed successfully, with **147/147 source configurations** (seven shapes × 2/3/4 reader counts × seven PE1/PE2/PE4 policies) and **126/126 paired Phi contrasts**. Audit recomputed all original E0 source capacities/CSV observations on exactly the same Git revision and agreed; audit did **not** independently execute GCC/RTL. **No original GCC, Vitis, Vivado or RL calls**.

Exact source population: **committed-value** Phi reductions **102**, ties **24**, increases **0**, maximum saved **12 source scalar slots**; **hypothetical in-place first-MAC** reductions **0**, ties **126**, increases **0**, max **0**. All **126/126** committed-value source contrasts satisfied the **reviewer-proposed, unproved** `S_Phi <= R*(max_reader_PE-1)` under the one fixed serial source schedule, with **zero violations**. Across all contrasts, largest absolute reduction as a fraction of complete original source mutable arrays was **5.882352941176471%**; for the largest fixed shape `(16,2,16,16)`, maximum was **0.38910505836575876%** (R4/allPE4, 12/3084 declared scalar elements, source scratch 1036→1024). A source scalar slot/nominal 16-bit scalar declaration is **not a routed FPGA BRAM cell, port, clock cycle, timing, or power measurement**; the fused in-place model is **not generated C++**. These data support a bounded, model-dependent source interval relation only. They do **not** establish a universal Phi theorem or a hardware/QoR benefit. All original R2s v1 files are **now frozen** and must not be re-run or overwritten, as are earlier R2i/R2j/R2l/R2m/R2n/R2o/R2p/R2q/R2r studies.

Prior `R2s pending` notes below document the state **before** the successful local run and do not describe current status. Next scientific gates: prove/refute the bound analytically under exact event semantics; feasible globally optimal whole-tensor allocation (rather than merely greedy first-fit); actually emit and GCC-test in-place reader source; controlled Vitis CSynth addressing/array-padding/storage binding experiment with proper numeric criteria; prior-art capability/novelty evaluation. Scientific paper remains **MAJOR REVISION**; RL-compatible C3 is an interface, not a trained RL result. [Full R2s original protocol](R2S_PHI_SCALING_SOURCE_PROTOCOL_20261103.md).

## 2026-11-03 — R2s Phi scaling law source-sensitivity code IMPLEMENTED; original experiment NOT yet run

Original R2r source revision `fbdd34ea363647dbea700223f923d0dad8547732` is **verified and frozen**: 22 focused and 1,531 full pytest PASS, six original instrumented GCC baselines, **216/216** deliberately mutated generated C++ programs detected (**192 dynamic shadow ownership/read count, 24 earlier R2h source/numeric checker**), zero invalid/preflight/unknown/surviving mutants, 222 GCC attempts. Read-only audit checked all saved mutant source SHAs and populations but performed **zero** new GCC executions. This is bounded mutation adequacy, not general refinement/RTL safety.

**New prospective R2s research code and tests**: `src/fpgai_research/experiments/cst_r2s_phi_scaling.py` with `tests/test_cst_r2s_phi_scaling.py` and [prespecified R2s protocol](R2S_PHI_SCALING_SOURCE_PROTOCOL_20261103.md) explicitly uses existing R2f source events, R2h E0 layout and R2q dual lifetime accounting to evaluate **147 configurations = seven shapes × R{2,3,4} × seven specified PE1/2/4 policies**, and **126** scalar-controlled non-scalar reader Phi contrasts. It directly TESTS (and preserves violations of) the REVIEWER-PROPOSED inequality `committed_source_scratch_saved <= R*(max_reader_pes-1)`; **this is a conjecture under one static source issue policy, NOT a theorem or FPGA result**. Includes the full bounded shape (16,2,16,16), exact scalar baseline, all declared accumulator/counter array counts, hypothetical first-MAC in-place liveness, source-event SHA, disjoint/tensor/element capacities, exact signed delta percentages, create-only JSON+CSV, same-revision read-only source recomputation audit. No GCC/Vitis/Vivado/RL calls. The quick local research plan uses one shape (21 configurations), paper plan all seven (147).

**DO NOT report any R2s outcome before local verification.** R2q frozen original `7890c6f363cc9dd61c8bfa29ae121107ee3d86de` remains 18/27 committed-value reductions against scalar, 0/27 hypothetical in-place reductions. Original R2r/R2q source and all vendor/native outputs unchanged. Paper remains **MAJOR REVISION**; scaling alone cannot substitute for independent prior-art proof, stronger tensor optimum, actual in-place emitted source or HLS binding mechanism.

## 2026-11-03 — R2r ORIGINAL 216 generated-C++ mutations WORKSTATION VERIFIED (source revision frozen)

The user's clean original checkout `fbdd34ea363647dbea700223f923d0dad8547732` passed **22/22 focused** and **1,531/1,531 full pytest tests**. Then the original **create-only** `build/r2r_emitter_shadow_216_mutants_original_v1/` and read-only `build/r2r_emitter_shadow_216_mutants_audit_v1.json` were produced from that exact source revision. **Six unmutated instrumented C++ baselines qualified**, followed by **216/216** distinct prespecified actual generated-C++ mutations (nine fault families, four payloads, six graphs). Of the 216 mutants, **192** were detected by *actual source-instrumented slot-ownership/read-completeness shadow checks*, **24** by the existing independent R2h GCC/NumPy source checker, **0 survived**, **0 were compile-invalid**, **0 were preflight-invalid**, and **0 unexplained runtime failures** occurred. **222 original GCC attempts** were reported. The separate read-only audit confirmed the 216 mutant source SHA/identity/denominator receipts and category populations, without running GCC again. **Zero Vitis/Vivado calls, zero RL training.**

Scientific conclusion: **stronger bounded E1 mutation adequacy for the nine specified compiler-error families and six generated fanout source graphs**, not arbitrary program correctness, independent binary execution instrumentation, proof of all possible alias errors, all-input equivalence, FPGA hardware ports/timing, BRAM savings or scientific priority. The shadow checker derives obligations from the fixed fanout topology and instrumented generated source; it does not use the printed source event witnesses to decide physical slot ownership, but still shares the static model/shape. Preserve both original R2r v1 outputs **immutably**. R2q original `7890c6f363cc9dd61c8bfa29ae121107ee3d86de` remains separately frozen with 36 E0 cases, Phi 18 reductions/9 ties in committed source vs 0 reductions/27 ties under hypothetical in-place accounting.

Next scientific blockers: **R2s scaling/bounded Phi upper-limit falsification**, the missing feasible tensor-optimum and in-place emitted C++, mechanism-level controlled Vitis CSynth storage binding/padding/addressing, and prior-art capability comparisons. C3 RL-compatible decision interface remains secondary (no learned-policy benefit). **No GitHub Actions/tokens; no original experiment outputs modified.** [R2r exact protocol](R2R_EMITTED_SOURCE_SHADOW_MUTATION_PROTOCOL_20261103.md).

## 2026-11-03 — R2r emitted-source dynamic shadow/216-GCC-mutation sprint IMPLEMENTED, unverified; R2q original frozen

Original R2q source revision `7890c6f363cc9dd61c8bfa29ae121107ee3d86de` and its immutable `build/r2q_adversarial_accounting_36_v1` / `build/r2q_adversarial_accounting_audit_v1.json` are **WORKSTATION VERIFIED**: **17/17 focused, 1,516/1,516 full pytest**, 36/36 source-memory configurations, 27 Phi contrasts **18 committed-value savings + 9 ties versus 0 hypothetical first-MAC savings + 27 ties**; never re-run/overwrite them. C1 capacity gain is lifetime-convention dependent, not hardware QoR or independent optimization novelty. R2n/R2o/R2p original source and R2i/R2j/R2l/R2m original vendor/physical data remain frozen.

**New R2r code, tests and experiment planning committed; workstation tests and 216-mutant original experiment are PENDING.** The R2h real generated C++ receives an extra *source-injected dynamic shadow memory* that checks actual scratch STORE/LOAD slot ownership, the graph-derived number of reads per element and complete eventual release—**independent of the program's own printed `FPGAI_SLOT`/typed event labels**. A prespecified nine-fault-family × four-payload × six-static-graph cohort attempts **216 real C++ source GCC mutants**, plus six nonmutated GCC baselines; the existing R2h real source/numerical verifier is retained as a separate checker. Per-mutant original C++, source SHA and actual checker outcome are preserved even when mutants compile-invalid, are caught, remain unknown or survive. Preflight-only rejections are **not** scored as GCC detection. The separate read-only receipt/SHA and deterministic source-regeneration audit **does not rerun GCC**.

The quick paper-series profile uses one graph × 4 source mutations; the full paper profile uses the original six × 36 matrix. The focused local pytest checks a one-graph × 36 real GCC mutant population in an ephemeral test directory before original experiment execution. **No new original R2r results can be claimed until actual local original-source receipts appear**. No GitHub Actions, Vitis/Vivado, training/changes to `rl_environment/` or edits to frozen source experiments. Review [R2r exact protocol/commands](R2R_EMITTED_SOURCE_SHADOW_MUTATION_PROTOCOL_20261103.md). The paper remains **MAJOR REVISION, not submission-ready**; source-only mutation adequacy cannot prove all-input RTL or hardware correctness.

## November 3, 2026 — Original R2q source accounting COMPLETED and read-only recomputation VERIFIED

**Original source revision FROZEN:** `7890c6f363cc9dd61c8bfa29ae121107ee3d86de`. User workstation reported **17/17 focused** and **1,516/1,516 full regression pytest PASSED**. Then it ran new create-only `build/r2q_adversarial_accounting_36_v1/` with **36/36** fixed fanout source configurations and separately ran `build/r2q_adversarial_accounting_audit_v1.json`. The read-only E0 source/revision/CSV recomputation agreed exactly. **No new original GCC compilations/executions, Vitis, Vivado or RL training**, so these are theoretical SOURCE-memory accounting results, not new executable semantics or physical measurements.

The original outcome denominators are **36/36 disjoint-to-element-exact committed scratch savings**, **32/36 feasible tensor first-fit layouts improved over disjoint**, **36/36 element-exact improved over feasible tensor first-fit**, **36/36 element-exact improved over the optimistic tensor concurrency lower bound**. That last lower bound is *not* an achieved tensor-granular allocation.

Across the **27** original paired Phi interventions:
- **Committed-value COMMIT→last-READ intervals:** 18 reduced, 9 tied, 0 increased; largest absolute change **2 unit slots**.
- **Hypothetical reader-output in-place first-MAC→last-READ intervals:** **0 reduced, 27 tied, 0 increased**; maximum absolute change **0**.

This **confirms the central independent-review warning**: the Phi-dependent scratch advantage depends on the chosen memory/lifetime accounting convention. The alternative lifetime peak is NOT current emitted R2h source, which allocates separate complete accumulator and scratch arrays, and is NOT a tested fused C++/RTL implementation. Fixed-order optimal element coloring remains a classical algorithm, not automatically scientific novelty. Stronger E1 emitted alternative, emitted-source mutation, tensor optimum, HLS binding factorial, independent prior-art capability comparisons and C3 policy-benefit studies remain open. Do not claim hardware resource improvement, general graph semantics, novel RL policies or publication readiness from R2q.

For original source/protocol and transparent limitations, see [R2q independent review triage](INDEPENDENT_REVIEW_TRIAGE_AND_R2Q_PROTOCOL_20261103.md); preserve original R2i/R2j/R2l/R2m physical and R2n/R2o/R2p source studies.

## 2026-11-03 — R2p original GCC VERIFIED; independent Claude review MAJOR REVISION; R2q source accounting sprint IMPLEMENTED, execution pending

**R2p original immutable observation:** Local checkout `dac395598f2ab96bc9b7a41255b0529c72b22593` passed **29 focused** and **1,506 full pytest**, then original GCC **6/6 candidate actions** across sequential and two-reader fanout: **3 E1_GCC_QUALIFIED, 2 REJECTED, 1 UNKNOWN_E0**, 3 GCC E1 attempts, 0 failures. Separate source/graph/input SHA read-only receipt audit agrees; no Vitis, Vivado or RL training. Preserve `build/r2p_unified_c3_original_gcc_v1/` and `build/r2p_unified_c3_original_gcc_audit_v1.json` without rerunning, rewriting or deleting.

**Independent scientific review (`CST_INDEPENDENT_REVIEW_20261103.md`, supplied by author): MAJOR REVISION.** The primary novelty claim is not yet scientifically distinct: disjoint baseline has no reuse; Phi-sensitive source-slot savings depend on committed-value versus hypothetical in-place reader-output accounting; emitted R2h source also declares full-sized `acc`/counter arrays; interval coloring is classical; Phi labels in C3 are nearly constant among admitted mappings; native 32 ns WNS and the post-HLS-selected six warning-only routed BRAM ties do not establish physical performance. The reviewer reports a larger 4,115-intervention sweep, but those figures are **reviewer-derived, not original R2q results**; licensed Vitis/Vivado raw reports were unavailable to the reviewer.

**New R2q code written, NOT executed locally yet:** `model_ir/source_allocation_baselines.py` / `experiments/cst_r2q_accounting_baselines.py` / `tests/test_cst_r2q_source_accounting.py`. A prospective 36-case **source-E0** study compares no-reuse, **feasible** whole-tensor first-fit, a **lower-bound only** tensor concurrency peak, classical element-optimal source slots and a separate **hypothetical** reader-in-place first-MAC lifetime; counts actual separate R2h accumulator+counter arrays as well. It records 27 mapping interventions, creates `build/r2q_adversarial_accounting_36_v1/` only once, and supports an independent read-only *source recomputation* audit, no GCC/Vitis/Vivado or RL. Original 17→16 versus hypothetical 19→19 reviewer counterexample is a falsifiable test, not an asserted new experimental fact. Series quick/paper profiles now include `accounting_baseline_sensitivity_e0`. The active manuscript was revised to mark source novelty unresolved and to preserve all physical BRAM/LUT/FF/DSP/WNS measurements without superiority claims. Its 336 historical sprint-note lines were archived verbatim, not deleted.

Scientific gate and exact local commands: [Independent-review triage and R2q protocol](INDEPENDENT_REVIEW_TRIAGE_AND_R2Q_PROTOCOL_20261103.md). **No GitHub Actions/tokens, no frozen `rl_environment/` changes, no original native R2i/R2j/R2l or R2n/R2o/R2p experiment implementation changes.** This work addresses only the first portion of reviewer experiment A; separate in-place emitted C++, stronger tensor-optimum, large scaling study, mechanism CSynth, emitter-level mutation and proper novelty comparison are still open.

## 2026-11-03 — R2p policy-neutral COMMON C3 interface and source-only study IMPLEMENTED; workstation verification PENDING

**Previously completed R2o original evidence (FROZEN):** Original clean Git checkout `ce8083c0ad5b827232e1e67322811b25edd3ac79`, **27 focused + 1,484 full local pytest PASS**, 3 static independent reader fanout graphs (degrees 2,3,4), 12 source-candidate actions complete, **9 actual GCC E1-qualified, 3 rejected, 0 unknown, 0 failed**. Independent read-only source/manifest/stdout SHA receipt audit agreed, with 102/210/372 runtime address witnesses per original E1 source execution for each respective graph. Separate R2n prior source evidence was **3 E1/3 reject/3 E0 UNKNOWN across 9 actions**. Neither demonstrates FPGA bank ports, E4 or trained RL.

**New R2p prospective source-only sprint:** `src/fpgai_research/model_ir/unified_decision_interface.py` defines a *shared facade* over the existing `DecisionState` sequential source and `FanoutDecisionState` multi-reader source compilers: one typed `DecisionAction`, one conservative E0 mask, one normalized E1 source-executed GCC status vocabulary, and a weight-private variable-sized node/edge graph feature schema. It delegates to the **original production compiler transitions, source certificates, C++ emitters and numerical/source-value checkers**; it does NOT add a new generalized compiler, trained RL agent, arbitrary DAG legality or hardware feasibility mask. The public observation omits weight payloads and source state SHA; source provenance is available separately. All result statuses and original SHA hashes are preserved; UNKNOWN/failed input cannot be called E1; GCC input/CPP/stdout/trace source digests and finite numeric receipts are verified in the applicable domain. A finite action/GCC budget prevents silently dropped candidates. The two pre-existing source API semantics remain separate.

`src/fpgai_research/experiments/cst_r2p_unified_interface_study.py` and `tests/test_cst_r2p_*.py` implement a **prospective 2-domain × 3 candidate = 6-action** original source-GCC experiment with E0/E1 parity checks, clean-tracked Git revision, complete returned action denominator, immutable original JSON observations, and separate read-only JSON/digest/status audit. Both quick/paper `experiments/compiler_ir/series.py` profiles now include `unified_policy_interface_e1` after R2n and R2o stages. **No R2p original workstation results yet**, and new tests also **not yet workstation verified**. Do not cite R2p E1 success counts until original study and audit are run. [R2p full protocol and local commands](R2P_UNIFIED_C3_SOURCE_INTERFACE_PROTOCOL_20261103.md).

No `rl_environment/`, original R2i/R2j/R2l/R2m/R2n/R2o vendor or source experiment implementations, or GitHub Actions were altered. The existing routed physical six-pair BRAM ties and mixed LUT/FF/DSP/WNS remain published as **descriptive warned OOC**, not strict clean-DRC E4.

## 2026-11-03 — R2o ORIGINAL 12-action multi-reader GCC study and read-only audit WORKSTATION VERIFIED

Source-pinned checkout `ce8083c0ad5b827232e1e67322811b25edd3ac79`, clean tracked state. User ran **27/27 focused** tests and **1,484/1,484 full pytest** (138.32 s), then the original **create-only** `build/r2o_c3_multi_reader_original_gcc_v1/` experiment on **three** prespecified static MatMul fanout graphs with **2, 3 and 4 independently mapped readers** and four actions per graph. Results: **12/12 original outcomes recorded; nine E1_GCC_MULTI_READER_QUALIFIED; three REJECTED invalid producer-PE actions; zero UNKNOWN and zero failed E1 executions; nine original GCC executions**. This is **nine qualified proposals, not 12 successes**. The two-reader, three-reader and four-reader graph arms respectively recorded **102, 210 and 372** source-address witnesses per qualified GCC execution. All three original roles per graph that were E1 attempted passed their selected single fixed host-float input numerical, real source-address, and mapping/lifetime checks.

Independent *read-only* source/manifest/stdout SHA receipt auditor `build/r2o_c3_multi_reader_original_gcc_audit_v1.json` reported identical original source revision, action statuses, case and witness populations. This receipt integrity audit **does not independently replay GCC or numerical semantics**. No original Vitis/Vivado or RL training calls occurred. R2o source-only E1 is not physical memory, achieved Fmax, power, HLS bank-port feasibility, clean-DRC E4, arbitrary DAG legality, all-input mathematical equivalence, or learned RL gain.

**FREEZE** these original R2o outputs at source `ce8083c...`; do not overwrite/re-run to revise outcomes. Earlier R2n original nine-action GCC campaign and R2i/R2j/R2l/R2m results remain independently frozen. The next compiler sprint may create a policy-independent shared facade over bounded row-chain and multi-reader adapters, but must not silently call this a generalized arbitrary-DAG source compiler. [Original R2o protocol](R2O_C3_MULTI_READER_DECISION_SOURCE_PROTOCOL_20261103.md).

## 2026-11-03 — R2o C3 multi-reader Phi action interface, original GCC study and read-only audit IMPLEMENTED; workstation evidence PENDING

**R2n original verified and frozen:** source revision `6296f99eb5d52371bffb9755989a340aadbe7ff6`; 37 focused and 1,464 full tests passed on the user's laptop; original R2n three MatMul chain depths × three actions = 9 outcomes, 3 E1-qualified, 3 rejected, 3 E0-unknown, 0 E1 failed. Independent original R2n source receipt audit agreed; no vendor calls and no trained RL. Do not rewrite R2n results using new code.

**New R2o prospective source and tests:** `src/fpgai_research/model_ir/fanout_decision_interface.py` defines a **new bounded sibling adapter** using the existing typed `FunctionalModel`, `ArchitectureRequest`, target limits and common `DecisionAction`, directly reusing **R2f multi-reader Phi/last-use E0** and **R2h real generated-source aliasing + GCC NumPy/address E1**. It does **not yet unify** sequential-chain and multi-reader source decision states. The only editable actions in the pilot are independently selected reader PE width and disjoint/per-element-optimal scratch layout, for exact 2–4 reader MatMul fanout + left Add chain + Relu. Unsupported actions reject and retain the original complete state. Source SHA/manifest/original-GCC-stdout conflicts and attempted overwrites are fatal. The action mask distinguishes source `REJECTED`, E0-admitted but GCC unrun `UNKNOWN_E1_NOT_RUN`, `FAILED_E1_SOURCE_EXECUTION` and actual source `E1_GCC_MULTI_READER_QUALIFIED`. It discloses that source E1 is **not** HLS/Vivado clock/port/BRAM qualification or learned RL evidence.

`cst_r2o_multi_reader_action_study.py` prepares **12 prespecified actions across 3 static fanout graphs** with original generated C++, emitter manifests, full GCC stdout and failure receipts in a **new create-only source-pinned study**; `cst_r2o_multi_reader_action_audit.py` independently checks raw artifact SHA/count/status integrity **without rerunning GCC**. Both quick and paper compiler-IR experiment profiles include the new stage; focused tests and original source pilot are newly committed but **NOT yet executed on the user's workstation**. No new results, pass rates, or original E1 campaign counts may be claimed before the local evidence is received.

[Read R2o complete source experiment protocol and commands](R2O_C3_MULTI_READER_DECISION_SOURCE_PROTOCOL_20261103.md). The original R2i/R2j/R2l/R2m physical resource/WNS evidence, original R2n source results and frozen `rl_environment/` were unchanged. No GitHub Actions/tokens were used. Full arbitrary DAG mapping legality, partial action design, bank port timing and true RL learning remain open.

## 2026-11-03 — R2n C3 ORIGINAL GCC campaign and independent audit WORKSTATION VERIFIED

At original tracked revision `6296f99eb5d52371bffb9755989a340aadbe7ff6`, user executed source validation: **37/37 focused pytest**, **1,464/1,464 full pytest** in 137.17 seconds, then new original create-only GCC `build/r2n_c3_e1_qualified_actions_v1/` for three static MatMul chain depths **2, 3, 4** with **9/9 complete candidate actions**. Each depth had **1/3 `E1_GCC_SOURCE_QUALIFIED`**, **1/3 source compiler `REJECTED`**, **1/3 `UNKNOWN_E0_CERTIFICATE`**. Across all three: **3 E1 qualified** with real GCC source lifetime/numeric checks, **3 rejected**, **3 E0 unknown**, **0 failed source E1** and **3 actual source E1 executions attempted**. User then ran `cst_r2n_e1_action_audit` as a separate read-only study (`build/r2n_c3_e1_audit_v1.json`), independently reconfirming counts, input/source SHA and evidence-tier status consistency. Audit **did not execute GCC**. Original original Vitis, Vivado and RL training calls **0**. This validates **bounded static single-reader row-chain source** decision evidence only; no general multi-reader certified action interface or learned RL gains.

Experimental source revision and original artifacts are **frozen**. Do not rerun, overwrite, or delete either R2n source or audit. Subsequent new code is a new source revision and is NOT retroactive proof for the original R2n. See [R2n original protocol](R2N_C3_E1_QUALIFIED_DECISION_IR_20261103.md) and [primary manuscript](MANUSCRIPT_DRAFT.md).

## 2026-11-03 — R2n C3 GCC-qualified source action gate (committed; original workstation unverified)

Implementation: `model_ir/decision_interface.py::source_executed_action_mask` screens a finite typed candidate tuple through one real compiler transition per candidate and only promotes bounded E0 mapped-value `row_chain` actions if actual GCC host-float numerical and source trace checks pass on explicit fixed input. It retains `REJECTED`, `UNKNOWN_E0_CERTIFICATE`, `FAILED_E1_SOURCE_EXECUTION` and `E1_GCC_SOURCE_QUALIFIED` distinctly; forged source hashes fail fatally and failed execution is never counted as success. It makes **zero Vitis/Vivado calls, zero trained RL calls, no hardware-feasible masks**.

The new original GCC R2n study (three graphs with depths 2,3,4 × three action roles = **nine prespecified actions**) is **implemented but NOT RUN**. It now requires a **clean tracked Git source revision before original GCC execution** and stores source SHA, input hash, actual source trace and numerical receipts. `cst_r2n_e1_action_audit.py` separately checks immutable JSON statuses/counts/SHA after the study with no new GCC; it cannot independently rerun numerical semantics. Original study and audit have source-only tests and are included in existing experiment-series source tests; the original GCC study is a new quick/paper stage. No user-workstation result should be claimed yet; run [R2n protocol](R2N_C3_E1_QUALIFIED_DECISION_IR_20261103.md).

This is an incremental **C3 secondary contribution** with 100% explicit boundary: existing typed complete-design single-reader MatMul chains, not graph-wide multiple mapped readers or arbitrary fan-out. It does not demonstrate a new RL algorithm, GNN sample efficiency, transfer, physical resource optimization, all-input numeric soundness, or DRC-clean E4. The main C1/C2 scientific contribution and existing original R2i/R2j/R2l physical results are unmodified.

## 2026-11-03 — R2m FINAL original physical report WORKSTATION VERIFIED

Original checkout `dc32d277b5e339a5f403f68e6fe1faf5cbc98a9e` passed **11/11 new focused** and **1447/1447 full pytest** (137.81 s). Original R2m source-pinned analysis completed with no new GCC/Vitis/Vivado calls; original complete output was created at `build/r2m_original_six_physical_paper_v1/`: `results.json`, `all_36_pairs.csv`, `paper_table.md`. Exactly 72 original cases/36 matched source pairs, 24/36 E3-qualified and 12/36 E3 unknown. The original six predeclared HLS BRAM-positive pairs 20,23,28,31,32,35 have **6/6 routed observed, 6/6 routed BRAM ties at 1.5 → 1.5 BRAM tiles, 0/6 clean-DRC E4**; 30 other physical pairs remain unknown. Original raw report digests and individual warning rule counts are retained in the create-only R2m `results.json`.

**Measured routed LUT, FF and WNS are now resolved from saved original Vivado reports**, replacing earlier tentative/no-values commentary:

| Pair | Original cases | HLS BRAM_18K estimated savings | Routed BRAM tiles | LUT disjoint → optimized | FF disjoint → optimized | WNS disjoint → optimized ns |
|---|---|---:|---|---|---|---|
| 20 | 040/041 | 1 | 1.5 → 1.5 | 649 → 684 | 396 → 407 | +26.356 → +27.304 |
| 23 | 046/047 | 1 | 1.5 → 1.5 | 656 → 693 | 396 → 407 | +26.941 → +27.008 |
| 28 | 056/057 | 1 | 1.5 → 1.5 | 870 → 889 | 501 → 503 | +25.491 → +27.133 |
| 31 | 062/063 | 1 | 1.5 → 1.5 | 878 → 819 | 497 → 494 | +26.359 → +27.054 |
| 32 | 064/065 | 1 | 1.5 → 1.5 | 883 → 880 | 504 → 489 | +27.205 → +26.863 |
| 35 | 070/071 | 1 | 1.5 → 1.5 | 869 → 858 | 504 → 499 | +27.119 → +26.121 |

Optimized LUT increases **3/6**, decreases **3/6**; FF increases **3/6**, decreases **3/6**. WNS increases descriptively **4/6** (mean **+0.335333 ns**) and decreases **2/6**. Zero meaningful physical BRAM allocation improvements observed, but **not** a general “FPGA optimization fails” statement; the evidence is selected OOC designs with **warnings** and cannot support strict E4, achieved Fmax, board I/O, power, or peer-compiler superiority. Prescribed original follow-up vendor calls total ten, in addition to the earlier original pair20 two; R2m report itself called no vendors.

**Source-only R2m post-verification improvements are newer code**, not part of that workstation-tested checkout: a corrected Markdown bold closing marker, explicit routed DSP in the summary table, and optional descriptive LUT/FF direction summaries with regression test assertions. Run tests again before citing an R2m v2 export. Preserve v1 output and every original R2i/R2j/R2l artifact.

Scientific interpretation: certificate-backed source allocation is proven within its bounded source contracts, conditional HLS resource predictions did not translate into physical BRAM tiles in this six-pair experiment, and external feature-equivalent comparison to published compiler artifacts remains unfinished. See [current manuscript results](MANUSCRIPT_DRAFT.md), [R2m protocol](R2M_FINAL_ORIGINAL_PHYSICAL_PAPER_EXPORT_20261103.md), [pinned comparison obligations](PRIOR_ART_PINNED_ARTIFACTS.md).

## 2026-11-03 — R2l original physical six-positive-pair campaign COMPLETE (descriptive OOC, NOT strict E4)

**User-workstation source-pinned original evidence; frozen inputs.** After R2l checkout \`3071d6fe70a0eb3202cea8f02bcf53f6a3b7e1f4\` passed **10/10 focused** and **1436/1436 full** tests, original R2i SHA \`b2d5dcc4f18fad7cb1cd2c342101dd33185e817f\` remained unchanged. Original pair20 was already routed in \`build/r2j_fanout_physical_36_pairs_v1\`; five matched follow-up pairs (23, 28, 31, 32, 35, cases **046/047, 056/057, 062/063, 064/065, 070/071**) were newly executed using unchanged canonical R2j code in create-only \`build/r2l_five_positive_physical_followup_v1\`. All **ten** new original Vivado OOC vendor executions exited 0, completed routes and met requested-clock timing, but **all have DRC warnings**; no strict clean-DRC E4. Original pair20 also has warnings.

The independent read-only R2l final ledger (user saved as \`build/r2l_final_six_pairs_after_followup_v1.json\`) reconstructed the entire **72-case/36-pair** HLS source cohort (72/72 CSim E2, 48/72 CSynth E3, 24/36 matched E3, 12/36 E3 UNKNOWN). The six prespecified E3-qualified HLS BRAM-positive pairs **20,23,28,31,32,35** now have **6/6 matching routed observations**, **0/6 routed BRAM decreases, 6/6 routed BRAM ties, 0/6 routed BRAM regressions**, and **0/6 clean-DRC E4**. The remaining 30/36 physical matched pairs were **not** tested with Vivado and remain unknown; they are not ties. Source scratch savings and six HLS-estimated BRAM reductions **do not** imply fewer physically routed BRAM tiles.

Matched route WNS, requested 32 ns clock (optimized minus disjoint, **descriptive OOC slack only**):

| Original index | Cases | Disjoint WNS ns | Optimized WNS ns | Δ WNS ns | Routed BRAM Δ |
|---|---|---:|---:|---:|---|
| 20 | 040/041 | +26.356 | +27.304 | +0.948 | tie |
| 23 | 046/047 | +26.941 | +27.008 | +0.067 | tie |
| 28 | 056/057 | +25.491 | +27.133 | +1.642 | tie |
| 31 | 062/063 | +26.359 | +27.054 | +0.695 | tie |
| 32 | 064/065 | +27.205 | +26.863 | −0.342 | tie |
| 35 | 070/071 | +27.119 | +26.121 | −0.998 | tie |

**4/6** positive descriptive WNS differences and **2/6** negative; mean optimized-minus-disjoint WNS **+0.335333 ns**. These are not achieved Fmax, physical area improvements, clean-DRC E4 gains, board bitstreams, proof of generality or external novelty.

Interpret this as a **prespecified negative translation result** for the specific six HLS-BRAM-positive designs on the chosen part/tool constraints. It is not a statistical claim about all FPGA workloads. The user paste shows detailed WNS but omits absolute routed LUT/FF/DSP/BRAM and individual DRC rule populations for the new five pairs; those should be read separately from the original source-attested Vivado reports, never invented. Original pair20 LUT/FF regression remains as previously attested. Ten foreground Vivado executions **did occur** for follow-up; the read-only auditor's \`new_original_GCC_Vitis_Vivado_calls:0\` refers **only to the audit step**.

**Preservation:** Do not rerun or overwrite source-attested R2i HLS, original R2j pair20, R2l cases046/047/056/057/062/063/064/065/070/071, or final R2l audit. Subsequent analyses must be create-only, hash-aware and vendor-call-free. Publication gate remains **NOT READY** for strict E4 physical gain and external comparator parity; adjust paper claims to a narrower verifiable IR/lifetime/certificate contribution. 

## 2026-11-03 — R2l original follow-up pair23 workstation executed, routed BRAM TIE

User-workstation evidence from unchanged original R2i source `b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`, and R2l physical follow-up `build/r2l_five_positive_physical_followup_v1`:

- New R2l code and tests at checkout `3071d6fe70a0eb3202cea8f02bcf53f6a3b7e1f4`: **10/10 focused and 1436/1436 full pytest passed**.
- The new physical follow-up was prepared create-only, 72 cases/36 source pairs retained, with predeclared six HLS BRAM-estimate-positive pairs **20,23,28,31,32,35**; only five not previously observed.
- Pair **23**: `case_046` (disjoint) and `case_047` (per-element optimal) both reported `PREPARED_NOT_RUN` before native execution, then each completed an original **Vivado OOC route** (exit 0, ~69s), nonnegative WNS **+26.941 ns** and **+27.008 ns**, respectively. Optimized-minus-disjoint WNS is **+0.067 ns descriptive only**. Both reported DRC warning-only routed observations; **neither clean-DRC E4 qualified**.
- The independent read-only source-pinned R2l completion ledger reports **2/6 HLS-positive routed matched pair observations** (original pair20, new pair23), **2 routed BRAM ties**, **0 routed BRAM reductions**, **0 routed BRAM regressions**, **0/6 strict E4 qualified**, and **4 HLS-positive pairs physically UNKNOWN** (28,31,32,35). This is a second example where a source/HLS-predicted BRAM advantage did not survive routing.
- The R2l ledger's `new_original_GCC_Vitis_Vivado_calls:0` refers only to the **read-only ledger invocation**. The preceding original R2l physical runner performed **two real original Vivado invocations** for cases 046 and 047. Do not conflate these call counts.
- The supplied terminal paste does not expose pair23 LUT/FF/DSP/BRAM absolute values or raw DRC rule IDs. Only the independently audited **paired BRAM tie** is established numerically from the R2l completion summary. Detailed resources and warning identities must be read from source-attributed saved reports; do not invent them.
- Original pair20 `build/r2j_fanout_physical_36_pairs_v1`, original R2i study and the subsequent pair23 cases are **frozen**; never rerun or overwrite. The user should finish the remaining four original pairs with the same pinned experiment code and canonical R2j execution path, and separately publish a new create-only final audit.

**Publication remains NOT READY** for clean-DRC E4, physical area win, generally achieved Fmax, or feature-equivalent external novelty. Source and HLS E2/E3 results remain valid within their own tiers.

## 2026-11-03 — R2k workstation VERIFIED; R2l separate five-pair physical follow-up implemented

**User workstation verification:** After fast-forwarding to \`41131aae9d7426c383a07c88a4f28eb97c7cf5f1\`, the new R2k forensic suite passed **24/24 focused tests** and the complete source regression passed **1426/1426** in 137.47 seconds. Original raw source-attributed Vivado \`drc.rpt\` receipts for original R2j pair20 (\`case_040\`/\`case_041\`) were both hash-verified with **zero new vendor calls**. Both designs have the same **nine DSP rule/cell witnesses** (\`DPIP-2×3\`, \`DPOP-3×3\`, \`DPOP-4×3\`), the same \`RTSTAT-10×1\` and the same reported **26 no-routable-load nets**. Whether the 26 nets have exactly identical *names* is still UNKNOWN. Neither the identical warning instances nor advice to pipeline DSPs proves the cause of the routed BRAM tie/LUT+FF regression. **Strict E4 remains 0**.

**New R2l prospectively implemented on later main:** \`cst_r2l_physical_completion.py\` provides a read-only 72-case/36-pair source-identity ledger, predeclares original HLS BRAM-positive pairs **20,23,28,31,32,35**, preserves original pair20, and checks source/clock/DRC evidence for a *separate* physical follow-up study covering the five previously unobserved pairs. No new Vitis/Vivado calls have been performed. Nine standalone isolated new R2l ledger unit tests passed in the development environment, but **the new R2l code and the full repository suite require user-workstation verification** before executing new original Vivado cases. No positive routed memory advantage, DRC-clean E4 or publication readiness is asserted.

Runbook: [R2l five-pair physical follow-up](R2L_FIVE_POSITIVE_PHYSICAL_FOLLOWUP_20261103.md). Original R2i HLS, R2j pilot, read-only exports and R2k diagnostic artifacts remain immutable.

## 2026-11-03 — Verified R2j post-route diagnostic overturns pair20 HLS BRAM prediction

**New original user-provided read-only evidence:** At checkout
`142ca73d569ec6944ee30db3382eac333e887905`
the new physical diagnostic's **16/16** focused tests and
full **1,418/1,418** regression passed.
Source-attributed original R2j pair 20 (`case_040`
disjoint/`case_041` per-element optimal) was independently
read-only audited; **no new GCC/Vitis/Vivado calls**.

**Correct graph/Phi:** `fanout3_3x3x4x4`, `scalar`
(previous preliminary written label `last_reader_pes2` was
wrong and has been explicitly corrected).

Original Vitis HLS-estimated BRAM_18K **3 vs 2**
did **not** persist to routed utilization:
actual Vivado routed **BRAM tiles 1.5 vs 1.5** (TIE),
LUT **649 vs 684** (optimized +35, a 5.39% increase),
FF **396 vs 407** (optimized +11, 2.78% increase),
DSP **7 vs 7**, URAM **0 vs 0**.
Routed WNS **+26.356 ns vs +27.304 ns**, optimized
descriptive slack **+0.948 ns**. Both routed and met
the requested clock, but both disclosed the SAME
**10 noncritical DRC warning-class violations**:
`DPIP-2`×3, `DPOP-3`×3, `DPOP-4`×3,
`RTSTAT-10`×1; `unconstrained_io=false`,
`critical_warnings=0`. Actual DSP instance and
net identities have not yet been extracted from the
saved original `drc.rpt`.

**Evidence classification:** 2/72 original OOC routed
warning-only observations, 1/36 observed matched pair,
**0/72 strict clean-DRC E4 cases**, **0/36 strict E4
pairs**. The remaining 35 physical matched pairs are
unobserved. No physical BRAM improvement is established.
The source-level and native HLS E2/E3 results remain
valid as their own evidence tiers; 6/36 HLS-estimated
BRAM reductions must NOT be interpreted as 6 physical
reductions. This one pair is a counterexample to
guaranteed physical advantage from unit-slot reduction.

**Original data is frozen**: `build/r2j_fanout_physical_36_pairs_v1`,
`build/r2j_fanout_physical_first_audit_v1` and the
read-only diagnostic output `build/r2j_physical_pair20_readonly_diagnostic_v1.json`.
See [R2j pair 20 numerical physical evidence and DRC rules](R2J_PAIR20_POSTROUTE_NUMERIC_DIAGNOSTIC_20261103.md).
Do not waive warnings, reinterpret HLS BRAM, or
retroactively claim E4. Publication firstness,
external comparator parity and strict E4 remain
unqualified.

## 2026-11-03 — Original R2j physical pair20 observed, NOT clean-DRC E4

Source-pinned original R2i study `b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`
is preserved. R2j source/test checkout at `adaa373145ed2e3ecf508b78091db4620338e777`
passed **8 focused and 1402 full tests**. Complete native R2i
read-only effect exporter: **72/72 E2 CSim**, **48/72 E3 CSynth**,
**24/36 both-E3 native pairs**, **6 HLS-estimated BRAM savings**,
**18 HLS ties**, **12 E3-unknown pairs**, no HLS regression on
qualified cases. All **24/24 qualified pairs** reduced
HLS-estimated worst-case cycles, mean 24.9583 cycles
and median 23 cycles. These are NOT physical measurements.

R2j source-pinned physical staging prepared **48/72**
original E3-qualified Verilog/ROM packages and kept
**24/72 E2-only cases blocked** in the full 72-case denominator.
Original routed pilot was preselected pair20: `case_040`
disjoint and `case_041` per-element optimal.
Both original Vivado OOC invocations exited 0,
completed route and met timing; routed
**WNS +26.356 ns versus +27.304 ns**,
descriptive optimized-minus-control WNS **+0.948 ns**.
Both have independently classified disclosed DRC warnings:
**2/72 warning-only routed observations, 1/36 matched
routed-observation pairs, ZERO strict E4-qualified cases
or pairs**. Do not misrepresent as clean DRC.
Original physical counts and exact warning rule IDs were
**not included in the supplied terminal paste**, so
actual routed BRAM improvement is UNKNOWN.
Original physical tree and first export are immutable;
do not rerun these two Vivado cases.

New subsequent read-only diagnostic tool
`experiments/cst_r2j_physical_diagnostics.py`
and tests read the existing original physical audit to
show DRC IDs, routed BRAM/LUT/FF/DSP, descriptive
effects and source provenance, with no native calls.
Those new tests and tooling have **NOT** been verified
on the user's workstation yet. See
[R2j original warning-only pilot record](R2J_ORIGINAL_WARNED_PHYSICAL_PILOT_20261103.md).
Continue no new original physical calls until actual
DRC warning categories/resources have been inspected.
Scientific firstness and submission remain NOT READY.

## November 3, 2026 — FULL ORIGINAL R2i NATIVE COHORT FINISHED; R2j physical prospective

**Original research source immutable at**
`b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`.
User workstation completed **72/72 original Vitis HLS executions**
with **72/72 independently qualified numerical CSim (E2)**.
Only **48/72 original CSynth reports (24/36 matched pairs)**
met the strict source/latency/resources/uncertainty E3 gate.
The other **24 cases / 12 pairs** passed CSim but original
Vitis `latency_best = undef`; they remain **UNKNOWN E3**
without imputed cycles or resources.

Across all **36** prespecified pairs: **6** report one lower
**HLS-estimated BRAM_18K** for the optimized arm,
**18** HLS-estimated BRAM ties, **0** qualified BRAM
regressions, **12 UNKNOWN**. All 24 E3-qualified
pairs showed lower **HLS-estimated worst-case cycles**,
but this conditional finding excludes 12 E3-unknown pairs
in the original denominator. There is **NO** new placed/routed
clean-DRC physical E4 data and **NO** FPGA Fmax/bitstream
or independently established external novelty.

The original complete 72-case study
`build/r2i_fanout_native_36_pairs_v1` is frozen.
The exact original native audit was saved as
`build/r2i_fanout_native_final_audit.json`.
See [R2i complete original native results](R2I_FULL_ORIGINAL_NATIVE_HLS_RESULTS_20261103.md).

New subsequent R2j implementation, **NOT YET
workstation-tested and with no Vivado executions**:
read-only 36-pair native effects exporter plus separate
72-case/36-pair OOC Vivado source/RTL/ROM packaging,
strict routing/DRC audits, opt-in physical case runner.
Original 24 E2-only cases remain visible but blocked
from physical qualification. First preselected HLS BRAM
reduction pilot: pair20, case040 vs041, 3 vs2
HLS-estimated BRAM_18K, not yet physically verified.
See [R2j strict physical protocol](R2J_STRICT_FANOUT_PHYSICAL_PROTOCOL.md).

**Publication remains NOT READY** pending routed
hardware comparisons and valid external peer parity.

## Original R2i native first-pair result — 2026-11-03 (E3 only)

Original local study pinned at
`b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`
passed **6/6** new focused tests and **1,394/1,394** full
Python regression, preparing **72** source-matched Vitis packages.
First **2/72** original native Vitis invocations succeeded,
and **both** passed independently audited native
CSim/CSynth E2/E3 qualification. **1/36** source-memory
matched pairs qualified. For that first 2-reader scalar-Phi
graph, disjoint vs per-element optimized:
scratch **30 vs 17** scalar source slots;
HLS BRAM_18K **0 vs 0** (tie), LUT **2566 vs 2564**,
FF **478 vs 476**, DSP **4 vs 4**,
worst HLS latency **355 vs 342 cycles** and
estimated clock **7.683 vs 7.683 ns**.
**35/36** native pairs remain unrun/UNKNOWN; no routed
physical E4 or external compiler parity achieved.
All original native evidence is immutable. Original
workstation checkout must **remain pinned** until
the 36-pair Vitis study completes; do **NOT git pull**
mid-study, since source-preflight checks the original
manifest revision. See
[R2I_ORIGINAL_NATIVE_PILOT_RESULTS_20261103.md](R2I_ORIGINAL_NATIVE_PILOT_RESULTS_20261103.md).

## 2026-11-03 — R2h matched source memory original cohort VERIFIED

**User workstation original at SHA `efc6497cdab2ec5734a0c3750972655b9ea397b9`:**
13/13 focused tests and **1388/1388** full tests passed.
The original 36 matched source pairs completed **72/72 real GCC
compilations and executions** using the same runtime 2–4-reader fanout
algorithm with disjoint versus independently certified per-value
scratch addresses. **36/36** pairs passed real address-witness
checks, reduction and Phi validation and all-intermediate/final
NumPy comparisons; all **36/36** disjoint-to-optimized comparisons
reduced actual GCC **scalar scratch-array element counts**.
Across 27 graph/Phi interventions, **18/27** changed the
certified source minimum with fewer source slots, **9/27 tied**,
**0/27 regressed**. The original read-only audit independently
rechecked **72/72 saved source/trace records**, all **36/36**
matched pairs and the 27 mapping contrasts with **zero new
GCC/Vitis/Vivado** calls.

Original folders:
`build/cst_r2h_original_36_matched_72_gcc_v1`,
`build/cst_r2h_independent_source_address_audit_v1`.
Both are **frozen** and must never be reexecuted or overwritten.

**Source C++ scalar scratch slots are NOT FPGA BRAM, RAM ports,
hardware timing, bit-exact fixed-point synthesis or physical memory.**
Accumulation/counter arrays were counted separately and all
scalar locals excluded. The result is source-level E1
memory materialization, not vendor E3/E4 proof.
The mature PLDI 2001 / ASPLOS 1998 / POPL 2016
schedule-storage prior art prevents broad firstness claims.
The next gate is source-pinned, matched Vitis HLS CSim/CSynth
and independent read-only native report audit, followed by
clean-DRC Vivado and meaningful peer artifact parity.

## 2026-11-03 00:04:39 — original complete v6 exploratory physical study

The original workstation terminal at git revision
`6de412a94bc7cd1423e58bb858875d56916b6e2d`
reports **8/8 focused** cross-cohort tests and
**1,236/1,236 full suite** passing. The new
read-only reviewer audited separate source cohorts:
40/40 original CPU/GCC series stages completed
at `1cefdd8566658cb6b69e12444c9b5ce9ba6e4a31`;
the original fixed-point HLS / Vivado source revision
is `8814bd0005e8a852348b49f19131b2410117216b`.
Cross-revision numerical pooling is forbidden.

C1/C2/C3 finite source protocol readiness reports
were true for the exact supported subsets, but
research novelty/physical superiority/RL advantage
were false. Original 52 decision proposals:
24 source-admissible, 28 rejected; 0 restricted
oracle mismatches; 3 GCC numerical qualifications.
R3/R4: 3 static graph depths, 9 certified
proposals, 3 certified legal and 3 source-admissible
but certificate-UNKNOWN; zero trained RL policies.
Phi causal secondary study: 24 original GCC cases,
18 effective Phi interventions, 18 changed C++
and source-event hashes, **0/18 actual last-use
and 0/18 source scratch capacity effects**.
Original CSV SHA
`d22219c4e6afbde1ab784a7e9de29cf876b2873f60ed8125dea009f2f7780898`.

A further **ten original Vivado 2023.2** v6 cases,
`case_002`…`case_011`, ran as separate
foreground invocations with original exit code 0:
duration seconds (167,170,95,94,92,88,97,99,110,108).
Final independent controller: **12/12 case receipts,
12/12 full-routing/timing/allowed-warning observations,
0/12 strict DRC-clean E4, 0 prepared, 6 prespecified
matched source-storage pairs, none blocked**. The
original case_000/001 first-pair report contained
disjoint minus shared:
`BRAM=0.0; DSP=+3; LUT=-155; FF=-10;
URAM=0; WNS=+0.195 ns`.
Warnings present: case_000 DPIP-2=16, DPOP-3=19,
DPOP-4=19, RTSTAT-10=1; case_001 DPIP-2=13,
DPOP-3=16, DPOP-4=16, RTSTAT-10=1.
All remaining five pairs' numeric values must
still be extracted from the original case files,
not fabricated by extrapolating first-pair effects.

Historical v3-v5 remains immutable. Existing v6
source/report/ROM artifacts must NEVER be
regenerated/rerun. The six-pair report should be
created in a **new** output via
`cst_nchw_physical --study build/nchw_bram_vivado_rom_v6
--export-output build/nchw_v6_final_pair_export_v1`,
then independently reviewed. The new
`experiments/compiler_ir/summarize_research.py`
also reports six-pair complete denominator and
source-only Phi-negative finding; **its latest
enhancement/tests have not yet been run on the
workstation**.

See [the prospective Phi causality protocol](CST_R2_NEGATIVE_FINDING_AND_NEXT_FALSIFICATION.md).

# FPGAI compiler-IR experimental evidence ledger

**Updated:** 2026-11-02 (Europe/Amsterdam). **Purpose:** distinguish *executed, source-attributed* CPU/GCC evidence from unexecuted Vitis experiments and hypotheses about scientific novelty. This is a record of GitHub Actions results, not native-board validation.

## Confirmed CI checkpoints

### End-to-end CPU/GCC + source-only HLS packaging

- Commit: `5e3a6202696e0eb021f69d79dd767dbadf3566d4`.
- Successful [compiler-IR CPU experiment run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41360169745).
- Targeted new compiler tests: **38 passed** (the workflow log).
- `array_ci/summary.json`: 96 configurations; 48 matched baseline/layout pairs; 48 distinct effective mapping fingerprints; 96 unique generated C++ hashes; 8/8 GCC-host numerical cases passed; 0 Vitis calls.
- `row_ci/summary.json`: 18 paired full-tensor/row-lifetime source cases; 4/4 host-checked pairs numerically passed; maximum **5× reduction in declared intermediate C++ array elements**, not measured BRAM; 0 Vitis calls.
- 12 source-pinned original HLS cases prepared (4 matched 3-policy groups); read-only source audit and scorecard generated; 0 original Vitis HLS calls and **zero** measured/qualified native synthesis results. The paired table contains 8 explicitly unmeasured comparisons.
- The workflow uploads `build/compiler_ir/` as a short-retention GitHub Actions artifact under a source-derived name. Preserve exact source SHA and raw CSV/JSON if incorporating results into the paper.

### Existing full repository regression

- At `72755747435427a34ac3d75d9bf16e921b6a57e5`, [CPU regression run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41360466702) reported **808 passed, 2 skipped**. (A newer revision includes the extra multi-operator row-chain campaigns and requires its own final CI confirmation.)
- An earlier failed workflow was caused by a **test error-message assertion mismatch**, not silent native HLS success; it was fixed in `91c76203bf4a7e8393674447b6a46a2b86af2b1a`. Keep failed snapshots in historical provenance rather than treating them as passing.

## New prospective experiments after those checkpoints

- `src/fpgai_research/model_ir/row_chain.py` and `codegen/hls/row_chain.py` add source-level row-epoch tensor lifetime composition for 3–8 MatMul operations in strict chains; source-generated C++ includes independent per-operator Phi and an internal scratch row per boundary.
- `validation/cpp_chain_trace.py` independently checks generated host C++ temporal event ordering across internal boundaries.
- `tests/test_cst_row_chain.py` and `tests/test_cst_chain_campaign.py` cover 3–4 operator graphs and experiment integrity.
- `experiments/cst_chain_campaign.py` can create 12 CPU cases with 4 host numerical and 2 executed Phi+epoch trace checks; [compiler-IR CI workflow](../../.github/workflows/compiler-ir-cpu.yml) was extended to run/archive the chain campaign. **Do not claim specific chain campaign results until its relevant workflow completes successfully.**

## Evidence semantics

| Category | Proven by current checkpoint? | Interpretation |
| --- | --- | --- |
| E0: typed intermediate contracts / bounded source-level witnesses | Yes for completed subset | The implemented compiler understands declared element/layout/lifetime relationships, within supported geometry/constraints. |
| E1: generated C++ variants, actual GCC host execution | Yes for completed cases | Same mathematical model can produce distinct source access/schedule alternatives with passed host-float numerical tests. |
| E2: original Vitis ap_fixed C simulation | **No** | Generated testbenches exist; original vendor tool has not been run for CST comparison. |
| E3: original Vitis HLS synthesis cycles/resources/timing | **No** | No measured hardware QoR, physical BRAM or timing claim. |
| E4+: actual routed FPGA and executed hardware | **No** | No physical speedup or deployed hardware proof. |
| Novelty vs Allo, HIDA, StreamTensor, Calyx, etc. | **Unresolved** | CPU engineering results are useful but do not prove new scientific priority; feature-level comparison and evidence still needed. |

## Source preservation and non-RL policy

- Latest source updates are committed to `main`, but **do not pull remote main into the currently running source-pinned RL L0 checkout** until its original revision audit finishes.
- RL source backup reference: `backup/rl-environment-frozen-2026-11-01` at `5221094af585bd501e842deca94d19c2289bb628`.
- No new RL agent/environment/native campaign is part of these studies.

## Required for paper readiness

1. Latest **full** CPU regression green, including multi-layer chain code.
2. Archived create-only CPU/GCC and actual generated-source traces, paired to exact SHA.
3. Novelty evidence stronger than typed layouts or simple row scratch, with implemented and independently checked semantics.
4. Source-attested original Vitis CSim and CSynth on a small matched triplet study; do not cite absent QoR.
5. Precise paper-vs-prior-art capability comparison, methodological limitations, provenance and release/IP review.

**No completed compiler-IR paper or publication-level scientific novelty is claimed by this ledger.**


## Verified final CPU experiment milestone — 2026-11-02

A complete successful compiler-only CI run and complete repository regression are now available at **exact same SHA** \`6282fcea6e87cf0dc84cad28f4479dec0404da63\`.

- Full tests: **819 passed, 2 skipped**, source [GitHub Actions run 41361561271](https://github.com/umutcanaltin/fpgai_research/actions/runs/41361561271).
- Compiler-IR focused regression: **52 passed**, source [run 41361561258](https://github.com/umutcanaltin/fpgai_research/actions/runs/41361561258).
- Layout source study: 96 emitted cases, **24 distinct functional graphs**, **24 with two architecture mappings**, 48 matched layout pairs, eight actual GCC host numerical qualifications passed.
- Row-epoch source study: 18 two-operator pairs, four GCC numerical pairs qualified, maximum **5× reduction in intermediate C++ source declaration elements** (not physical BRAM).
- Chain source study: 12 cases (**8 three-operator, 4 four-operator**), 12 independently bounded liveness certificates, four GCC host numerical qualifications passed, two actual C++ Phi/row-event trace checks passed, maximum **3× source-declaration ratio**.
- Total selected host numerical checks: **16/16 accepted**, not a general population success confidence interval.
- Original Vitis HLS source preparation: 12 matched source/Tcl/testbench cases, eight prospective native pairwise comparison rows **all unmeasured**; no CSim, CSynth, physical Fmax or RTL performance evidence.
- CPU source CSV and JSON, manuscript-ready E0/E1 table, 12 prepared cases and read-only native scorecard are uploaded by the compiler CI workflow artifact for run 41361561258.

**Related-work update:** include PACT'26 ST-Flow (DOI 10.1145/3838684.3846865). It materially narrows potential novelty; a direct semantic/experimental comparison with its published artifact is unresolved. The manuscript draft does not claim scientific firstness. See [MANUSCRIPT_DRAFT.md](MANUSCRIPT_DRAFT.md) and [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md).

## Controlled schedule-intervention research milestone (2 November 2026)

**Immutable code revision:** `adb28108c7f13e52ae6c3b7ff15b87675762bf6d`. Both GitHub Actions jobs finished successfully:

- [Full repository regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41364453968): **844 passed, 2 skipped**.
- [Scientific compiler experiment workflow](https://github.com/umutcanaltin/fpgai_research/actions/runs/41364454009): **79 focused tests passed**, all source-only/GCC experiment commands completed. Archived artifact ID `13106353444`, file name `compiler-ir-cpu-adb28108c7f13e52ae6c3b7ff15b87675762bf6d`.

### Prespecified explicit-schedule experiment — actual outcomes

The create-only `schedule_ablation_ci/summary.json` reports **12 graphs and 168 candidate schedule trials**, comprising **30/30 accepted legal controls** and **138/138 correctly rejected deliberately illegal interventions**. Exact mutation denominators: early consumer 30/30, duplicate stage 30/30, missing terminal stage 30/30, invalid operator 30/30, insufficient scratch slots 18/18. Every invalid trial recorded the intended first counterexample class. Six selected generated-host numerical tests passed (6/6) and three executed row-event traces passed (3/3). The descriptive median finite checker wall time was **10,314 ns on this CI runner**; it is neither a portable throughput claim nor a claim of hardware performance.

**Interpretation:** The operator-only Phi checker passes the *unchanged operator IR* and has no proposed cross-operator schedule input. The new schedule checker rejects these five targeted intervention classes; this is a **scoped representation/legality ablation**. The experiment does not estimate recall over naturally occurring compiler bugs, general-size graphs, or external prior-art tools.

### Other completed E0/E1 studies at same revision

- Matrix layout: **96** emitted configurations from **24** functional graphs; **48** graph/mapping signatures, **96** distinct generated C++ hashes, **8/8** host checks accepted.
- Two-operator row epoch: **18** matched pairs, **4/4** selected numerical host checks accepted; maximum 5× ratio in *declared source scratch elements*, not hardware BRAM.
- Three-/four-operator row chain: **12** emitted cases (8 three-operator, 4 four-operator), **12/12** bounded liveness checks, **4/4** selected host numerical checks, **2/2** separately executed Phi+row-epoch traces.
- The three previous campaigns have **16/16 selected numerical host checks**. The new schedule intervention campaign adds **6/6** separate host checks. Do not conflate the denominators or count repeated models as independent network generalization.

### Native feasibility — not yet measured

- Family A: original 12 matched whole/column/row HLS packages prepared, 8 prospective pairwise HLS comparisons **all unmeasured**.
- Family B: **6** matched row-block schedule HLS packages prepared (B=1,2,M across two separate chain models), four prospective pairwise HLS comparisons **all unmeasured**.
- **Zero actual original Vitis CSim/CSynth calls** in this GitHub-hosted CPU workflow. Neither physical RAM, achieved clock, FPGA performance nor native fixed-point equivalence has been established for the new CST experiments.

Raw case-level CSV, first counterexamples, summaries, source-hashed HLS packages, and empty-HLS scorecards are retained in the immutable GitHub Actions artifact linked above (standard finite artifact retention applies). The controlled study protocol is [SCIENTIFIC_STUDY_PROTOCOL.md](SCIENTIFIC_STUDY_PROTOCOL.md).

**Research gate:** H1–H3 have positive evidence in the explicitly bounded synthetic test domains; H4 and H5 remain unresolved. Do **not** call this a first-of-kind IR or publish speedup/BRAM claims without original native data and direct comparison with ST-Flow/Allo/StreamTensor.

## Independent exhaustive oracle and scaling study — actual successful run

**New E0 source verification:** GitHub Actions [research CI run 41367136765](https://github.com/umutcanaltin/fpgai_research/actions/runs/41367136765), pinned to commit fc3ab997cf4b80aa81ff820f6bcd93ae09a5a727: **95 focused tests passed**.

- Five finite complete six-event permutation universes; **3,600 candidate schedules** total. Independent constructive oracle: **150 legal**, **3,450 illegal**; the production checker agreed on all 3,600, **zero false accepts and zero false rejects**.
- Domain-specific oracle legal sequences: 4 (2 rows, 3 stages, one slot), 20 (2 rows, 3 stages, two slots), 6/30/90 (3 rows, 2 stages, 1/2/3 slots).
- E0 scaling: **15 static MatMul graphs**, 2/3/4 operators, row sizes 2/8/32/128/512; **126 raw schedule-checker timing samples** retained, all candidate legal controls accepted. Largest finite checked event count 12,288. Median CPU schedule-checker wall time **699,432.5 ns on that CI runner**, not portable performance or native HLS latency.
- Raw source-pinned cases and source-only HLS preparation remain in the GitHub Actions artifact for that exact run. The independent oracle shares the documented atomic-stage specification, so this is **bounded differential verification**, not a proof of general generated C++ or hardware equivalence.
- **S3 trained-model coverage preflight** was added *after* this pinned run. It reuses checksum-validated MNIST ONNX assets, reports full operator inventory and rejects unsupported CST lowering without dropping Relu/Flatten/Conv. **No actual trained model was executed by this preflight in this CI run**, and no native HLS evidence exists.


## Latest trained-network admission and oracle integration verification

Source revision **a1346df12d7fc449d5e149bb7470b744568fae1c**: [research CI 41367630447](https://github.com/umutcanaltin/fpgai_research/actions/runs/41367630447) finished successfully, **98 focused tests passed**. This re-executed the 3,600-permutation independent oracle study, the 15-graph scaling experiment, earlier E0/E1 campaigns, and the trained-network *admission tests*.

The new trained-model preflight performs only E0 provenance and graph coverage. A source-verified MLP/ONNX directory must still be supplied externally for real-asset audit; no trained MNIST model has been experimentally compiled by the restricted row-chain lowering in this study. The MatMul–Relu–MatMul negative control verifies rejection without deleting a nonlinear operation. This is not an E1 inference or original HLS result.

All detailed experimental results belong to their exact source SHA; latest documentation-only commits after a1346df do not constitute newly executed experiments.


## Verified nonlinear CST research milestone — 2026-11-02

**Source-pinned commit:** `b563ba9c87d33b87036c7c0f2be9d2d1e9de50ca`. The following CI jobs all completed successfully for the compiler implementation:

- [Full repository regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41369433057): **893 passed, 5 skipped**.
- [Compiler-IR research campaign](https://github.com/umutcanaltin/fpgai_research/actions/runs/41369433106): **128 focused tests passed**, create-only E0/E1 source studies and negative controls completed.
- [Independent ONNX Runtime/GCC bridge](https://github.com/umutcanaltin/fpgai_research/actions/runs/41369394773): **3/3 cross-framework tests passed** with real ONNX, ONNX Runtime 1.31.1, and imported FPGAI source compiled by GCC; synthetically generated ONNX weights, NOT a trained MNIST asset.

**New source-level scientific result:** `nonlinear_ci/summary.json` records **16 generated configurations across six fixed Functional graphs**, retaining the actual Flatten → Gemm(bias, optional transB) → Relu → Gemm(bias) source and varied row blocks. **6/6 selected GCC numerical checks and 3/3 executed row-epoch traces passed**. All sample weights are deterministic synthetic controls. This establishes supported end-to-end nonlinear CST lowering, not arbitrary neural-network coverage or trained-model accuracy.

Independent E0 replay at this same source SHA still reports **3,600/3,600 agreement** (150 legal, 3,450 illegal, zero errors) against a constructive bounded oracle; controlled schedule interventions **30/30 legal accepts and 138/138 designed invalid rejects**; checker scaling **15 graphs and 126 raw CPU timing samples**.

**Prepared but not executed:** `experiments/cst_trained_mlp_host.py` runs exact checksum-verified original trained MNIST ONNX vs ONNX Runtime, full Functional IR, and generated GCC C++; `experiments/cst_trained_mlp_native_prepare.py` creates source/testbench/Tcl packages with original trained weights and labels for later original Vitis. Neither can run against actual trained-model files from GitHub-hosted CI because the private trained assets were not attached or checked into the public repo. No real trained MNIST CST experiment, Vitis CSim/CSynth, FPGA physical QoR or prior-art parity result is claimed.

**Scope:** sequential MatMul/Gemm/Relu plus leading Flatten, with complete topological operator preservation and rejection of unsupported parallel/unary and branching/Conv paths. Batch-one MNIST provides no B-row scratch reduction. Real trained asset hashes, native vendor evidence and direct comparator implementations are still required for strong publication claims.

## Certified policy-neutral decision IR — first verified E0/E1 study (2026-11-02)

**Original source-pinned run:** [GitHub Actions 41377827461](https://github.com/umutcanaltin/fpgai_research/actions/runs/41377827461), compiler revision `3564d4b829adbe5328935fffe42c62a1be630feb`, source-evidence artifact `13111808584`. Both the quick and paper one-command experiment series finished successfully with **17/17 stages PASSED each** (no failed/blocked stages); the new `decision_interface` stage ran in both profiles.

**Paper-profile independent restricted action study:**
- **4 prespecified static MatMul-chain graphs** and two documented target parallelism caps; **52 original action candidates** including invalid/no-op/over-budget cases, **24 accepted** and **28 rejected** by the actual `resolve + compile_cst` transition gate.
- **52/52 agreements** with a separately implemented *restricted algebraic* PE/SIMD/B-row action oracle (0 false-legal, 0 false-illegal *within that limited domain*); every case and source SHA retained in `series_paper/decision_interface/cases.csv`.
- **3/3 selected actual GCC-host numerical checks passed** for accepted source programs.
- **8 one-action source-level policy demonstrations** (seeded uniform-random-legal and a fixed simple source-scratch heuristic), with identical compiler transition interface and budgets; **zero trained RL policies, zero original Vitis calls**.
- Observation tests cover supported variable-size graphs and mathematically equivalent graph-renaming invariance; proposal results include deterministic source-hashed certificates and fail-closed rejection reasons.
- Source-attested outputs: `series_paper/decision_interface/summary.json`, `cases.csv`, `policies.json` and `series_paper/evidence_index/paper_evidence.json`.

**Interpretation and stop rule:** The experiment verifies that policy-neutral finite complete-design actions can share one source compiler and restricted action mask; it does **NOT** demonstrate a novel policy, real hardware feasibility, RL cross-network transfer, full Phi-derived element readiness, multi-reader last use, or firstness relative to CompilerGym/RL4ReAl/Allo/ST-Flow. That stronger semantic novelty remains the next required research gate L4/L5.

## RL-ready certified CST IR — executed source-pinned experiment milestone

At **immutable tested revision `2b2abe023fc5318698426c85687129d7ca109723`**, [full regression CI](https://github.com/umutcanaltin/fpgai_research/actions/runs/41426710931) passed **964 tests, 5 skipped**. [Quick and paper whole-series CI](https://github.com/umutcanaltin/fpgai_research/actions/runs/41426710838) each passed **21/21 stages**, zero failed, zero blocked. Raw E0/E1 artifact [13132951180](https://github.com/umutcanaltin/fpgai_research/actions/runs/41426710838/artifacts/13132951180) includes create-only results, source hashes, individual logs and conservative paper evidence index.

New bounded studies in the auto-series: `mapped_event_causality` (three fixed legal operator mappings, stale Phi-owner events, premature commits and early-read negative interventions), `fanout_e0` (abstract two-consumer last-use release and early overwrite), and `mapped_source_e1` (three actual GCC HLS source traces matched against separate Phi-derived logical value obligations). Certificate-aware search actions mark mapped proof absences `UNKNOWN`, not legal. These are **not** claims about arbitrary general DAG source lowering, achieved clock/Fmax, vendor resources or trained RL sample efficiency. No original Vitis CSim/CSynth, FPGA routing or external prior-art baseline was executed by this CPU workflow.

The frozen RL research source remains unchanged: 42 inspected `rl_environment` Python files matched the backup revision `5221094af585bd501e842deca94d19c2289bb628`.

## Restricted two-reader CST E0 allocation and E1 source lowering — 2026-11-02

[Immutable whole-series run 41429203318](https://github.com/umutcanaltin/fpgai_research/actions/runs/41429203318) at `d2b5a45a72d8608b0a9e235d9233447fbbb690f3`: **23/23 quick stages PASSED, 23/23 paper stages PASSED, zero failed/blocked.** Original E0/E1 [artifact 13133591865](https://github.com/umutcanaltin/fpgai_research/actions/runs/41429203318/artifacts/13133591865).

- `fanout_exact_e0/`: Four prespecified bounded abstract ACK/address allocation cases, of which two have exhaustive finite minima, one has no solution under a declared scratch cap, and one is explicitly UNKNOWN under a candidate budget. The source independent oracle in tests enumerates every offset mapping in small domains.
- `fanout_source_e1/`: Three nondegenerate static MatMul fanout shapes, each with source-disjoint versus h/g last-reader-reused scratch. **Six actual compiled GCC host-float numerical checks and six executed source event traces passed**, with input/source/Functional SHA. Each source displays 4MN vs 3MN scratch *elements*; no hardware BRAM result is reported.
- The generated diamond is a **profile-restricted five-operator prototype**. It does not implement arbitrary fan-out, per-Phi mapped multi-reader proof, fixed-point CSim qualification, concurrent bank ports, RTL or measured HLS QoR. Trained original MNIST, licensed original Vitis campaigns, and independent published-tool comparisons remain outstanding.
- The E0 exact optimizer searches the canonical abstract stage-order; the emitted C++ is checked using its distinct real per-element event order. The E0 abstract capacity optimum is **not asserted** to be optimal for the executed C++ implementation or actual FPGA hardware. No RL training/native Vitis was launched by this CI.

## Exact fan-out allocation against *executed source order* (2026-11-02)

[Immutable CPU original-series run 41430232732](https://github.com/umutcanaltin/fpgai_research/actions/runs/41430232732) at `47390b43ad58f1bf7df2e1f2c8193d4bf4a07c34`: **24/24 quick and 24/24 paper stages passed**, no failed/blocked stages. [Artifact 13132714920](https://github.com/umutcanaltin/fpgai_research/actions/runs/41430232732/artifacts/13132714920) contains the source-pinned case manifests and new `fanout_source_optimum_e1/` files.

Three distinct static MatMul fanout graph sizes were evaluated, each with original disjoint versus **exactly optimized under the source's sequential E0 event relation** scratch addresses. All six cases were GCC host-float-executed, independently checked numerically and event-traced; all three optimum allocation certificates found strictly smaller source scratch than their matched disjoint controls. A separate 2×2 exhaustive offset-tuple verifier checks the reported 8/16 source-element minimum within the same explicit source event model. The original E0 abstract stage-order allocator remains a different control. None of these checks establishes original Vitis fixed-point numerical equivalence, actual HLS utilization/timing/BRAM, multiport arbitration, Phi-conditioned concurrent fanout, or published-tool superiority. No vendor program or RL training executed in this experiment series.

## MatMul-Phi-conditioned two-reader source experiment (2026-11-02)

**Verified immutable source:** `bc796e133a5f5f87b36b74b2727537e94b702a8b`, [original whole-series CI 41432640743](https://github.com/umutcanaltin/fpgai_research/actions/runs/41432640743), **25/25 quick and 25/25 paper stages passed**, with zero failed/blocked stages; raw [source-pinned artifact 13135006870](https://github.com/umutcanaltin/fpgai_research/actions/runs/41432640743/artifacts/13135006870).

The `fanout_phi_source_e1` stage presents a prespecified three-case producer mapping/shape catalog with original generated GCC source: 3/3 actual host-float numerical comparisons and 3/3 independent source loop-Phi/reduction/read/last-use checks passed. Each case retains full Functional fingerprint, resolved architectural fingerprint, generated C++ SHA, Phi iteration SHA, trace event hash, mapping choices and numerical error. Separate negative tests mutate the producer ordinal and reject incorrect mapping, fixed-point format, bounds and unsupported non-scalar downstream mapping.

This result is **E0 source-loop Phi projection plus E1 executed GCC host-float verification only**. It does not prove source-to-Vitis RTL correctness, original fixed-point CSim, physical timing, BRAM/port use or general branched graph support; it does not support a firstness/novelty or RL sample-efficiency claim.

## Two independently mapped MatMul readers — original full E0/E1 evidence (2026-11-02)

**Original immutable source:** `425772d6be0f26f036ff48564f5b6fb571b94186`. Original GitHub Actions compiler IR experiment [run 41442961276](https://github.com/umutcanaltin/fpgai_research/actions/runs/41442961276) completed **26/26 quick** and **26/26 paper** stages with zero failed/blocked. Retained [run artifact 13140229048](https://github.com/umutcanaltin/fpgai_research/actions/runs/41442961276/artifacts/13140229048). Original stage `two_phi_readers_e1`: **24 total E1 GCC source cases, 24 numeric passes, 24 executed source trace/refinement passes**, comprising **three graph shapes, four source mapping policies, and 12 disjoint/reused scratch-layout pairs**. All twelve layout pairs preserved Functional/Architectural hashes and logical event digests and reduced **source scratch array extent**, not measured HLS BRAM.

Each retained case records Functional graph, resolved architecture, C++ source, producer Phi, reader0 Phi, reader1 Phi, exact source event, last-use SHA, 2MNQ expected H-read multiplicity, host-float numerical error and source scratch slots. Negative tests reject unknown read owners, missing/duplicate contributions, overwritten live H values, altered source ordinal and mismatched architecture. Source hashes and raw-case denominator remain checkable after the run.

**Evidence ceiling:** E0 typed bounded logical source-order contracts plus E1 *actually compiled* GCC host-float C++ behavior. Three MatMul operators are independently mapped, while Add/Relu remain scalar; static five-op diamond only. No original Vitis tool invoked, and the source scratch reduction does not establish physical RAM occupancy, DSP/LUT counts, timing, energy, concurrent access legality, or SOTA superiority. RL source-pinned historical work remains untouched.

## L5f, L6b and L7 original source-pinned checkpoint — 29-stage series (2026-11-02)

**Verified exact code SHA:** `a8e03eafd68262c9414e6e9d579c33ac163c67d2`; [original 29-stage quick/paper run 41446968133](https://github.com/umutcanaltin/fpgai_research/actions/runs/41446968133) **29/29 passed in quick and 29/29 in paper**, 0 failed/blocked; [original raw artifact 13143062150](https://github.com/umutcanaltin/fpgai_research/actions/runs/41446968133/artifacts/13143062150).

- **L5f exact address allocation:** `branch_exact_allocator.py` returns finite `OPTIMAL_E0`, `UNKNOWN_BUDGET`, `INFEASIBLE_WITHIN_CAPACITY` under exact Phi owner/multi-read lifetime intervals and fixed sequential source-event order. Independent brute-force offset search tests on tiny instances. Three finite source-layout minima, six source-matched original GCC numerical and trace passes, plus an UNKNOWN budget control; **zero native Vitis calls**. Source-array extents are not BRAM/energy measurements.
- **L7 controlled E0 ablations:** seven prespecified logical contract cases, two legal/five mutated against full typed specification. False acceptances **relative to that full E0 specification**: metadata-only 5/5 invalid; deliberately naive first-read-ACK 4/5; cardinality-without-Phi-owner 1/5; full Phi-read-and-last-use 0/5. These are *weak internal proxies*, not run peer compilers, independent RTL truth, or empirical hardware QoR.
- **L6b explicit abstract bank-port UNKNOWN:** nine prespecified source-event/cycle hypothesis cases: 3 absent-schedule UNKNOWN, 3 caller-declared **abstract** feasible, 3 abstract port-overloaded rejects; **zero hardware-qualified** cases. No physical cycle inference, actual bank partition, Vitis CSynth or port-resource claim.

The full 29-stage series records `hardware_quality_claim_established=false`, `prior_art_novelty_claim_established=false`, `trained_mlp_executed=false`; this is scientifically correct. Upstream source/artifact pin inventory lives in [PRIOR_ART_PINNED_ARTIFACTS.md](PRIOR_ART_PINNED_ARTIFACTS.md) and is NOT an executed cross-compiler comparison. Frozen historical RL is untouched.

## Graph-wide mapped edge witnesses — source-pinned L6a CPU/GCC study (2026-11-02)

**Code SHA `c1298ea35bf526b743f39c910615aa72f29afadf`**, original [GitHub Actions one-command run 41450443712](https://github.com/umutcanaltin/fpgai_research/actions/runs/41450443712): **30/30 quick, 30/30 paper stages**, 0 failed/blocked; [raw original artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41450443712/artifacts/13144403380). Original full [CPU regression 41450443849](https://github.com/umutcanaltin/fpgai_research/actions/runs/41450443849): **1,033 tests passed, 5 skipped**.

Stage `graph_phi_e1`: 3 prespecified graph topologies (6/7/8 operators), 2 MatMul architecture policies and 2 scratch layouts = **12 actual original GCC host-float numerical validations and 12 independent GCC source/refinement trace passes**, six legal disjoint-versus-post-last-use source scratch reductions, zero original Vitis. Raw `series_paper/graph_phi_e1/cases.csv` retains per-case Functional/architecture/C++/source-event/last-use hashes, graph/edge counts, Phi-sha map, host-float numerical error, source scratch elements and pass flags. Summary and source manifests retain the exact original source revision. Newly added `GraphEdgeWitness` and `GraphValueWitness` are first-class source-level artifacts, compared to a digest independently rederived from the GCC trace; they are **not** formal hardware proofs.

Important negative controls include a **modified source read index that stays within the array bounds**, rejected because the same index drives actual memory load and trace, and missing/duplicated consumer reads, forged Phi owner, early-live alias and mismatched source identity. The original 30-stage experiment run was executed only after these controls passed.

**Ceiling:** static 2D graph subset, source-order E0 + executed GCC host-float E1; no original fixed-point Vitis, qualified hardware banks/clock, RTL, arbitrary ONNX DAG, external peer artifact equivalence or priority over prior art. See [graph-wide proof obligations](GRAPH_PHI_BOUNDARY_CERTIFICATES.md). RL historical runs and policy source unchanged.

## Nonidentity graph source access + independent semantic oracle original evidence (2026-11-02)

Immutable original experiment SHA `dbf6aade8228cbbe703037fb5108715cd0dd64f9`; [GitHub Actions original 32/32 quick/paper run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625112) **32/32 per profile, zero blocked or failed**. Original [artifact 13146491650](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625112/artifacts/13146491650); [CPU regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625106) **1,048 passed, 5 skipped**. Prespecified `graph_access_e1` cases: **16/16** actual GCC host-float numerical and **16/16** source read/COMMIT/Phi/edge-witness trace passes; **8/8** source scratch disjoint-vs-reuse pairs. Four graph/shape configurations × two independent mapping policies × two storage layouts; full raw cases retained. Original Vitis calls: **zero**.

`graph_semantic_oracle_e0`: 4 positive graph-access controls and 4 in-bounds bijective semantic source-index mutants. A deliberately correlated faulty access helper was shared ONLY within the synthetic lifetime verifier and its event source: **4/4 mutant false accepts** versus independent NumPy grid oracle **0/4 false accepts**, all 4 valid controls accepted. These comparisons are relative to NumPy source-index semantics, NOT hardware truth, third-party compiler failures or stochastic generalization. No experiment row omitted. [Exact source scope and limitations](NONIDENTITY_GRAPH_ACCESS_AND_ORACLE.md).

## Original Conv2D kernel source/load/NumPy evidence (2026-11-02)

Original immutable code `6af10f9e2ec71810181b51560d227d8dba133303`. [Whole 33-stage quick/paper run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41456356235) passed **33/33 quick and 33/33 paper**, 0 failed/blocked, [raw artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41456356235/artifacts/13146878908); [CPU regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41456356220) passed **1,061 tests, five skipped**. Source-pinned `conv_phi_boundary_e0_e1_numeric`: **8/8 independent E0 padded-kernel source access checks, 8/8 real GCC generated Conv host-float numeric results, 8/8 actual GCC per-kernel Conv reader source LOAD address/Phi trace checks, and 8/8 early-alias/post-last-reader E0 logical controls** (4 kernel/stride/pad geometries × 2 Conv mapping policies). All eight rows retained as `series_paper/conv_phi_boundary_e0_e1_numeric/cases.csv`, alongside summary and original GCC log. A mutated in-bounds actual C++ load address is rejected.

Important: source trace is **only Conv reader per-kernel reads**, not a full timeline of all scalar READ/COMMITs. The E0 logical buffer offsets are NOT emitted as shared native source scratch for the canonical Conv emitter. Original Vitis calls **zero**; no hardware scheduling, physical BRAM saving or scientific priority inferred. [Technical source scope](CONV2D_NCHW_BOUNDARY_PROOF.md).

## Source-pinned generalized NCHW graph E0 / partial original E1 checkpoint (2026-11-02)

Original source `fcb1f621bb6c0eaadae8f2206e823f59bd2bf21c`; [original GitHub Actions run 41473339211](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339211): **34/34 quick, 34/34 paper, zero blocked/failed**. Original [artifact 13156047647](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339211/artifacts/13156047647), [full regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339307): **1,070 passed, 5 skipped**.

New stage `nchw_graph_e0_partial_e1`: 4 graph/shape cases from three distinct families (7-,5-,6-operator DAGs) ×2 mapping policies ×2 mathematical scratch layouts = **16 retained data rows**, **8 unique original GCC HLS C++ executions** (one execution reused for two logical source-layout hypotheses). E0 lifetime/witness and independent NumPy read semantics passed **16/16**; GCC source numeric and actual Conv intermediate kernel-load/owner attestation reported **16/16 rows**, **8 distinct source runs**. Eight safe matched logical source-array reductions; `physical_alias_emitted=false` in all 16 cases. Exact Functional/Phi/C++/logical event/last-use/witness/independent source-map/GCC-trace hashes and full CSV retained.

No native Vitis CSim/CSynth, qualified HLS memory bank/port, source-ordered non-Conv READ/COMMIT, or executed third-party compilers. Source savings are **not** hardware savings. [Complete gates](GENERAL_NCHW_CST_EDGE_CERTIFICATES.md).

## Original GCC all-intermediate and ACTUAL shared-source scratch E1 (2026-11-02)

Immutable source `319ab4e94a57019d0249dbe2254a925e748e4ace`: [original 36/36 quick + 36/36 paper stage run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917065) — zero failed/blocked, raw [artifact 13157444356](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917065/artifacts/13157444356). [Full CPU regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917137): **1,083 passed, five skipped**. Two new mandatory stages **`nchw_all_intermediates_e1`** and **`nchw_real_source_scratch_e1`**.

`nchw_all_intermediates_e1`: **8 unique original GCC C++ executions** (four graph settings × two mapping policies), all eight accepted by **post-operator intermediate tensor numerical/identity comparison** against independent NumPy; **50 real operator snapshots and 1,764 observed tensor elements** recorded. Source/C++/graph/architecture hashes, per-operator max errors, all tensor SHA values in original CSV/JSON. This is NOT per-STORE/READ source scheduling evidence.

`nchw_real_source_scratch_e1`: actual two source layouts (fully disjoint local C++ scratch and conservative whole-operator reuse), **16 different generated C++ source variants across eight matched pairs**. Original GCC executes each source twice for independent intermediate-value and actual Conv input-load/Phi checking: **32 original GCC runs, 16/16 numerical and 16/16 Conv read checks passed**. Every one of the 8 paired **real local C++ scratch allocations** is smaller under reuse (minimum **8 elements**, sum of pair differences **408 elements**). The external output remains distinct; C++ memory sharing is materialized, NOT inferred from abstract source addresses alone. **Zero native Vitis calls, zero hardware-qualified cases, no measured BRAM/ports/timing.** Original case CSV and full receipts retained. [Exact scientific scope](ACTUAL_NCHW_INTERMEDIATE_AND_SCRATCH_EVIDENCE.md).

## Observed real C++ READ/WRITE and matched native-source packages (2026-11-02)

**Actual original full-access GCC run:** source SHA `77fe5dfcd96ad81ff8d71f36fe57cb88a47411e4`, [37/37 quick and 37/37 paper](https://github.com/umutcanaltin/fpgai_research/actions/runs/41480551778), [raw artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41480551778/artifacts/13159323224); full regression **1,092 passed, five skipped**. Stage `nchw_real_source_rw_e1` executed **12 original GCC sources**, 3 graph families × 2 mapping policies × 2 actual local storage layouts. **12/12** observed real produced-tensor READ/WRITE order/identity, scratch overwrite and final NumPy checks; source traces counted **3,424 actual reads and 1,928 writes**, six lower-capacity real shared-source pairs (minimum eight C++ elements). The independent oracle includes ReLU conditional second reads and MaxPool's extra seed read. No original Vitis calls.

**Native package-preparation original run:** source `e5502be096bd4b8ab0cf15b82e2ad4c96d0eac84`, [38/38 quick/paper](https://github.com/umutcanaltin/fpgai_research/actions/runs/41481941961), [raw artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41481941961/artifacts/13159804282); full CPU **1,095 passed, 5 skipped**. Stage `nchw_native_pair_packages_e1` created exactly **12 matched actual source/testbench/Tcl packages**, six source storage pairs, original Vitis calls **zero**, native CSim/CSynth **NOT_RUN**, hardware-qualified cases zero. No fake resource or timing measurements. Later audited case-manifest integrity and guarded native execution are separate source changes requiring their own CI.

All source claims and exact workload/clock/numerical conditions: [NCHW source event and Vitis handoff](NCHW_SOURCE_ACCESS_AND_NATIVE_HLS_HANDOFF.md). Publication originality/physical QoR remain unqualified.

## First complete original Vitis E3 source-pair study, and auto binding pilot — user transcripts, 2026-11-02

The user supplied original terminal output for a **12-case, six-pair explicit Vitis `bram_2p` binding study** generated at `8814bd0005e8a852348b49f19131b2410117216b`: 12/12 CSim qualified and 12/12 CSynth + 8.64 ns uncertainty-adjusted timing qualified; zero physically implemented cases. BRAM_18K pairs: **5→5, 64→64, 9→9, 1→1, 32→32, 7→12**. Worst-case cycle pairs: **1005→975, 588→586, 757→758, 537→537, 425→426, 1659→1633**. The user-provided original 12-case audit and paper-export summary claim all six pairs source-attributed, with zero missing vendor cases. These are *user-reported original Vitis transcripts*, not raw vendor files archived in this repository; avoid reclassifying them as published, post-route, or independent artifact verification.

New source-inference pilot at `8fe0ef31f6acc69c25902fe3f7ab94757e6e42bd`: only cases 000 and 001 executed, both CSim/CSynth/timing qualified; each **0 HLS-estimated BRAM_18K, 975 cycles, HLS margin +14.267 ns**, while the corresponding older explicit `bram_2p` cases are **5 BRAM_18K and 1005/975 cycles**. Remaining **ten auto policy cases are presently UNKNOWN**. No physical FPGA metrics; no default directive or forced reuse conclusion.

**Research code progression:** source-pinned E3 original effects analyzer in `cst_nchw_native_effects`; new `cst_nchw_cross_policy` performs independent read-only 2×2 source equivalence and native audit, allows different original Git revisions ONLY if generated C++ differs exactly by the single requested BRAM pragma, and keeps all source/case denominators. [Source audit, CLI and scientific limits](NCHW_SOURCE_ACCESS_AND_NATIVE_HLS_HANDOFF.md). RL stays frozen.

## Native HLS-to-Vivado OOC ROM packaging correction (2026-11-02)

Original attached workstation output showed explicit-BRAM cross-policy source audit succeeded (two auto-vs-explicit source pairs qualified; ten AUTO cases still missing), but original Vivado source preparation stopped at an original `*_ROM_AUTO_1R.dat` file with `UNSUPPORTED_HLS_MEMORY_INIT_ASSETS`. This is **preparation failure**, not original Vivado failure or E4 data. Corrective code now preserves all eligible original local `.dat/.mem` Vitis RTL ROM images, hashes them as distinct research evidence, stages exact bytes under Vivado cwd and Verilog directory, registers them with Vivado project as memory initialization files, records strict input stability in new Vivado invocation receipts, and keeps historic Vitis ROM provenance explicitly legacy when its preexisting source receipt lacked a ROM field. Unknown IP setup assets remain unsupported and fail closed. All physical results remain UNKNOWN until genuine original routed Vivado reports. See [native policy and OOC handoff](NCHW_NATIVE_BINDING_AND_VIVADO_OOC.md).


## New 2026-11-02 source-only completion protocol — unverified code

Original user-provided workstation transcript: 71 focused/1182
full tests passed on the then-current main; create-only physical
`build/nchw_bram_vivado_rom_v6` contains 12
`PREPARED_NOT_RUN`, zero original Vivado receipts and a six-pair
source population. No physical v6 measurement exists.

Subsequent new unexecuted source modules:
`cst_research_completion` foreground case-by-case native continuation,
`cst_paper_gate` source-matched E3/physical/novelty blocker report,
and `cst_phi_counterfactual_effects` source-only causal
analysis of actual two-reader GCC records. Their tests are
unverified until an actual user workstation log is received.
The new studies/exports must be create-only and all original
failures preserved.

Original Vitis `bram_2p` six-pair study source revision
`8814bd0005e8a852348b49f19131b2410117216b`.
Neither this revision, nor old Vivado v3-v5, nor frozen RL
studies may be edited or rerun.
Reference: [FINISH RESEARCH EXECUTION](FINAL_RESEARCH_EXECUTION.md).


## 2026-11-02 late original workstation: 40/40 complete CPU and first routed matched pair

**Original uploaded terminal evidence** records a full committed
`1cefdd8566658cb6b69e12444c9b5ce9ba6e4a31` checkout and:
- `python -m compileall` clean; **1228/1228 full pytest passing**,
  no failures, from the final complete output;
- source-pinned paper CPU/GCC series at
  `build/cst_c123_paper_20261103`: **40 PASSED / 0 FAILED /
  0 BLOCKED**, including real ONNX crosscheck, CST value traces,
  mapping-dependent multi-reader source, the new R3/R4 RL-compatible
  interface study and 24-case Phi counterfactual effects;
- the series explicitly reports `source_evidence_qualified=true`,
  `all_original_hls_pairs_qualified=false` *within this new
  source-only series*, `native_tool_use_requested=false`,
  `trained_mlp_executed=false`,
  `prior_art_novelty_claim_established=false`. The *earlier*
  original 12-case Vitis native study is a **different
  independently source-pinned cohort**, not canceled by this flag;
- v6 `case_001` original foreground Vivado 2023.2, process
  duration **99 seconds, exit 0**; independent routed status
  `OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS`,
  `routed_observation_valid=true`, `routed_qualified=false`.
  Original `case_000` remains the same accepted warning-only
  status (102 s, exit 0). **Two** original Vivado receipts;
  **two** valid warning-only routed cases; **zero** strict
  clean-DRC E4 cases; 10 remaining prepared. The first
  source-matched disjoint/shared pair is ready for independent
  **descriptive** original report comparison; its numeric LUT,
  FF, DSP, BRAM and WNS differences are NOT in the transcript.

These completed source stages are meaningful E0/E1 scientific
engineering evidence, but they do **not** establish a trained RL
advantage, Phi-induced *actual* last-use difference, scientific
firstness or native physical resource improvement; inspect the
original, hash-verified summary and original routed case reports
before any specific result claim.

A new read-only `experiments/compiler_ir/summarize_research.py`
reviews the immutable 40-step CPU/GCC source cohort and the
independently audited 12-case Vivado cohort together while
**explicitly refusing cross-revision numerical pooling**. It
reports raw per-case resources, DRC rules, WNS and descriptive
disjoint-minus-shared differences for an original pair ONLY if
both original cases are valid audited routed observations.
Source-only counterfactual and RL case statistics are surfaced
without imputing missing physical evidence. New script/tests
are **not yet verified on the user's workstation**.

## 2026-11-03 00:12:41 — final original six-pair export and independently audited paper gate

Original latest workstation source commit was
`3f4e8bfec209a4a06e293674137920610d416da3`.
`python -m pytest -q` **1,240/1,240 tests PASSED**.
All three create-only original audited commands completed:
`build/nchw_v6_full_physical_export_v1`,
`build/cst_full_cross_cohort_review_v1`,
and `build/cst_original_paper_gate_full_v1`.
Zero Vivado invocations were performed by their
audit/export runs. Full source-pinned v6 original
HLS/physical source SHA:
`8814bd0005e8a852348b49f19131b2410117216b`.

Original HLS **12/12 cases and 6/6 source-matched pairs
qualified**; physically all **12/12 routed observations**
and **6/6 paired observations** were independently
accepted with noncritical DRC warnings, but strict
clean-DRC E4 **0/12 cases, 0/6 pairs**.
All external published-compiler parity gates
remain NOT_ESTABLISHED. Original DRC warnings
present in all six pairs. No new vendor tool
was executed during source export.

Original paired values (disjoint minus shared),
in strict prespecified pair order 000/001,
002/003, 004/005, 006/007, 008/009,
010/011:
- BRAM `0, +8, 0, 0, 0, -2.5`;
- LUT `-155, +57, -4, 0, +9, -350`;
- FF `-10, -234, -120, 0, -507, -21`;
- DSP `+3, 0, 0, 0, 0, 0`;
- WNS ns `+0.195, -1.470, +0.537,
  -1.182, +0.707, -1.046`.
- BRAM original per-pair disjoint vs shared:
  `(2.5,2.5), (32,24), (4.5,4.5),
  (0.5,0.5), (16,16), (3.5,6)`.

This evidence supports a heterogeneous
**physical-design tradeoff**, not a universal
physical BRAM improvement or scientific novelty.
The separately pinned paper CPU/GCC cohort
reports strong-Phi causal mapping conditional
last-use and source scratch capacity changes
**zero across 18 effective mapping edits**.
The original C3 source-only RL interface supports
bounded complete-design actions and certificate
masks, **not an observed learned agent outcome**.
Full write-up and scientific limits:
[ORIGINAL_SIX_PAIR_PHYSICAL_RESULTS_20261103.md](ORIGINAL_SIX_PAIR_PHYSICAL_RESULTS_20261103.md).


## November 3, 2026 — original R2b executed-GCC source study qualified

Original user workstation performed `git pull --ff-only` to
`5d449ca595dbdfdf7a837ce862eb5fa8f89ce9ce`;
`python -m compileall` successful,
`tests/test_cst_r2b_source_campaign.py` **8 PASSED**,
full pytest **1253 PASSED**. The prospective, NEW
create-only original run `build/cst_r2b_original_gcc_e1_v1`
produced **24/24 verified original GCC compilations,
24/24 GCC executions and 24/24 independent NumPy/event
checks**, without original Vitis or Vivado calls.
The original separate read-only source audit verified
**24/24** saved source SHA, output trace SHA, E1
READ/COMMIT events and H-only READ positions.

The exact original summary reports:
`changed_original_executed_h_last_read_contrasts=18`,
`source_read_only_rank_or_owner_changed_contrasts=12`,
`source_read_only_rank_or_owner_unchanged_contrasts=6`,
`source_final_reader_identity_changed_contrasts=6`,
`source_storage_pairs=12`, and
`mapping_conditioned_source_capacity_difference_established=false`.
The 18 contrasts include each of 9 graph/Phi
interventions duplicated across disjoint and
last-reader-reuse layout controls. The scientifically
honest unique-denominator result is **6/9 final
H-only read rank/consumer differences** and
**3/9 changed final consumer identities**,
with 3/9 source read-order ties. The original
fixed-serial source `0/18` last-use/capacity
result remains unchanged.

A new read-only second-level
`cst_r2b_causal_audit` has been committed
to reconstruct the 9 unique interventions,
numeric original graph and input and both
source layout controls directly from original
24-case files. Its new tests/code have
**not yet been verified by the workstation**.
The original native v6 routed experiment is
separate, with **12/12 valid warning-disclosed
observations but 0 strict E4** cases.
No scientific firstness over outside compiler
work or RL advantage has been established.
See [R2b complete original E1 observation](R2B_E1_ORIGINAL_RESULTS_20261103.md).

## November 3, 2026 — follow-up read-only original R2b causal audit

User's original saved `build/cst_r2b_original_gcc_e1_v1`
24-case E1 GCC study was independently audited in
`build/cst_r2b_independent_causal_audit_v1`
with **no new original GCC/Vitis/Vivado calls**.
Original E1 source SHA remained
`5d449ca595dbdfdf7a837ce862eb5fa8f89ce9ce`.
All 24 original sources, stdout provenance and
NumPy outputs were rechecked. The two storage-layout
controls for each graph/Phi produced identical event
and numerical results. Therefore the 18 layout-scoped
interventions are **9 distinct graph/Phi edits**:
global source-event index changed 9/9,
actual H-only last reader rank/consumer changed 6/9,
final consumer identity changed 3/9, and H-only
last-reader state was unchanged in 3/9.

The three graph shapes' reported scratch source
allocation control differences were 8, 8 and 12
logical slots, respectively, **constant across all
Phi variants**. This is *not* a Phi-induced capacity
saving. Hardware/RTL clock cycles, HLS/physical
resource improvement and external compiler prior-art
novelty remain unestablished.

On the same workstation, focused audit tests returned
**3 passed and 1 failed**; full regression returned
**1256 passed, 1 failed**. The only failure
was mismatch of an expected error regex in the
synthetic duplicate case. Lower-level original
source audit rejected the duplicate as intended.
Source check logic and original evidence did not
fail. A test-only repair and an independent
unsupported mapping-mutation check have been
committed subsequently, but not yet verified
on the user's workstation.


## 2026-11-03 — R2c source allocation implementation, original execution PENDING

At source parent
`578178890607fa085ef0c30a9a270e123108a3bb`
the user's workstation passed focused 5/5
and full **1258/1258** Python tests.
New follow-on R2c files were subsequently
committed under `src/fpgai_research/model_ir/`,
`src/fpgai_research/codegen/hls/`,
`src/fpgai_research/validation/`,
`src/fpgai_research/experiments/`
and `tests/`. They implement a bounded
source-event scalar-unit interval-graph allocator,
independent exhaustive finite reference oracle,
actual source C++ slot materialization and GCC
trace+NumPy verification, prospective 12 matched
original paired source cases and read-only auditor.
**No R2c original GCC study or new Python regression
has run on the workstation at this checkpoint.**
Original R2b, historical native v6 and original
RL source remain unchanged.

The strict accounting variable is
`declared_mutable_source_array_elements`:
the allocated C++ scratch array plus 4*m*q
other declared mutable array elements
(`acc_a`, `acc_b`, `add_a`, `add_b`).
Scalar local temporaries are explicitly excluded;
this is NOT synthesis or physical FPGA BRAM data.
Conservative scientific contract:
[R2C_EXACT_VALUE_LIFETIME_STORAGE_STUDY.md](R2C_EXACT_VALUE_LIFETIME_STORAGE_STUDY.md).

## 2026-11-03 — original 96-case C2 E0 mutation adequacy completed

At original source `82ef9f2c0fe101dadcf6a1c013540b8f3780abf6`,
the user ran the new E0 original create-only
`build/cst_C2_e0_mutation_adequacy_v1` study.
All **96 prespecified mutant cases** were rejected:
8 fault classes × 3 graph shapes × 4 Phi reader policies,
including 12 premature A, 12 premature B,
12 premature Add, 12 illegal live H aliases,
12 wrong owners, 12 duplicate H reads,
12 missing H reads, and 12 safe-but-
Phi-issue-nonconforming source orders.
The last class was intentionally accepted by
the generic branch safety certificate (12/12)
and rejected by the separate independent
Phi issue policy specification (12/12).
No original GCC/Vitis/Vivado calls occurred;
the observed source E0 result is **not** an
independent proof of all future mutation classes.

Focused new remediation suite 21/21 passed.
Full regression **1294 passed, one failed** due to
an expected-error-regex mismatch in
`test_mutated_runtime_trace_and_source_identity_are_rejected`:
the independent oracle rejected the tampered
GCC trace before a generated-event sequence equality
assertion could run. A test-only correction
was committed to the new revision afterwards;
the user has NOT yet supplied post-correction
workstation test results. Never rerun, delete
or overwrite the original 96-case folder.

Follow-up F1 R2d code added after the original
cohort: loop-driven C++ source computes actual
H READ subscripts from a runtime ordinal, while
a distinct Python closed-form Phi issue reference
checks executed GCC stdout independently of the
generator's precomputed event list. New
source/order/numeric mutants, 12 original
prospective cases and a read-only SHA/NumPy
audit are committed. The R2d code and
all new tests are **unexecuted on the
user's workstation as of this record**.
See `R2D_RUNTIME_LOOP_INDEPENDENT_ISSUE_PROTOCOL.md`.
No previous original R2b/R2c/Vivado experiment
must be rerun or reclassified as R2d.

> **2026-11-03 original R2d runtime-loop GCC cohort — workstation VERIFIED.**
> At original SHA `09ac395ef5c73cae59940c4761f3255d695882d1`,
> `compileall` succeeded; focused R2b/R2d source tests
> **23/23 passed**, full pytest **1,310/1,310 passed**
> in 122.16s. Original new create-only source cohort
> `build/cst_r2d_original_runtime_loops_v1` completed
> **12/12** real GCC compilations and executions, with
> **12/12** runtime-loop Phi order, reduction/alias and
> independent NumPy checks qualified. The original
> separate read-only audit under
> `build/cst_r2d_runtime_independent_audit_v1`
> rechecked **12/12** stored source/trace SHA pairs,
> mapped READ order, alias/reduction, and NumPy
> numerical outputs, with zero new GCC/Vitis/Vivado
> calls. Source numerical absolute errors were
> approximately 5.93e-8 to 1.19e-7.
>
> This is **independently checked source-level
> runtime loop index behavior**, not measured
> FPGA-cycle scheduling, RAM port/BRAM resources,
> general DAG behavior or external scientific firstness.
> Original artifacts MUST NOT be re-executed or
> overwritten. Prior R2b/R2c and C2 E0 cohorts
> remain separately source pinned.
>

## November 3, 2026 — R2d original execution complete; R2e implementation pending

The user's source-pinned original R2d source study at
`09ac395ef5c73cae59940c4761f3255d695882d1`
is **complete**: 23 focused tests, 1,310 full pytest
tests, 12 original GCC runtime-loop cases, and all
12 independently read-only audited source/trace/NumPy
cases passed. There were zero Vitis/Vivado calls.
The new runtime issue order is source-level, not a
measured FPGA cycle schedule.

A **NEW, prospective and unexecuted R2e implementation**
now responds to the missing H producer contribution
certification: full separate runtime
`CONTRIBUTE` index/source witnesses, producer
`COMMIT`, typed H source `COMMIT` and adjacent
actual numerical H value witnesses. An independent
oracle checks every reduction term and input/weight
address; a GCC validator compares actual intermediate
H and final Y against separate NumPy arithmetic
checks and checks reader Phi and scratch legality.

The next **12-case original R2e GCC cohort** and
its separate read-only SHA/NumPy audit are
implemented but **have NOT been executed on the
user's workstation**. R2e code/tests have not
passed workstation regression yet. This is a
NEW source experiment, not retroactive proof
about the completed historical R2d source files.

Correct interpretation: the R2e source trace is
instrumented next to real C++ reduction operations,
and H/final numerical output is independently
checked. This is NOT a formal proof of all
possible numerical inputs, bit-exact `ap_fixed`
arithmetic, arbitrary graph producers, FPGA
hardware clocks, PE parallelism, BRAM, RAM port
legality or external novelty.

**Run the new R2e focused and full pytest suite
before launching any new R2e original campaign.**
See [R2e full prospective protocol](R2E_PRODUCER_COMPLETENESS_PROTOCOL.md).

## 2026-11-03 — original R2e producer completeness qualified; R2f fanout E0 unexecuted

**New original user-provided receipt, immutable original revision
`d41568c1f913e245692815f733a537799c6df7bd`:**
`compileall` succeeded; R2e focused tests **20/20 passed**,
entire pytest suite **1,330/1,330 passed**.
The independent R2e original source experiment
`build/cst_r2e_original_producer_gcc_v1`
completed **12/12 original GCC compilations/executions**,
with **312 complete producer contribution witnesses**,
12/12 source H reduction/COMMIT checks,
12/12 intermediate H numeric checks,
12/12 independent reader Phi/alias and final
NumPy checks. The separate read-only audit
`build/cst_r2e_readonly_producer_audit_v1`
confirmed **12/12** saved original source and
trace SHA, producer scalar-sum H numeric values
and full consumer numerical traces, performing
**0 GCC/Vitis/Vivado calls**. Original source and
all historical campaigns are frozen.

**Implemented afterward, but NOT tested on the
user's workstation:** CST R2f bounded compositional
source certificates for **2/3/4 reader MatMul
fanout**, complete producer/consumer/Add-chain
readiness, independent fixed Phi issue conformance,
source scratch liveness and a prespecified
**36-case E0** graph/mapping study. A safe
producer-reader interleaving is *logically*
admissible; it has no executable source/hardware
receipt. See
[R2F_COMPOSITIONAL_FANOUT_E0_PROTOCOL.md](R2F_COMPOSITIONAL_FANOUT_E0_PROTOCOL.md).

R2f has **not** demonstrated general DAG
compilation, a new interval-coloring algorithm,
actual GCC source-level translation,
native HLS, clean-DRC physical BRAM,
RAM port timing or external scientific firstness.
Research remains **not paper submission ready**.

## November 3, 2026 — R2f original E0 verified; R2g 36-case host-GCC E1 prospective

Original R2f cohort at
`98676f7786687eddace0cab9124868448f9bdc24`:
**27/27 focused; 1357/1357 full regression; 36/36 source
certificates; 18/27 changed logical unit-slot capacity;
9/27 ties; no regressions.** All 36 have complete
producer/reader/Add/final E0 readiness and independent
Phi source order; no original GCC/Vitis/Vivado.
This data cannot be equated to physical BRAM
or actual C++ optimized scratch reuse.
Historical original E0 source folder remains frozen.

New R2g source-level experiment implemented on main:
actual runtime-loop C++ read indices for fanout 2–4,
real producer and reader reductions, Add-chain writes,
actual committed numerical intermediate values,
separate forward-rank Phi and source readiness/alias
checker and NumPy for EVERY intermediate. New
**36 original GCC cases** and an independent
read-only SHA/NumPy audit are source-pinned, create-only.
An adversarial test suite injects wrong runtime ordering,
missing producer contribution, altered arithmetic,
wrong committed H value and early reader commits.
All are NOT YET workstation executed as of this
record. See `R2G_RUNTIME_FANOUT_E1_PROTOCOL.md`.

The remaining scientific proof gap is not "more tests":
R2g disjoint-source operation is different from
the 18-case R2f optimized *logical unit-slot*
counterfactual. A same-order actual optimized-source
comparison, Vitis CSim/CSynth, strict clean-DRC
Vivado measurement, exact external peer parity
and broader operators remain separate publication
gates. RL remains frozen.


## 2026-11-03 — Original R2g 36-case executable fan-out source fully verified

**Original source:** `0a92f4fa0630d81c51bb53ad04bb42304a67d2e5`.
User's workstation report: focused R2g tests **18/18 passed**;
full suite **1,375/1,375 passed**. Original source-pinned
`build/cst_r2g_original_fanout_36_gcc_v1` executed **36/36**
real GCC compiled source programs successfully across fan-out 2–4,
three shapes and four Phi reader policies. Every case passed
independent runtime source-event Phi/readiness/alias checking and
all-intermediate/final NumPy numeric comparison.

Original independent read-only artifact audit at
`build/cst_r2g_fanout_36_readonly_audit_v1` rechecked
**36/36** source/trace/input SHAs, complete producer/read/Add
events and intermediate numerical values, without new GCC,
Vitis or Vivado calls.

**Falsification boundary:** All original R2g C++ used **disjoint
scratch**, NOT R2f's per-element optimized scalar-unit layout.
The 18/27 R2f E0 logical capacity improvements are not source-executed
memory savings in original R2g, and no hardware-cycle/BRAM,
arbitrary-DAG, external firstness or trained RL claim is established.
Original R2f and R2g study directories and source SHA remain frozen.
