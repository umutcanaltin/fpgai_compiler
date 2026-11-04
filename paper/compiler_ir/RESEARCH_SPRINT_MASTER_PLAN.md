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

## 2026-11-03 — R2n C3 original GCC E1-qualified action screen IMPLEMENTED, workstation test PENDING

The new source-only, policy-independent `source_executed_action_mask` (`src/fpgai_research/model_ir/decision_interface.py`) differentiates `REJECTED` typed actions, `UNKNOWN_E0_CERTIFICATE` source-admissible but unmapped actions, `FAILED_E1_SOURCE_EXECUTION` and actual independently checked `E1_GCC_SOURCE_QUALIFIED` actions. It calls the **real compiler transition once per action** and, for mapped E0-qualified bounded static row-chain proposals, checks actual generated source GCC value/refinement trace and host-float numerical output against independent reference for an explicit fixed input. It fails closed on source identity mismatch, imposes an E1 call budget, and preserves failure outcomes instead of silently certifying them. **No RL training, Vitis, Vivado or physical feasibility**.

The prospective `cst_r2n_e1_action_study` covers 3 chain depths × 3 action categories, preserving all 9 outcomes in new create-only `build/r2n_c3_e1_qualified_actions_v1/`; no results have yet been executed on the user's workstation, so **no success counts are claimed**. New focused tests cover real GCC, source provenance forgery, false E1 promotion, unknown/reject/failure separation and create-only outputs; they are **newly committed, not workstation verified**. The new stage is wired into quick/paper compiler-IR series, with no GitHub Actions. Full instructions and conservative scientific limitations: [R2n C3 protocol](R2N_C3_E1_QUALIFIED_DECISION_IR_20261103.md).

This is an incremental **secondary C3 implementation**. Complete source-backed independent-Phi multi-reader DAG action semantics, partial decisions, bank/port feasibility and trained RL benefit remain open. The six original routed physical pairs and frozen RL environment remain unchanged.

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

## 2026-11-03 — R2i native fanout handoff implementation (UNEXECUTED)

After independently verified original R2h (1388 full tests, 36/36
matched source scratch reductions, 72 original GCC executions and
72/72 read-only source trace verification at
`efc6497cdab2ec5734a0c3750972655b9ea397b9`),
the new R2i study prepares the **entire exact 36-pair
population / 72 native Vitis source packages** from the
same R2h source layout lowering.

R2i reuses canonical Vitis `write_vitis_package`, numerical
`tb.cpp`, fixed-target/clock/uncertainty `run_hls.tcl`,
and existing provenance-aware vendor runner,
`summarize_case`, and `qualify_case`. The native
source and audit commands make **zero original vendor calls**.
Original native CSim/CSynth is **explicit opt-in**,
bounded by pair and requires both
`--execute-native --confirm-isolated-native`.
The native auditor shows incomplete/failed cases
as UNKNOWN/REJECTED (not zero benefit); complete
qualified HLS metrics are estimates, NOT achieved
FPGA clock or routed BRAM.

**The 72 prepared sources, native tool calls, E2/E3 reports,
and strict Vivado E4 results are NOT yet workstation-verified.**
Run focused/full tests, prepare new
`build/r2i_fanout_native_36_pairs_v1`,
and inspect a one-pair pilot before extending.
See [R2i experiment protocol](R2I_MATCHED_NATIVE_FANOUT_HLS_PROTOCOL.md).

## 2026-11-03 — original R2g completed; R2h matched actual source memory pending

**Source-pinned original R2g `0a92f4fa0630d81c51bb53ad04bb42304a67d2e5`:**
the user observed 18/18 focused tests, **1375/1375** full regression
passing, **36/36** original GCC runtime-loop C++ fanout programs
passing producer/reader/Add Phi and all-intermediate numeric
checks, and a separately executed **36/36 read-only audit**
of saved original source, runtime trace and numerical results.
The original R2g scratch layout was **disjoint**; no optimized
allocation was executed in that original cohort.
The original source/input/trace artifacts are frozen.

**New R2h (prospective; not yet executed on the workstation):**
A matched **36-pair / 72-GCC-program** source experiment
now emits control disjoint versus certified minimum
per-element scratch array layouts with the same runtime
fanout source order, Phi and input. Both arms write/read
through actual runtime C++ table-selected scratch
addresses and print the **same C++ address variable**
next to the source memory operation. All actual event
addresses, full producer/reader/Add source readiness,
Phi issue and intermediate/final numerical values
are audited independently, with a separate read-only
72-source verification campaign and mutation tests.

**Scientific limits:** If R2h passes, the measured effect
will be **C++ scalar-scratch array capacity under one
serialized source event order**. It does not establish
physical FPGA BRAM, port legality, clocked schedule,
throughput, bit-exact HLS or scientific firstness.
Only an original native HLS / strict clean-DRC
physical matched study and externally pinned
feature-equivalent peer comparison can address
those remaining paper gates.
Full source protocol:
[R2H_ACTUAL_ALLOCATED_FANOUT_SOURCE_PROTOCOL.md](R2H_ACTUAL_ALLOCATED_FANOUT_SOURCE_PROTOCOL.md).

# FPGAI CST Model IR — scientific sprint roadmap

**Date: 2026-11-02.** Scope: compiler research in fpgai_research/main only. RL/Gym, the frozen L0 study and previous native receipts remain untouched. This is a falsifiable research program, not a feature checklist or a promise of novelty.

## Primary research question

Can coupling operator-local Phi (spatial/temporal execution) with first-class tensor-element availability, producer/consumer legality and physical scratch lifetimes provide independently verified executable architecture choices beyond the operator-only FPGAI Model IR? Do these mechanisms remain distinct from the actual capabilities of Allo, HIDA, StreamTensor and ST-Flow?

## Sprint backlog and evidence gates

| Sprint | Implementation and scientific experiment | Acceptance criteria | Status |
| --- | --- | --- | --- |
| S0 — Evidence baseline | Preserve prior CPU/GCC and control experiments and exact source SHA | Prior adb28108 CI: 844 passed, 2 skipped; 30/30 legal controls, 138/138 designed invalid schedules, 22 selected numeric source checks across campaigns. Preserve raw CSV and run URLs. | Complete at prior pinned SHA |
| S1 — Independent legality oracle | A separate constructive oracle enumerates ALL valid finite schedules; run a differential checker study over all permutations, not selected mutants. | Five independent small domains, 3,600 complete six-event schedules. Require zero false accepts and false rejects; save all rows and exact source SHA. | Implemented on main, CI not yet verified |
| S2 — Checker scalability | Sweep graph depths, row counts and row-block widths; record IR resolution, finite liveness checking and schedule-checking wall times independently. | 15 graphs, 2–4 operators, 2–512 rows, all supported B row blocks, three raw timing samples each, all success/failure statuses. CPU cost only, not HLS performance. | Implemented on main, CI not yet verified |
| S3 — Real trained network semantics | Bridge the existing provenance-checked MNIST MLP model and input assets to CST semantics without dropping ReLU or changing weights. Add CNN/branching only after typed legal exchange, correct lowering and numerical oracle. | At least one actual trained model, source/model/input hashes, independent ONNXRuntime/PyTorch outputs; honest unsupported cases for unimplemented operators. | Outstanding |
| S4 — Original Vitis campaigns | Run two already-prepared native source sets on a separate licensed Vitis machine: 12 whole/column/row and 6 B=1,2,M. | Source/clock/part/numeric matched original CSim and CSynth reports, fail/unsupported cases, cycles/resources/estimated timing, no physical timing claim from target clock. | Prepared only; zero Vitis calls |
| S5 — Causal ablations | Hold graph/Phi/precision constant; compare operator-only IR, typed edges, unchecked temporal scheduling, and certified CST IR; avoid comparing unrelated models. | Legal design coverage, first counterexamples, actual compiled source differences, negative controls, numerical error and CPU verification overhead. | Scoped operator-only ablation done; S1/S2 extend it |
| S6 — Published comparator parity | Inspect and when feasible run pinned Allo, ST-Flow and StreamTensor artifacts on matching supported semantics. | Exact feature/rule matrix and compatible native runs; inaccessible or unsupported cases are unknown rather than FPGAI wins. | Outstanding |
| S7 — Paper evidence completion | Create publication figures/tables only from source-pinned original artifacts, full audit, release/IP review. | E0/E1 complete, E2/E3 for any hardware claim, direct novelty comparison, transparent limitations. | Not paper-ready |

## Research hypothesis → experiment map

**H1 (representational choice):** Same Functional graph and effective operator Phi, distinct B=1/2/M storage schedules, actual changed HLS C++, equivalent numeric results. Do not count differing hashes alone.

**H2 (independent legality):** Correct classification of all candidate schedules on bounded domains against an independently constructed oracle; plus five preregistered invalid intervention classes. Reject any false acceptance, record counterexample and full denominator.

**H3 (lowering fidelity):** Generated C++ executes with original Phi iteration assignments and exact expected producer-ready/consumer-finished events; reference numeric acceptance on all selected cases. Host float is not bit-exact ap_fixed.

**H4 (HLS trade-offs):** Compare source-matched cycles, estimated clock and LUT/FF/DSP/BRAM using original Vitis CSim+CSynth. Source buffer reductions are not hardware BRAM reductions.

**H5 (novelty versus research):** Demonstrate with actual related-work artifacts what the joint Phi/element lifetime certificate and legal lowering offer beyond existing systems. If equivalent prior art exists, narrow the contribution.

## Planned figures and tables

F1: operator Phi + tensor-element contract and actual IR example. F2: different legal schedules and live-value occupancy. F3: designed faults plus exhaustive independent-oracle confusion matrix. F4: CPU checker scaling with raw timing distributions. F5: native Vitis Pareto (latency/BRAM/DSP/LUT), **empty until executed**. T1: numerical/operator/graph coverage and unsupported cases. T2: external compiler semantic parity by pinned artifact. T3: component ablation, generated source and diagnostics.

## Non-negotiable scientific controls

Prespecify model weights, graph topology, precision, sample selection, exact source SHA, selected Vitis version/part/clock, numerical tolerance and finite verification limits. Treat small synthetic graphs and designed defects as limited-domain evidence, not general neural-model correctness or real-world bug recall. Archive all case-level input/results and failures. Do not reinterpret compiler regression counts as scientific contribution. Hardware experiments must use a dedicated native toolchain, never a running frozen RL checkout.

**Next execution gate:** Make S1/S2 GitHub CPU workflow green, archive source-pinned raw data, then implement S3 real-trained-model bridge while preserving full mathematical operators. Publication only after S4/S6 evidence supports claims.

## Verified sprint progress and next acceptance target

- **S1 complete (bounded domain):** 3,600/3,600 independent oracle classifications agree with the production checker, zero false acceptance/rejection in the five exact small domains; CI run 41367136765.
- **S2 complete (initial empirical sample):** 15 static chain graphs, 126 raw CPU checker samples up to 512 rows; archive source SHA fc3ab997. This does **not** establish universal scaling.
- **S3 initiated:** strict trained-MNIST ONNX admission audit implemented in experiments/cst_trained_model_preflight.py. It does not silently remove nonlinearities or invent private weights. The actual trained-MLP/CNN assets are *not in the public repository*; the private source folder and available ONNX dependencies are required on the workstation. Full CST model lowering for Flatten/Gemm/Relu/Conv/branching is still **not implemented**.
- **S4/S6 blocked by external evidence:** native licensed original-Vitis execution, and reproducible published-compiler artifact comparisons. Do not call paper finished until these are independently measured.

Next engineering work should follow the research question: define a **semantics-preserving non-MatMul graph composition law** with explicit read/write ownership and independently generated counterexamples, then implement ReLU/Gemm and branching/Conv only once shape/numeric semantics and actual HLS lowering can be verified. Synthetic model weights alone are not a trained-network benchmark.


## S3 nonlinear compiler implementation checkpoint (source committed, native evaluation outstanding)

The restricted sequential CST row-chain compiler now has a genuine source implementation for **MatMul, Gemm (including transB=1 and optional constant bias), Relu and leading Flatten(axis=1)**. Every network operator remains present in the typed Functional/Architectural/Lowered IR and emitted source; unsupported branching, arbitrary Conv, mixed numerical formats and unimplemented unary spatial mappings continue to fail closed. For B=1,2,M row-blocks the generated source uses actual distinct scratch allocation/indexing. The new tests compare source execution against a separate NumPy reference and actual row-ready/consumed traces for odd row tails and both Gemm weight orientations.

- E1 *synthetic* nonlinear factorial study: experiments/cst_nonlinear_campaign.py, **16 prespecified configurations** of 2/3/5 rows, two Gemm weight-orientation modes and distinct B. The script preserves numerical checks, first source SHA, operator sequence and trace evidence. These are deterministic synthetic weights, **not trained MNIST**.
- Actual *trained* MNIST MLP study: experiments/cst_trained_mlp_host.py requires original verified ONNX/inputs/labels/metadata, preserves Flatten/Gemm/Relu/Gemm and compares original ONNX Runtime with Functional IR plus GCC generated C++; never manufactures weights or substitutes a CNN.
- Original *trained* native CSim package: experiments/cst_trained_mlp_native_prepare.py prepares create-only original-Vitis source/testbench/Tcl from the same assets and retains source provenance. **No Vitis native call is made by this preparation**. Batch-one (M=1) has no row-scratch reduction, and must not be used to claim such an improvement.
- Tests and CI: tests/test_cst_nonlinear_row_chain.py, tests/test_cst_nonlinear_campaign.py, tests/test_cst_trained_mlp_host.py and tests/test_cst_trained_mlp_native_prepare.py. **All latest CI outcomes for this new nonlinear code must be checked before marking S3 complete.**

S3 is **implemented for a sequential static MLP-shaped subset**, not for arbitrary branched neural networks, Conv2D, dynamic shapes, or a verified original trained-model run. Those remain separate scope gates; S4/S6 require original Vitis and independent prior-art artifacts.


## S3 acceptance checkpoint — actual CI results

Sequential nonlinear CST source lowering now supports the complete static **Flatten(axis=1) → Gemm(optional transposed constant weights + bias) → Relu → Gemm** MLP pattern with real HLS C++ loops and explicit per-intermediate B-row scratch. The actual strict ONNX frontend, NumPy reference, ONNX Runtime and GCC host C++ agree for three synthetic ONNX cases; controlled nonlinear source sweep yields 16/16 generated B/orientation configurations, 6/6 selected GCC numerical checks and 3/3 row traces.

**CI at immutable b563ba9c:** 893 passed / 5 skipped full suite, 128 passing focused CST tests, three separate ONNX checks. [Evidence ledger](REPRODUCIBILITY_LEDGER.md) contains source and run URLs.

**S3 is not fully closed scientifically:** real original trained-MNIST weights/input files were not available to execute `cst_trained_mlp_host`; its source and future original-Vitis preparation are complete and fail-closed. Branching and Conv2D CST row-lifetime compilation remain missing; batch-one trained MLP is a correctness test rather than a row-tiling memory-reuse benchmark.

**Next critical priority is not additional small synthetic tests:** run actual original trained-model E1 with verified source hashes, then original Vitis CSim/CSynth and a genuine ST-Flow/Allo/StreamTensor artifact comparison. Do not finalize manuscript claims while H4/H5 are unresolved.

## Publication-critical novelty decision (2026-11-02)

The authoritative [novelty thesis](NOVELTY_THESIS_AND_PROOF_OBLIGATIONS.md) supersedes any assumption that B-row scratch arrays, multi-stage IR, typed layouts or local Phi checks are the paper's novel contribution. The new **primary acceptance gate** is semantic ‘Phi -> per-value ready/reads/last-use/physical-address’ derivation and independent source-refined certificates, including incomplete reductions and fan-out. Current Phi hashes and atomic row-stage checks are *insufficient*. Implement this gate before asserting scientific novelty. Novelty relative to Allo/HIDA/StreamTensor/Dato/ST-Flow/Piezo/Wavelet must be tested rather than claimed; original Vitis QoR is separately unmeasured.

## Additional publication-critical sprint: certified RL/search interface

**S9 — learning/search-friendly IR (not an RL agent sprint).** Design a canonical, immutable, variable-graph observation and typed architecture-edit catalog derived from the **same** Phi-to-tensor-value lifetime certificate as compiler lowering. `transition(state, action)` must deterministically accept a legal new state with proof/evidence or reject it with a concrete counterexample; `valid_actions` must be sound within explicit verifier bounds, with UNKNOWN never treated as legal. Include source provenance, graph/device constraints, no HLS QoR leakage, graph permutation tests and policy-independent random/greedy/exhaustive action-list comparisons. **Use the same transition path for manual, heuristic and later RL agents.**

**Scientific evidence gate:** R1–R4 in [RL_READY_IR_NOVELTY_AND_EXPERIMENTS.md](RL_READY_IR_NOVELTY_AND_EXPERIMENTS.md) must show correct legal action masks and cross-graph representation invariants. The learning interface must actually depend on mapped Phi and full edge lifetimes; attaching a GNN vector or Gym wrapper alone fails the thesis. R5 (trained shared GNN policy and source-attested original-HLS sample-efficiency comparison) is a **separate deferred RL study**. Existing RL files, training setup and frozen source-pinned L0 remain untouched.

## Certified RL-ready compiler sprint series — newly scheduled implementation

The full L0–L9 scientific dependency/acceptance matrix is in [CERTIFIED_DECISION_IR_SPRINTS.md](CERTIFIED_DECISION_IR_SPRINTS.md). First software tranche L1–L3 includes typed immutable complete-design states, a single actual compiler legality transition for manual/heuristic/future RL requests, exact finite candidate masks, public weight-free variable-size graph observations and a source-pinned independent *restricted-domain* action experiment. `cst_decision_campaign` is wired into both one-command CPU profiles and the scientific test workflow. **Do not mark these experiments empirically passed until the new source revision's CI has finished.**

Publication-critical remaining work: L4 true `Phi -> per-element production/last-contribution/read` and source/refinement proof, L5 multi-reader lifetime and constrained physical bank allocation, L6 complete partial-design formalism, L7 full causal ablations, L8 source-qualified original Vitis and prior-art artifact parity. RL policy training is explicitly reserved for L9 and `rl_environment/` remains frozen.

## L5 constrained completion checkpoint and subsequent dependency plan (2026-11-02)

The existing L5a abstract multi-reader last-use contract has been extended in two independently bounded ways, with [23/23 quick and paper stages verified](https://github.com/umutcanaltin/fpgai_research/actions/runs/41429203318):

- **L5c restricted complete:** exact finite E0 address search/unknown/boundary classification; independent tiny-domain brute force controls. This is *not* HLS memory/bank optimality.
- **L5b restricted complete:** actual generated and GCC-executed five-operator MatMul/Relu/Relu/Add/Relu diamond, two readers, safe post-last-reader scratch reuse, six matched source/numerical/trace cases. No general DAG lowering or operator-local Phi source owner proof.

**Next scientific critical path:**
1. Align the E0 exact allocator's event schedule with the executed source's per-element interleaved order; verify any “optimal” claim under the *same* schedule and independent small exhaustive oracle.
2. Replace the diamond-only source special case with a typed graph-wide fan-out legality/refinement law that can carry independently resolved producer/consumer Phi ownership, port multiplicity, and source-extracted addresses. Fail closed on unsupported Add/broadcasting, banking and concurrent cycles.
3. Execute original native Vitis CSim/CSynth on a separate licensed, source-pinned compiler-only checkout; report failed cases and paired resources/cycles, not inferred physical BRAM.
4. Benchmark exact claimed invariants against actual pinned Allo, HIDA, ST-Flow, StreamTensor artifacts. Novelty remains falsifiable. Do not resume RL development as part of these compiler-paper sprints.

## New source-order-aligned allocation evidence

At [immutable revision 47390b43, source-pinned CI 41430232732](https://github.com/umutcanaltin/fpgai_research/actions/runs/41430232732), the exact finite E0 storage allocator has been connected to the restricted E1 diamond's **actual per-element source order**, with 24/24 quick and paper studies passed. The 3-shape matched allocation/source experiment includes six actual GCC numerical and trace checks, a separate tiny-domain exhaustive reference oracle, no original Vitis calls, and explicitly smaller emitted source scratch versus disjoint cases.

The former open task “align E0 allocation with actual source event order” is **closed for this exact five-operator source pattern only**. Subsequent sprints should generalize fully typed producer/consumer Phi and multi-reader reuse into the normal compiler IR and source backend, demonstrate branching operator legality and explicit bank/port constraints, and run original Vitis + peer-system artifact parity before asserting scientific novelty or physical performance benefits. RL remains frozen.

## S9/L5d research checkpoint — Phi-conditioned producer in source fan-out

The [original 25-stage quick/paper E0/E1 source-pinned research run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41432640743) verified a new first bounded linkage: MatMul producer `Phi` controls actual emitted shared-lowering C++ in a two-reader static diamond, with independent per-contribution GCC source tracing and source-ordered lifetime checks. Three qualified original host-float source cases retain mapping, source and event hashes; no original vendor/HLS tools ran.

**Next falsifiable sprint gates:**
1. Implement *two independently mapped MatMul/Gemm consumers* of one shared intermediate with per-input-element multiplicity, typed read-owner events, and last-use tracking across both consumers; no premature reuse, no silently scalarized architecture.
2. Extend the exact source-order allocator from scalar consumer acknowledgments to per-contribution mapped reader events; retain an independent finite offset oracle and mutation controls.
3. Add explicit abstract memory port/bank feasibility contracts only where declared; an unknown hardware mapping must not become a certified mask.
4. Run matched real original Vitis CSim/CSynth and separately compare pinned related-work implementations before presenting hardware or originality claims. Frozen RL stays untouched.

## L5e achieved (2026-11-02): real two-consumer, two independent MatMul Phi mappings

**Scientific target unchanged:** certificate-backed, compositional space-time tensor boundary IR, where per-operator `Phi` induces verifiable reductions, readiness, read-owner multiplicity, last-use and source-address lifetime. This must generalize to graph-wide composition and use the SAME legality relation in an optional learning-ready transition interface; policy training remains frozen. It is research into a falsifiable compiler-IR relation, **not** a general FPGA product feature or claim to invent buffering/HLS.

**Original source-pinned milestone:** `425772d6be0f26f036ff48564f5b6fb571b94186` [original CI run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41442961276) green **26/26 quick/paper**. Bounded L5e has original GCC source for producer + two independent MatMul reader mappings, exact per-contribution read-owner/last-use checks, 24 numerical/trace passes and 12 safe source scratch-layout pairs. No original vendor synthesis or firstness established.

**Falsifiable priorities, ordered:**
1. **L5f** Exact source-order address optimizer with per-contribution multi-reader last-use and independent finite brute-force oracle; report `UNKNOWN_BUDGET` when exhausted, not assumed optimal.
2. **L6a** General graph edge certificates for varied fan-out, broadcast, Concat, convolution/attention where qualified. Source implementation must preserve operator-local `Phi` and produce *actual* per-element reads/commits; unsupported graphs fail closed.
3. **L6b** Explicit bank/port and concurrency resource constraints (or `UNKNOWN` for not yet checked). Couple legal source address plans to supported memory primitive assumptions only where measured/declared.
4. **L7** Four ablations: no boundary contract, hash-only Phi, consumer-ACK-only, and full Phi+per-contribution read/ready/last-use, with source-matched fixed graph/mapping and numerical pass gate. Measure false legality, certified valid plans, compiler overhead and source scratch extent; no circular oracle.
5. **L8** Source-attested original Vitis CSim/CSynth on equal-accuracy kernels, then separate QoR claims only if observed. Compare real pinned Allo, HIDA, ST-Flow, StreamTensor behaviors and overlapping scientific claims at artifact level before declaring novelty.
6. **L9** After a sound graph-wide transition/mask that uses exactly the same verifier, expose immutable RL-ready `observe/actions/transition` artifacts. Do not restart RL training under the compiler paper sprint.

Historical source-pinned online Vitis/Gym experiments are not modified; they remain separate from this compiler-IR publication path.

## 2026-11-02 29-stage scientific sprint acceptance and next dependency gates

The bounded **L5f exact Phi/multi-contribution scratch source-array planner**, **L7 internal obligation ablation**, and **L6b fail-closed abstract bank-port UNKNOWN** are implemented, tested and source-pinned at `a8e03eafd68262c9414e6e9d579c33ac163c67d2`; original [29/29 quick and paper run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41446968133), raw [artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41446968133/artifacts/13143062150). Three finite source-layout optima, six original GCC paired checks, 7×4 controlled logical ablations, 3 UNKNOWN abstract-port controls and zero claimed hardware qualifiers. No original Vitis calls.

The **research thesis has NOT changed**: source-checkable, independently mapped per-operator space/time `Phi` combined with *first-class* tensor reduction completion, availability, per-contribution read ownership, last-use, and only justified logical address legality; eventually an RL-ready decision interface using the **same** obligation checker. Source-level evidence and proof obligations must precede any learned policy. A mere metadata hash, PE/SIMD mapping, buffer reduction, or IR name is not novel.

**Next critical-path, still OPEN:**
1. **L6a generalized graph IR/refinement:** multiple operator types beyond fixed MatMul diamond, arbitrary graph fan-out, precise broadcast/Concat/Conv read maps, legal layout conversions, graph permutations and compositional edge proofs. No silent scalar lowering of unsupported `Phi`.
2. **L6b hardware qualification:** map actual inferred memory primitives/ports/banks and source-to-original-Vitis schedule/RTL evidence; continue returning UNKNOWN when inferred cycles/port counts are absent. Current declared abstract example is NOT a hardware certificate.
3. **L7 paper evaluation:** external peer-artifact reproduction, stronger source-matched causal ablations, compiler overhead/memory comparisons, independent machine oracle and statistical/holdout graph evaluations; current internal mutation matrix alone cannot establish novelty.
4. **L8 native:** run original fixed-point CSim and CSynth on source-pinned trained networks with exact model/input/clock/part receipts and paired failures; no fabricated latencies/resources. [Peer artifact pin matrix](PRIOR_ART_PINNED_ARTIFACTS.md).
5. **L9 optional interface:** expose stable typed `observe/actions/transition/mask` grounded in full graph-time legality; independently test semantics and portability. The RL/Gym policy and seven historical online-HLS studies remain frozen/not edited.

**Stop-rule:** do not mark the paper publishable or the research complete until L6–L9 scientific acceptance gates and novelty audit are actually qualified.

## L6a verified intermediate checkpoint — compositional nested DAG source proof obligations

The bounded source-level compiler graph generalization is now verified at immutable code SHA `c1298ea35bf526b743f39c910615aa72f29afadf`, [original quick/paper 30/30 run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41450443712) and [full CPU regression 1,033 passed](https://github.com/umutcanaltin/fpgai_research/actions/runs/41450443849). A **first-class per-edge mapped tensor-value witness**, checked against original GCC C++ for six-, seven- and eight-operator non-diamond graphs, is implemented and source-pinned. The experimental row count is **12/12 numerical + 12/12 trace**, with 6 matched source-memory layout pairs. The source load address and the actual trace index are linked to the same emitted C++ variable. See [precise scientific contract](GRAPH_PHI_BOUNDARY_CERTIFICATES.md).

**Scientific contribution is STILL a falsifiable candidate:** one typed representation connects operator-local Phi to all needed consumer access-map events, producer readiness, final last-reader release, source scratch address and separately checked source execution. We must still demonstrate a nontrivial benefit/expressiveness gap relative to actual Allo, HIDA, ST-Flow, StreamTensor, Wavelet artifacts before making an originality or superiority claim.

**Next work, ordered:** (i) extend graph operator access maps and emitters to Conv/Concat/eligible broadcasting and mixed widths/shapes with precise numerical policies; (ii) independent graph-level finite oracle and causal graph-transfer experiments; (iii) qualified bank/port/clock legality, never turn an E0 abstract port feasibility into original HLS evidence; (iv) original source-matched Vitis fixed-point CSim/CSynth and trained-model assets; (v) executed pinned peer-system artifact parity and manuscripts; (vi) eventually integrate exact certificate-backed action masks in a weight-free RL-ready representation, while actual RL policy training stays frozen. Never collapse the five-layer product compiler architecture into this academic research subset.

## 32-stage L6a and L7 falsifiable checkpoint (2026-11-02)

[Original complete quick/paper evidence 32/32](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625112) at source `dbf6aade8228cbbe703037fb5108715cd0dd64f9` plus [1,048 CPU tests passing](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625106) qualified two new tranches: true nonidentity **Add broadcasting/Concat per-output-to-source access maps** with 16 original GCC numerical/trace cases, and an **independent NumPy semantic oracle** with 4 positive/4 corrupted mapping cases. Full source-pinned [artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625112/artifacts/13146491650). **Zero original Vitis calls, zero qualified hardware RAM/clock claims.** The controlled correlated-bug experiment shows why a producer and verifier derived from one helper cannot alone validate the source's intended semantics.

**Next work in scientific order:** (1) Extend per-operand graph source contracts to Conv2D and other realistic operators, including padding/stride/multiplicity and independently mapped consumer Phi, without silently weakening source-level checks. (2) Develop independently implemented finite event/time and topological graph reference oracles, typed compositional proof obligations and mutation families beyond shared-map bugs. (3) Original Vitis CSim/CSynth, source-attested trained network evidence, actual RAM ports/clock timing, and separate native QoR. (4) Executed pinned Allo/HIDA/ST-Flow/StreamTensor/Wavelet comparison experiments and priority/novelty reconciliation. (5) Only then widen and qualify RL-ready transitions/masks using the **same** tested legality relation. Policy training stays frozen. [Technical reference](NONIDENTITY_GRAPH_ACCESS_AND_ORACLE.md).

The **paper is not finished** and no firstness/performance superiority has been established. These are concrete bounded scientific contributions to the proof-obligation story, not a claim that NumPy or dataflow memory reuse are original algorithms.

## New bounded Conv NCHW/stride/padding checkpoint, 2026-11-02

E0 and partial E1 source evidence is now verified at `6af10f9e2ec71810181b51560d227d8dba133303`, [original quick/paper 33/33 source-pinned CI](https://github.com/umutcanaltin/fpgai_research/actions/runs/41456356235) and [full CPU 1,061 passes](https://github.com/umutcanaltin/fpgai_research/actions/runs/41456356220): four source geometries × two independently chosen Conv mapping policies yield **8/8 GCC numerical** and **8/8 actual executed Conv kernel-load + owner trace** cases. Independent NumPy sliding-window indices accept all eight, early logical overwrites are rejected and source-lifetime reuse controls pass. **No original Vitis**. [Detailed acceptance and limitations](CONV2D_NCHW_BOUNDARY_PROOF.md).

**Highest priority next:** (1) integrate NCHW Conv source taps into general graph-wide per-edge witnesses and actual materialized source COMMIT/scalar READ traces, not merely a new diamond; (2) independent finite dynamic/reduction event oracle and arbitrary branch tests; (3) native Vitis exact CSim/CSynth/DDR/clock/bank/port tests with qualified source provenance; (4) executable matched prior art and trained network benchmarks; (5) full publishability/novelty audit and only then RL-ready action masks using same sound checker. Frozen learned RL study stays untouched. The 5-layer Morfics product research architecture and professional compiler remain separate.

## 34-stage NCHW generalized graph milestone — evidence and next gates

At `fcb1f621bb6c0eaadae8f2206e823f59bd2bf21c`, [original 34/34 quick/paper run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339211) and [full 1,070-pass CPU regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339307) now verify the **bounded general topological rank-4 NCHW typed edge contract** beyond the fixed five-op Conv diamond. Includes Conv2D, Relu, broadcast Add, channel Concat and 2×2 MaxPool with independent Conv Phi and per-input/reduction consumption. The independent NumPy oracle and actual GCC Conv-reader load-index audit are separate. Four graph/shape configurations ×2 mappings ×2 *logical* scratch layouts yielded **16/16 recorded E0, source numeric and partial E1** checks; eight real original GCC executions. No native Vitis. [Source-level scientific boundaries](GENERAL_NCHW_CST_EDGE_CERTIFICATES.md).

**NEXT SPRINT SCIENTIFIC PRIORITY:** move from E0 producer-major lifetime and partial Conv-load source attestation to a **single actual source-executed global event schedule** checking all scalar/pool/concat/Conv READs, all COMMITs and physically realized source scratch aliases. After that, qualify actual original Vitis fixed-point CSim/CSynth and inferred RAM/port/schedule, run matched trained-model and external prior-art artifacts, test generalization, and only then expose certified action masks for RL without resuming trained RL policy work now. No firstness/novelty or PPA comparison is yet established.

## 36-stage original GCC intermediate and actual scratch materialization checkpoint

At code SHA `319ab4e94a57019d0249dbe2254a925e748e4ace`, [36/36 original quick/paper](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917065) and [1,083-pass full regression](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917137) verify E1 generated-source internal tensor numerical values (8 original runs, 50 operator snapshots, 1,764 source tensor values) and **physical C++ local scratch alias materialization** (8 disjoint/reuse pairs, 16 real source variants, **32 original GCC runs** for paired numerical/Conv loads). Every pair shows smaller **local C++ scratch array** under conservative whole-operator interval reuse, min 8 elements and total pair differences 408. Source hashes and all-case evidence in [artifact](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917065/artifacts/13157444356). Original Vitis calls 0. Read [full scientific scope](ACTUAL_NCHW_INTERMEDIATE_AND_SCRATCH_EVIDENCE.md).

**NEXT highest-priority paper sprint:** prove the *whole* source-observed READ/COMMIT event stream for every NCHW operator including MaxPool's physical extra first tap, not just logical E0 and after-stage snapshots; then run actual original Vitis fixed-point CSim/CSynth on both storage variants to determine whether legal and whether BRAM/clock/resource changes actually occur. In parallel independently execute pinned peer artifacts and matched trained-model evidence; do not assert novelty, priority or QoR before verified comparisons. RL training frozen and independent from scientific Model IR work.

## L6a observed source stream and next original HLS qualification gate

The scientific source-level READ/WRITE milestone is verified: immutable `77fe5dfcd96ad81ff8d71f36fe57cb88a47411e4`, [37/37 original quick/paper](https://github.com/umutcanaltin/fpgai_research/actions/runs/41480551778), 12 GCC source executions, 3,424 reads/1,928 writes across six genuine disjoint/shared pairs; independent path-sensitive NumPy and source causal replay all pass. The native *preparation* milestone at `e5502be096bd4b8ab0cf15b82e2ad4c96d0eac84`, [38/38 quick/paper](https://github.com/umutcanaltin/fpgai_research/actions/runs/41481941961), makes 12 source/clock/part/numeric-matched Vitis package directories, **without** running Vitis.

**Next empirical dependency:** verify latest native execution/audit code CPU tests; run one **actual original Vitis** disjoint/shared smoke pair with the new explicit foreground gates, qualify both fixed-point CSim reports and CSynth XML, then progress across all six precommitted pairs. Treat invalid alias source, changed fixed-point accuracy, missing latency/clock/port/timing and failed cases as results, not omissions. Only an attested native result can support a matched HLS BRAM or latency comparison; independent physical Vivado place/route is needed for physical timing/BRAM claims. External paper-artifact parity and trained-model evaluations remain mandatory before novelty/superiority assertions. RL/Gym stays frozen. [Exact commands and guarantees](NCHW_SOURCE_ACCESS_AND_NATIVE_HLS_HANDOFF.md).

## New source-observed vendor memory-policy evidence — explicit pragma not a free optimization

Original user-reported Vitis HLS 2023.2 explicit BRAM_2P policy generated from `8814bd0005e8`: all 12/12 CSim/CSynth qualified, but sharing original C++ scratch **never decreased estimated BRAM** (5 equal, one increases by 5), and cycle direction varied (three improvements, two regressions, one tie). The initial *separate* source-automatic inference pilot from `8fe0ef31f6ac` qualifies **two HLS cases** at 0 inferred BRAM_18K/975 cycles each, versus 5 BRAM_18K and 1005/975 cycles with the explicit pragma. This indicates that blindly adding a BRAM pragma may worsen HLS estimates, NOT that auto always improves physical hardware. Original RTL place-and-route and the ten remaining auto cases are not yet completed.

Implement and qualify the cross-study 2×2 audit with source-byte equivalence modulo the pragma, exact source revision/documentation and full missing-case denominators; never post-select faster layouts or default memory directives from one source-only example. Next hardware priority is original 10 remaining auto Vitis CSim/CSynth and independent, source-pinned Vivado OOC implementation for matched selected cases. Academic code must keep vendor directives in the per-vendor HLS lowering layer, and research RL training stays frozen. [Reproducible source policy experiment](NCHW_SOURCE_ACCESS_AND_NATIVE_HLS_HANDOFF.md).


## 2026-11-02 implementation continuation — physical and novelty gates

Latest *workstation-verified* point: `ad748e8b`, **71/71 focused and
1182/1182 full CPU tests**. The user created a source-pinned `v6`
12-case physical study (six disjoint/shared pairs) with
**12 PREPARED_NOT_RUN and zero native Vivado calls**. This is not a
physical result. The v4/v5 histories remain immutable.

Three follow-up source components have been added on `main` but
**are not yet independently workstation-tested**:

- `experiments/cst_research_completion.py`: fully matched population
  checks, zero-tool read-only plan, bounded opt-in **foreground**
  native Vivado continuation (one original case per receipt), never
  reruns a classified or rejected case, stops at a first rejection,
  and create-only routed original/paired evidence export.
- `experiments/cst_paper_gate.py`: independently audits original
  Vitis E3 and Vivado OOC case/revision/source matching; separately
  counts strict DRC-clean E4, exploratory warning-disclosed pairs,
  unknowns and explicitly-unmet novelty, trained-model, peer-artifact
  and IP/release approval obligations. Its submission-ready flag is
  NEVER set by numerical HLS or physical metrics alone.
- `experiments/cst_phi_counterfactual_effects.py`: 24 original
  two-reader GCC study cases -> 18 fixed-functional/producer
  reader-specific Phi counterfactuals with no-op actions included.
  It distinguishes changes to owner hashes, source events, actual
  last-use and physical C++ scratch slots, and refuses to infer
  scientific firstness from internal source comparisons.

The full [execution contract](FINAL_RESEARCH_EXECUTION.md) is now
the preferred user-facing handoff. Do not confuse newly implemented
source/tests with original executed peer-tool results. S4/L8 original
external baseline reproduction, real trained-model evidence and the
central distinctive Phi-dependent lifetime certificate experiment
remain open.

## External artifact reproducibility plumbing (2026-11-02)

Added `cst_prior_art_environment`: read-only source checkout
verification for the already recorded exact pins of ST-Flow,
Allo, HIDA and Wavelet; StreamTensor remains without a
qualified executable pin. All outputs explicitly say
`peer_compiler_executed=false`,
`certificate_parity_established=false`, and
`matched_hardware_qor_established=false`.
The code **is not** the missing S6 comparison: actual
source adapters, trained inputs, matched numerics, execution
logs, legality diagnostics, and native version alignment
remain separately required. Never relabel a pinned Git
revision as a reproduced benchmark.


## 2026-11-03 CST C1/C2/C3 source experiment integration: ready for workstation gate

Scope is **research compiler CST Model IR only**. The new canonical
C1/C2/C3 contribution contract is
`PAPER_CONTRIBUTIONS_AND_RL_INTERFACE.md`.
C1 operator-local Phi-conditioned tensor-element production/read
and release, C2 source/certificate proof obligations and independent
oracles, C3 a policy-independent RL-compatible graph/action/mask
interface. No RL training or separate RL environment changes.

Implemented and pushed to main:
- one-command CPU quick/paper series now includes
  `rl_compatible_ir_r3_r4` after `decision_interface` and
  `two_phi_counterfactual_e1` after the original
  `two_phi_readers_e1` real GCC stage;
- source-attested secondary Phi effects retain all 24 original GCC
  cases and all 18 controlled edits with ineffective actions
  counted as negative results;
- the original evidence index verifies each stage checksum,
  validates exact causal original GCC case CSV SHA across stages,
  and reports C1/C2/C3 SOURCE protocol statuses separately
  from unproven novelty, E4 and RL learning improvements;
- manuscript methods, fixed-primary contribution section,
  scientific protocol, publication gates and an exact complete
  create-only launch runbook were updated.

Latest **actually workstation-tested** baseline at
`98ee6097...`: focused **110/110**, full
**1221/1221** Python tests. Original v6 Vivado
`case_000` = warning-disclosed valid routed observation
and **not strict clean-DRC E4**; eleven cases unrun.
New integration/code is not locally verified yet.
Do not use historical v5 WNS/DRC counts as v6 data.

**Next dependency:** test new code, run the full original
source-pinned CPU/GCC paper series via
`bash experiments/compiler_ir/run_all.sh --profile paper
--skip-onnx --output build/cst_c123_paper_20261103` in a
clean checkout, inspect source index and negative Phi results.
Then independently execute only `v6/case_001` with
explicit Vivado confirmation, audit the matched pair and continue
the exact 12-case physical cohort only under verified gates.
Original external peer tools and trained network parity still
remain separate unresolved scientific obligations. RL R5 is
optional future work and cannot be claimed in this paper.

## Original physical and Phi evidence closure — 2026-11-03

**Executed, original and independently audited:**
- Full compiler paper CPU/GCC run 40/40 at source
  `1cefdd8566658cb6b69e12444c9b5ce9ba6e4a31`;
- Full local regression 1,240/1,240 passing on
  the latest pre-update main;
- Original Vitis E3 12/12 and six paired;
- Original routed Vivado v6 warning-only physical
  observations 12/12, matched exploratory pairs
  6/6, STRICT E4 0;
- Physical BRAM differences `[0,+8,0,0,0,-2.5]`;
  RL-friendly state/certificate interface tested;
- **R2 strong last-use/capacity causality = 0/18**.

**NEW next primary research sprint: R2b mapping-caused real
source scheduler and independent oracle**, as
prespecified in
`CST_R2_NEGATIVE_FINDING_AND_NEXT_FALSIFICATION.md`.
First implement a real C++ source source-event
interleaving of producer, both reader `Phi`
schedules and physical scratch last-reader release,
with an independent numerical and event oracle.
Then compare to the unchanged serial-source
baseline with fixed Functional graph and weights,
as a new create-only separately versioned original
study. Nonzero mapping effects must be derived
from **actual emitted/GCC-executed source read
order and address reuse**, not changed Phi labels,
hashes or manually selected schedules.
An actual power/timing gain or novelty relative to
external prior art is NOT assumed.

External comparison, model reproducibility and
publication permissions are still mandatory; do
not call the compiler research publication ready.

### R2b first prospective source-order implementation checkpoint

**New code only; not yet workstation-verified:**
`model_ir/interleaved_branch_schedule.py`,
`experiments/cst_r2b_interleaved_schedule.py`, and
`tests/test_cst_r2b_interleaved_schedule.py`.
Three fixed diamond shapes × four independent reader
PE policies define 12 bounded E0 alternative issue
cases, with source-*proposed* last-value-use indices,
counterexample checks and a frozen complete-denominator
new-study CLI. The original serial E1 negative
18/18 study remains unchanged. These new E0
events are **not emitted GCC C++**; next sprint is
instrumented and independently replayed E1 actual
source, a separate NumPy oracle, and original
source-address safety audit. Only after E1 should
R2b physical/HLS experiments be considered.
No RL algorithm/training or product repos changed.


## 2026-11-03 new R2b E1 generated-source implementation checkpoint

User's most recent original E0 R2b run:
**1245 tests passed**, 12/12 logical verification,
9/9 *global event-index* changes for independently
proposed Phi reader edits. No GCC/native tool
calls in this specific E0 experiment.

Implemented **but NOT yet tested on the user's PC**:
- `codegen/hls/r2b_interleaved_branch.py`
  emits actual source scratch loads/stores under
  a complete prospective Phi-conditioned
  two-reader PE-group issue schedule;
- `validation/r2b_interleaved_cpp.py`
  compiles and executes each original emitted
  source with GCC, parses all typed source
  READ/COMMIT events, checks conservative
  branch memory certificate and independent
  NumPy output;
- `experiments/cst_r2b_source_campaign.py`
  specifies 3×4×2 = 24 new original GCC
  cases with source/input/trace receipts and
  negative controls, create-only;
- `experiments/cst_r2b_source_audit.py`
  checks all original saved source/trace hashes
  and recomputes H-only final reader source
  rank and last consumer identity, read-only;
- `tests/test_cst_r2b_source_campaign.py`
  tests actual GCC numerical correctness,
  source trace mutation, false-read operand
  corruption, early alias rejection, and the
  complete 24-case source campaign.

**Important:** A changed global event index is
not sufficient if all H-only READ ranks and
final reader identities tie. Read-only audit
records this difference. This is serialized
host source E1, not physical PE architecture,
RTL or mapping-conditional bank capacity.
The old serial E1 causal result 0/18 remains
an intact negative experiment. The final
CST firstness thesis remains conditional on
external published-artifact parity and general
compiler semantics; no trained RL method was
run or claimed.

See [original R2b E1 execution protocol](R2B_E1_ORIGINAL_GCC_PROTOCOL.md).


## 2026-11-03 original E1 R2b result: executed source read-order milestone

The new prospective E1 generated-C++ and independent
NumPy source verifier are no longer merely planned:
original workstation **8 R2b tests + 1253 full tests
passed**, original source-pinned 24-case GCC run
completed **24/24** numerical/trace qualifications,
and the first read-only original case/trace auditor
reported **24/24** valid. This is E1 host source
evidence, not real PE/HLS hardware scheduling.

Original 18 layout-scoped controlled Phi comparisons:
global event changes 18, actual H-only last-read
rank/reader changes 12, final consumer identity changes
6, with 6 true H-only ties; there are **NINE unique
graph/Phi interventions** after collapsing the two
storage layouts; 6/9 H-only effects and 3/9
reader-identity changes. Source scratch allocation
does not improve as a function of Phi in these
cases. The original serial-source 0/18 negative
study and v6 non-clean-DRC physical study remain
separate unchanged populations.

New, untested code adds read-only
`cst_r2b_causal_audit` and
`test_cst_r2b_causal_audit.py` to
independently replay the original graph,
input, source stdout and nine unique
intervention groups without rerunning GCC.
**Next real scientific feature required:**
compare checked per-value source scheduling
under independent admissible source memory
allocation choices, or advance to broader
multi-operator graph generality, then
original HLS CSim/CSynth/physical parity
and external published-artifact comparisons.
Do not infer firstness, total hardware
memory savings or trained RL advantage
from source read order alone.


## New R2c: from source READ order to optimal safe per-element source placement

**Workstation-verified prior:** previous repaired
R2b causal audit 5 focused and 1258 full tests
passed, preserving the original R2b 24-GCC-case
6-of-9 unique Phi read-rank result.

**New committed R2c sprint deliverables (not yet
executed or qualified locally):**

- Per-element lifetime interval extraction from
  exact `COMMIT` to final `READ`,
  physical source address safety replay and
  deterministic earliest-release slot coloring;
- Independent finite brute-force slot-coloring
  oracle against small graphs; separate max-live
  lower bound for large fixed event sequences;
- Real C++ per-element scratch address accesses
  bound to adjacent runtime `FPGAI_SLOT`
  witnesses; independent actual GCC and NumPy
  verification (E1 only after it is run);
- Source-pinned 12 matched graph/Phi
  case pairs, each executing old last-reader
  whole-tensor reuse control versus new
  source-level optimal per-value allocation
  under the SAME emitted Phi-conditioned
  total source event sequence;
- Source array footprint includes the scratch
  allocation plus 4*m*q other declared mutable
  array elements. Scalar C++ locals and real
  FPGA/RTL/BRAM memory are not part of
  this count;
- Independent, read-only 12-pair rerun-free
  original evidence auditor and tamper/alias/
  numeric/unknown-unsupported tests;
- E0/E1 and publication go/no-go
  remain separate until the original new
  tests/executions report actual values.

**Falsification:** report all 12 scratch savings,
ties, regressions, and the actual count of nine
unique Phi edits that change **slot capacity**
versus merely address assignments. A changed
source-alloc digest with unchanged capacity
does not qualify mapping-conditioned memory
saving. The optimality theorem is the standard
interval-coloring property for a fixed total
source event order; do not claim new interval
coloring as scientific firstness.
Real HLS cycle/BRAM/port evidence remains
outside R2c and needs separate original tools.

Full protocol:
[R2C_EXACT_VALUE_LIFETIME_STORAGE_STUDY.md](R2C_EXACT_VALUE_LIFETIME_STORAGE_STUDY.md).

## F1 runtime-loop source witness milestone — implementation only

The C2 independent source oracle and 96-mutant E0
original campaign were user-run at `82ef9f2...`,
with **96/96** fault cases killed and zero E1/native calls.
Full regression had 1294 passing tests and one expected
error-message mismatch; a test-only correction has
since been pushed but not locally verified.

A new R2d loop-driven source compiler and its
GCC/NumPy independent-order auditor have now been
implemented separately from R2b/R2c. The dynamic
C++ ordinal-to-(row,col,kk) decoder performs
actual scratch READs and prints the runtime index,
rather than unrolling source event labels.
A corrupted but numerically complete loop order
is expected to be rejected by an independent
closed-form rank/owner oracle. New 12 original
case source study/read-only audit and fault tests
are committed **but unverified**.

See `R2D_RUNTIME_LOOP_INDEPENDENT_ISSUE_PROTOCOL.md`.
Next scientific gates: user regression then
new original R2d GCC evidence, general tensor
fan-out, independent producer contribution
certificates, native HLS port/timing and
exact prior-art comparators. Do not count
any result before original user execution.

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
