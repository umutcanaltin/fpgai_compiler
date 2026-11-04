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

## 2026-11-03 — R2o C3 multi-reader Phi action interface, original GCC study and read-only audit IMPLEMENTED; workstation evidence PENDING

**R2n original verified and frozen:** source revision `6296f99eb5d52371bffb9755989a340aadbe7ff6`; 37 focused and 1,464 full tests passed on the user's laptop; original R2n three MatMul chain depths × three actions = 9 outcomes, 3 E1-qualified, 3 rejected, 3 E0-unknown, 0 E1 failed. Independent original R2n source receipt audit agreed; no vendor calls and no trained RL. Do not rewrite R2n results using new code.

**New R2o prospective source and tests:** `src/fpgai_research/model_ir/fanout_decision_interface.py` defines a **new bounded sibling adapter** using the existing typed `FunctionalModel`, `ArchitectureRequest`, target limits and common `DecisionAction`, directly reusing **R2f multi-reader Phi/last-use E0** and **R2h real generated-source aliasing + GCC NumPy/address E1**. It does **not yet unify** sequential-chain and multi-reader source decision states. The only editable actions in the pilot are independently selected reader PE width and disjoint/per-element-optimal scratch layout, for exact 2–4 reader MatMul fanout + left Add chain + Relu. Unsupported actions reject and retain the original complete state. Source SHA/manifest/original-GCC-stdout conflicts and attempted overwrites are fatal. The action mask distinguishes source `REJECTED`, E0-admitted but GCC unrun `UNKNOWN_E1_NOT_RUN`, `FAILED_E1_SOURCE_EXECUTION` and actual source `E1_GCC_MULTI_READER_QUALIFIED`. It discloses that source E1 is **not** HLS/Vivado clock/port/BRAM qualification or learned RL evidence.

`cst_r2o_multi_reader_action_study.py` prepares **12 prespecified actions across 3 static fanout graphs** with original generated C++, emitter manifests, full GCC stdout and failure receipts in a **new create-only source-pinned study**; `cst_r2o_multi_reader_action_audit.py` independently checks raw artifact SHA/count/status integrity **without rerunning GCC**. Both quick and paper compiler-IR experiment profiles include the new stage; focused tests and original source pilot are newly committed but **NOT yet executed on the user's workstation**. No new results, pass rates, or original E1 campaign counts may be claimed before the local evidence is received.

[Read R2o complete source experiment protocol and commands](R2O_C3_MULTI_READER_DECISION_SOURCE_PROTOCOL_20261103.md). The original R2i/R2j/R2l/R2m physical resource/WNS evidence, original R2n source results and frozen `rl_environment/` were unchanged. No GitHub Actions/tokens were used. Full arbitrary DAG mapping legality, partial action design, bank port timing and true RL learning remain open.

## 2026-11-03 — R2n C3 original GCC E1-qualified action screen IMPLEMENTED, workstation test PENDING

The new source-only, policy-independent `source_executed_action_mask` (`src/fpgai_research/model_ir/decision_interface.py`) differentiates `REJECTED` typed actions, `UNKNOWN_E0_CERTIFICATE` source-admissible but unmapped actions, `FAILED_E1_SOURCE_EXECUTION` and actual independently checked `E1_GCC_SOURCE_QUALIFIED` actions. It calls the **real compiler transition once per action** and, for mapped E0-qualified bounded static row-chain proposals, checks actual generated source GCC value/refinement trace and host-float numerical output against independent reference for an explicit fixed input. It fails closed on source identity mismatch, imposes an E1 call budget, and preserves failure outcomes instead of silently certifying them. **No RL training, Vitis, Vivado or physical feasibility**.

The prospective `cst_r2n_e1_action_study` covers 3 chain depths × 3 action categories, preserving all 9 outcomes in new create-only `build/r2n_c3_e1_qualified_actions_v1/`; no results have yet been executed on the user's workstation, so **no success counts are claimed**. New focused tests cover real GCC, source provenance forgery, false E1 promotion, unknown/reject/failure separation and create-only outputs; they are **newly committed, not workstation verified**. The new stage is wired into quick/paper compiler-IR series, with no GitHub Actions. Full instructions and conservative scientific limitations: [R2n C3 protocol](R2N_C3_E1_QUALIFIED_DECISION_IR_20261103.md).

This is an incremental **secondary C3 implementation**. Complete source-backed independent-Phi multi-reader DAG action semantics, partial decisions, bank/port feasibility and trained RL benefit remain open. The six original routed physical pairs and frozen RL environment remain unchanged.

# CST Model IR: three scoped research contributions and RL compatibility

**Manuscript contribution contract — 2026-11-02.** This is the
research compiler paper, not a product specification. The three
contributions below are **candidate independently testable
contributions**, not established scientific firstness. Existing
Allo, ST-Flow, HIDA, StreamTensor, Wavelet, CompilerGym, RL4ReAl
and other accelerator/search systems have related facilities.

## C1 — Compositional Space–Time Model IR

For a static Functional operator DAG `G`, each operator carries its
own `Phi_v` mapping across requested spatial, temporal and
numeric architecture choices. The inter-operator typed contract
carries an explicit *source-value* identity and:
- every required producer reduction contribution before a value COMMIT;
- each reader's consumer-local access/index and where applicable
  the corresponding independent `Phi_reader` owner;
- the global last READ before RELEASE, including repeated reads
  and fan-out, and a physically emitted source scratch address/layout;
- source/version/evidence lineage, without interpreting an HLS
  requested period as achieved Fmax.

The principal comparison is **not** whether existing systems offer
PE/SIMD mapping, streaming, graph-level reuse, layout types or
dataflow scheduling. They do. Instead test whether the *particular
coupling* of independently selected Phi to complete-reduction,
multi-reader last-use and emitted source-address obligations is
operationally useful and distinct. A digest or different source
hash alone is insufficient.

**C1 accept/reject tests:** two independently mapped consumers of
one value; prove the final reader cannot be displaced by premature
scratch reuse; vary only one reader's Phi; independently inspect
actual source event ordering, accesses and numerics. A mapping
change that affects hashes but not legal value lifetimes counts as
an explicit **negative outcome**.

## C2 — Certificate-backed architectural legality

An independently checkable bounded certificate is generated before
a supported emission can be accepted. Its content should be
`(source tensor, element, reduction complete, COMMIT, Phi-owner
read events, global RELEASE, physical source address, numeric
contract, emitted source SHA, evidence level)`. The verifier
rejects the first concrete illegal reader/element/reduction/address
and distinguishes legal, rejected and unknown/bound-exceeded.

- **E0:** finite IR/event/certificate claims within the verified
  subset, not a hardware theorem.
- **E1:** actual GCC generated-source numerical and event/element
  trace checks where implemented.
- **E2/E3:** source-attributed original fixed-point Vitis CSim/HLS.
- **E4:** genuinely original fully routed, timing-met, DRC-clean
  Vivado design. Vendor-warning-only routed observations are a
  separately reported exploratory tier, **not E4**.

The same-compiler restricted ablation and independently replayed
published-peer artifact parity are distinct tests. A weak own-IR
control does not establish priority over outside compilers.

## C3 — Policy-independent, RL-compatible architecture interface

CST presents a **complete immutable, versioned state** with typed
graph/tensors, per-op requests, exchange choices, target limits,
compiler certificate status and source identity.
The same proposed architecture edit is evaluated by the same
`resolve -> compile_cst -> certificate` path irrespective of
whether a human, scripted optimizer, random search, greedy policy or
future RL agent proposed it.

```text
observation(S) -> variable-size operator/tensor/edge graph
action(S) = (scope, target, field, candidate value)
transition(S, action) -> Accepted(next_S, source-certificate)
                       | Rejected(original_S, concrete reason)
source_mask(S, finite proposals) -> SOURCE_ACCEPTED | REJECTED
certified_mask(S, finite proposals) -> BOUNDED_MAPPED_VALUE
                                       | REJECTED | UNKNOWN
```

For the conservative certified mask, **UNKNOWN is not a legal
action**. An accepted E0 source candidate is **not** a physically
feasible HLS candidate. A public graph observation contains no
raw trained-model tensor payload and does not leak unmeasured QoR.
The source compiler does not own the agent, reward, replay buffer,
training loop or search algorithm. RL suitability refers to a
*usable agent-facing representation*, not an RL performance gain.

### C3 experiments without training (appropriate for this paper)

- **R1 independent legality:** complete bounded action catalog and a
  separately specified restricted oracle; count all legal/rejected/
  unknown candidate actions, including unchanged and invalid edits.
  Prior `cst_decision_campaign` demonstrates a restricted subset.
- **R2 Phi causality:** same Functional graph and source memory
  proposal, change only a reader Phi, then test actual COMPLETE
  reduction, READY/READ/last-use consequences versus source-hash
  changes. This is not established merely by owner-projection hashes.
- **R3 representation robustness:** anonymous typed GNN observation
  under graph renaming and several graph depths; distinguish raw
  graph-source SHA (expected to change after renaming) from exposed
  anonymous features. This is NOT universal graph-isomorphism
  equivariance or transfer to unseen graphs.
- **R4 policy neutrality:** supply the SAME complete action via
  independently labelled external/human, random and heuristic
  proposers, and assert that the compiled next state, legal verdict
  and certificate are identical. Do NOT describe replay of the same
  action as three independently trained optimizers.

`python -m fpgai_research.experiments.cst_rl_interface_study
--output build/cst_rl_interface_r3_r4_v1` is an opt-in,
create-only, CPU-only **R3/R4 bounded test campaign** with
three 2-/3-/4-operator graphs; it records all cases, including
SOURCE-admissible but certificate-UNKNOWN whole-tensor edits.
The program/test suite was added **after** the user's last
1221-test workstation result and requires a new real workstation
verification before its measurements can be cited.

### R5 is an explicit separate empirical hypothesis

Do not claim improved RL convergence, rewards, sample efficiency,
hardware feasibility or generalization without trained agents
and fair, equal-budget original Vitis comparisons. A future
controlled R5 study should compare the SAME agent/GNN architecture
with (a) operator-only features, (b) operator plus tensor contracts,
and (c) these plus exact masks, using held-out networks, multiple
seeds, identical source/HLS budgets and explicit unqualified
failures. External search benchmarks such as CompilerGym,
RL4ReAl and HLS-specific RL methods remain prior art; the
claim that an IR is 'RL-ready' is **not a novel RL algorithm**.

## Latest original workstation evidence (2026-11-02)

At existing `main` commit `98ee6097aa74b9dd9b1f16a26bea43b650f33bc1`
the user executed:
- `compileall`, **110/110 focused** and **1221/1221 full CPU
  tests**, without reported failures;
- a read-only 12-case prepared v6 study (six exact paired
  source-storage studies, no Vivado calls);
- one original foreground Vivado 2023.2 `case_000` call,
  102 s, exit code zero; independent classification
  `OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS`,
  `routed_observation_valid=true`, `strict_e4=false`;
- original Vivado receipts = **1**, prepared cases = **11**,
  original strict E4 cases = **0**, matched routed pairs = **0**.

**The transcript does not include the original v6 WNS, resource
counts or individual DRC counts**; those must be obtained from
the separate read-only original-case auditor before numerical
physical claims. Historical v5 observations must never be passed
off as v6 evidence. The v6 DRC acceptance tier was informed by
v5 exploratory findings, so it is **not** an independent
pre-registered warning allowlist.

See [FINAL_RESEARCH_EXECUTION.md](FINAL_RESEARCH_EXECUTION.md),
[RL_READY_IR_NOVELTY_AND_EXPERIMENTS.md](RL_READY_IR_NOVELTY_AND_EXPERIMENTS.md),
[SCIENTIFIC_STUDY_PROTOCOL.md](SCIENTIFIC_STUDY_PROTOCOL.md) and
[PRIOR_ART_PINNED_ARTIFACTS.md](PRIOR_ART_PINNED_ARTIFACTS.md).

## Publication go/no-go

The title may describe this IR as **certificate-backed and
RL-compatible** once the finite interface experiments are independently
validated. The stronger phrase **'improves RL learning' is blocked**
until R5. Likewise broad 'first' and physical speedup claims are
blocked until direct published-artifact parity and full source-matched
physical outcomes exist. A negative result must be included rather
than redefined as a win.
