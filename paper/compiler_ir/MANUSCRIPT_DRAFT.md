> **ARCHIVAL RESEARCH DRAFT (4 November 2026):** This long rolling draft contains historically accurate prospective sections that are now superseded. The **active two-contribution C1/C2 compiler-paper manuscript** is [C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md](C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md); do not copy obsolete R2z/R3a prospective statements from this archive. Latest original results include R3d **3/2/1 of 6**, R3e **5/5 oracle and 25/25 selected mutants**, R3f **21/24/3 of 48**, and 1,705/1,705 full pytest (source pinned `fc40369f9493d93cc78c4886ae2e32c5d1f64093`). New R3g/R3h are unexecuted. [Frozen checkpoint](R3D_R3E_R3F_ORIGINAL_C1_C2_WORKSTATION_RESULTS_20261104.md). Remaining external scientific comparisons and proof limits mean **NOT SUBMISSION-READY**.

> **LATEST ORIGINAL EVIDENCE — 3 November 2026, superseding older 1/6 and 3/6 status notes below.** The author's workstation at code `6e87a4c42aaa30f6ad9e9be1ecccd7218c6ede99` passed **30 focused and 1,649/1,649 full pytest**. The original source cohort `f665b62ca81ee181a1447e3eddd23d7eba578186` has **6/6 original Vitis attempts, 6/6 numerically qualified CSim E2, 4/6 strictly E3-qualified CSynth**, and **2/3 originally prespecified complete matched E3 pairs**. **`case_002` and `case_003`** both failed their unchanged **9 ns effective HLS clock budget** with separate source-attested **9.181 ns** estimated clock reports; no E3 resource/latency comparison is valid for that pair. The two qualified independent-Phi interventions had **HLS BRAM18K Δ0 in both**, and *increased* worst HLS cycles: **624→911**, **2365→5436**; DSP, FF, LUT estimates also increased. All observations are **HLS**, not physical E4, proven general improvement, trained RL, or bit-exact all-input semantics. The new [R3c full-cohort publication result exporter/protocol](R3C_COMPLETE_CNN_HLS_PUBLICATION_RESULTS_20261103.md) is committed and **NOT yet workstation-tested**; it performs no vendor calls. Paper decision remains **MAJOR REVISION / NOT SUBMISSION-READY**; prior status notes below are historical snapshots.

> **Original workstation checkpoint — 3 November 2026 (later than earlier notes below).** On code `fb6747af88e09a68a763a85e66845042029c4967`, **45/45 focused and 1,633/1,633 full tests passed**. Original source cohort `f665b62ca81ee181a1447e3eddd23d7eba578186` remains 6/6 source packages SHA-attested and 12/12 GCC E1. Original native Vitis runs now cover **three** cases: `case_000` and `case_001` E2/E3-qualified; `case_002` E2-qualified but **FAILED_HLS_EFFECTIVE_CLOCK_BUDGET** at E3. The first matched `cnn_pool_dense` Phi contrast is qualified: estimated HLS BRAM18K Δ0, DSP +48, FF +1796, LUT +9146, worst cycles +287 (independent mapping vs scalar). This is **2/6 E3 and 1/3 complete two-arm HLS comparisons**; cases `003–005` unattempted. No new strict E4, Fmax or physical BRAM result. The new R3b narrow continuation past fully attested E3 clock failures and read-only rejected-clock timing diagnostics are implemented on `main` **but have NOT yet been tested on the workstation**; no new native run is claimed. The paper remains **MAJOR REVISION**, with novelty/baseline and physical evidence unresolved. [Original follow-up protocol](R3B_PUBLICATION_NATIVE_COMPLETION_AND_EVIDENCE_20261103.md).

> **Editorial evidence correction — 3 November 2026.** Original R2z source gate passed on pinned revision `f665b62ca81ee181a1447e3eddd23d7eba578186`: 1,626/1,626 full pytest; six source packages, 12/12 actual GCC E1 checks. The first actual R3a Vitis native case (`case_000`) also qualified E2/E3; HLS-estimated BRAM18K = 1. This is a **single native source case**, not a new controlled two-arm result. Remaining five cases and three HLS contrasts are UNKNOWN, with no R3a physical E4. The manuscript is **NOT submission-ready**; the main experimental text below refers to earlier, distinct original cohorts and must not merge their denominators with R3a. R3b follow-up code is newly committed, untested on the author's machine.

# FPGAI: Explicit Space–Time and Tensor-Lifetime Contracts for Compositional FPGA Neural-Network Compilation

**Scientific status (November 3, 2026): MAJOR REVISION.** This is a scientific working draft, not a demonstrated priority claim or physical FPGA optimization result. The external review identified major threats from the no-reuse allocation baseline, the committed-versus-in-place reader accumulator lifetime convention, weak HLS-to-physical causality, and prior-art overlap. The original source-pinned R2q accounting sensitivity experiment **has now been executed and read-only source-recomputed** at revision `7890c6f363cc9dd61c8bfa29ae121107ee3d86de` (17 focused, 1,516 full regression passes). Its complete 36-case results below confirm that the source capacity benefit depends on accounting assumptions. Historical sprint notes are preserved in [the verbatim archive](HISTORICAL_MANUSCRIPT_SPRINT_NOTES_20261103.md).


**Manuscript updated 2026-11-03; scientific review ongoing.** This work combines bounded space–time/tensor-lifetime IR semantics with independently audited source C++, Vitis HLS and six selected Vivado out-of-context (OOC) routed comparisons. The original R2h/R2i/R2j/R2l/R2m experiments are distinct from earlier MLP/CNN/vision studies and separate unexecuted HLS packages. All six routed pairs are warning-disclosed **descriptive observations**, **not strict clean-DRC E4**, achieved Fmax, board implementation or measured power. Original data and recorded source revisions remain frozen. See REPRODUCIBILITY_LEDGER.md, PRIOR_ART_NOVELTY_GATE.md and the final R2m paper export protocol.

## Abstract

Compositional FPGA neural-network compilation requires coordinating independently selected operator spatial–temporal mappings with the production, consumption, and safe reuse of intermediate tensor values. We investigate an executable intermediate representation (IR) that combines per-operator mapping provenance with explicit tensor-element availability, multi-reader lifetime obligations, and independently checkable source-address allocation certificates. The research compiler generates real C++ and performs bounded semantic checks for supported static MatMul fanout graphs. In a source-pinned 36-pair matched allocation experiment, all 72 original GCC programs executed with independent source-address and numerical checks, and all 36 optimized variants reduced source-level scratch **relative to a disjoint no-reuse baseline**; the subsequently verified R2q and R2s source-only studies supply tensor-first-fit and hypothetical in-place **accounting** controls, but no emitted fused in-place GCC/RTL comparator. Original Vitis HLS C simulation qualified all 72 cases; 48 of 72 cases (24 of 36 matched pairs) qualified for the stricter synthesis evidence tier. Among those 24 qualified pairs, six had lower HLS-estimated BRAM_18K and 18 tied. We then routed the six HLS-positive matched pairs using Vivado OOC implementation on the same target part. All twelve implementations completed routing and met their requested timing, but each disclosed DRC warnings, leaving **zero strict clean-DRC E4-qualified pairs**. Contrary to the HLS estimates, all six routed comparisons tied at **1.5 BRAM tiles** per arm. DSP allocation also tied in all six comparisons, while routed LUT, FF and WNS changes were mixed (mean optimized-minus-disjoint **+3 LUT**, **+0.167 FF**, **+0.335 ns WNS**). The results demonstrate bounded source semantic conformance and a source allocation difference versus no reuse, while exposing the failure of selected HLS-estimated memory reductions to translate to routed BRAM in the tested cohort. In the separately verified R2q and R2s source-only sensitivity studies, Phi-dependent differences disappeared under **hypothetical** first-MAC in-place accounting; that model is not the actual emitted R2h design.  A new source-pinned, global contiguous-tensor comparator proved 36/36 exact optima and improved upon greedy first-fit in 12/36 cases, while still showing 36/36 scalar element-level advantages under the chosen source memory convention. A controlled 32-arm native Vitis HLS factorial yielded 32 numerically qualified CSim executions and 30 E3-qualified synthesis estimates; two switch-lookup, automatically bound and exactly declared arms failed the effective clock budget. All R2u values are pre-route estimates and have no matching new Vivado implementation. They do not establish general FPGA QoR superiority, deployed performance or priority over existing compiler literature.

## 1. Introduction

FPGA compilation participates in designing the machine that executes a neural-network graph. The same MatMul operator may use one or many processing elements (PEs), different SIMD widths, distinct temporal reuse and tiling, and different pipeline initiation interval requests. The mathematics of the graph alone cannot determine where each logical iteration executes or how an intermediate value is physically stored and reused.

The original FPGAI progressive Model IR already distinguishes a **Functional Model IR**, **Architectural Model IR**, and **Lowered Model IR**. For an operator \(v\) with iteration domain \(\mathcal{I}_v\), its architectural map \(\Phi_v\) identifies requested and effective numerical/tile/spatial/temporal assignments. In the original array-backed compiler, each operator produces whole intermediate arrays, and subsequent operators consume those arrays in source order. The functional tensor identity is preserved, but intermediate **value lifetimes** are not first-class independently chosen architectural objects.

The present work asks a narrower question: can one retain independently selected operator-local mappings while **explicitly expressing and checking** when logical tensor elements become available, their concrete source-level addresses, and when a reused address is no longer live? This is a compiler correctness and representation question, not an automatic claim of improved Fmax, lower BRAM, or formal HLS equivalence.

Existing hardware compilers and mapping frameworks already make many of these ideas familiar. Timeloop, MAESTRO and AutoSA address spatial and temporal mapping; Allo composes hardware customizations; HIDA and StreamTensor represent hierarchical dataflow or streaming tensor layouts; ST-Flow (PACT 2026) already implements spatial/temporal/hybrid mapping and resource reuse on FPGA. Therefore the novelty claim considered here is not space–time mapping, hierarchical IR, tensor layout, or row buffering individually. The candidate distinctive contribution is the **specific architecture-to-element-lifetime relation with independent bounded checking and executable refinement in the FPGAI Model IR**. Whether this contribution is scientifically distinct enough from closely related compiler representations requires careful prior-art analysis and precise statement of the new contract; a direct execution benchmark against every existing compiler is not a prerequisite for claims limited to the independently checked IR semantics.

### 1.1 Three scoped, falsifiable contributions

**C1 — Compositional Space–Time Model IR (primary).**
For a static typed Functional DAG and independently selected per-operator
\(\Phi_v\) mapping, CST represents producer reduction completion,
consumer-specific input element access, multi-reader last-use and
actual source storage address/reuse obligations. These are bounded
source/event semantics, NOT a cycle-accurate hardware dataflow theorem.
An operator mapping or IR hash by itself does not prove a changed
lifetime; this is tested with source-executed controlled Phi
interventions and independent negative controls.

**C2 — Certificate-backed architectural legality (primary).**
A supported source scheduling/storage choice is checked against
bounded per-value event obligations, independently checkable
counterexamples, and generated C++ source/numerical traces.
CST distinguishes E0 logical legality, E1 original GCC executed
source, E2/E3 original fixed-point Vitis CSim/CSynth, and strict
E4 post-route timing plus clean DRC. Original warning-only
Vivado reports are a separate exploratory observation tier and are
never counted as strict E4. Full RTL/machine-checked correctness,
arbitrary operator/graph generality, physically optimal memory
allocation and performance superiority remain unproven.

**C3 — Policy-independent, RL-compatible decision interface (secondary).**
An immutable complete architecture state exposes a variable-size
operator/tensor graph, typed per-op mapping and exchange actions,
source-compiler-driven transitions, and a conservative three-valued
certified action mask that marks bounded verified actions, rejection,
and UNKNOWN separately. Equivalent complete actions must produce
the same source/certificate irrespective of who proposed them.
The compiler does not own any reward, RL agent or learned policy.
We test bounded R1–R4 interface correctness, Phi causality,
graph-renaming invariance and proposer independence. **No improved
RL learning performance or transfer generalization is claimed**
without a separate equal-budget multi-seed original-HLS R5 ablation.

Related-work analysis must determine whether this exact
joint `Phi -> source element readiness -> all readers -> release ->
real emitted source address` relation makes a defensible
contribution in light of Allo, HIDA, ST-Flow, StreamTensor, Wavelet and
compiler/RL interfaces such as CompilerGym or RL4ReAl. Each unexecuted
peer comparison remains UNKNOWN; restricted own-compiler ablation
alone cannot establish research firstness. A directly executed
external compiler benchmark is optional unless needed to support a
specific performance or capability claim.

## 2. Model IR and semantics

### 2.1 Functional Model IR

The functional model is a directed acyclic graph \(G=(V,E)\), where each operator has ordered input tensors, one named output tensor and static shapes. Each constant tensor has a stable immutable numeric payload. Functional validation enforces producer identity, topological dependencies, legal tensor shapes and unique producers. This is the baseline mathematical meaning shared by every architecture in a paired experiment.

### 2.2 Operator-local architectural mapping

For each operator \(v\), the Architectural Model IR binds its logical iteration domain \(\mathcal{I}_v\) to an operator-specific mapping

\[
\Phi_v(i)=(q_v(i),\kappa_v(i),r_v(i),p_v(i),\ell_v(i),f_v(i),g_v(i)).
\]

The terms respectively represent the numerical format, tile, coarse replica, processing element, SIMD lane, temporal fold/reuse step, and pipeline-related requested mechanism. The exact mapping values are source-level architectural intent, **not** measured HLS issue cycles, P&R placement or actual Fmax. Requests are rejected when unsupported rather than quietly substituted. Existing source-side and host-executed \(\Phi\) checks are reused in the extension.

### 2.3 Typed tensor exchange

For an intermediate tensor edge \(e=(u,v,t)\), we define the candidate compositional contract

\[
B_e = (t,\operatorname{prod}_e,\operatorname{cons}_e,
       A_e,\lambda_e,\mathcal{N}_e,P_e),
\]

where \(t\) is logical tensor identity and shape, the producer/consumer fields identify actual operators and their \(\Phi\) provenance, \(A_e\) is the concrete source-address relation, \(\lambda_e\) describes symbolic value-availability and storage lifetime, \(\mathcal{N}_e\) is the numerical compatibility requirement, and \(P_e\) identifies the supported exchange policy.

The current supported physical-source policies are **whole-tensor row-major**, **whole-tensor column-major** for nondegenerate 2D MatMul/Gemm-to-MatMul/Gemm exchanges, and a restricted **row-lifetime MatMul chain** policy. The current HLS emitter uses one common fixed-point type and one runtime model input/output. Mixed per-operator precision, arbitrary intermediate streams, FIFOs, DMA, and cast adapters are rejected or unsupported. A mismatch of PE or SIMD sizes alone does not require an adapter under the existing flat-array source contract.

### 2.4 Whole-tensor layout

For a static \(M\times N\) intermediate, logical coordinates \((r,c)\) can be represented as

\[
A_{\mathrm{row}}(r,c)=rN+c,\qquad
A_{\mathrm{col}}(r,c)=cM+r.
\]

Both are bijections onto \(\{0,\ldots,MN-1\}\). The compiler changes **both** producer writes and consumer reads. The boundary checker enumerates the 2D addresses under a finite bound; unsupported transformations fail before emission. This is a basic correctness/expressiveness control, not a novel mathematical layout technique.

### 2.5 Row-epoch lifetime and safe aliasing

For a restricted chain of static MatMul operators, let the intermediate edge \(e\) carry \(M\times N_e\) logical values. Under the whole-tensor program, the declared C++ intermediate storage contains \(MN_e\) elements. The row-lifetime contract instead uses a physical scratch array of length \(N_e\), with

\[
A_{e}(r,c)=c \quad \text{for the lifetime epoch }r.
\]

Addresses belonging to different logical rows are intentionally aliased **only because their live intervals do not overlap**. Each producer completes the reduction for all elements of row \(r\) before a downstream MatMul consumes that row. Before row \(r+1\) may overwrite the shared addresses, the consumer of row \(r\) must finish.

For \(L\) MatMul operators with internal widths \(N_1,\ldots,N_{L-1}\), the generated source declares

\[
S_{\mathrm{whole}}=M\sum_{e=1}^{L-1}N_e,\qquad
S_{\mathrm{row}}=\sum_{e=1}^{L-1}N_e.
\]

Consequently their **source-declaration ratio** is \(M\) for the supported program. This is a source-space observation, not measured FPGA BRAM, DSP, LUT or timing. Importantly, during the computation of one operator the input scratch may still be needed while output scratch is being populated. The independent certificate conservatively models this overlap; input lifetimes are not released before output writes are accounted for.

### 2.6 Bounded liveness certificate

The independent finite checker enumerates, for every row and edge address, producer-write and consumer-read events and maintains the set of live scratch values. It requires that each read reference a matching live produced row, that no write overwrite an address still holding a live value, that the typed scratch capacity match the edge geometry, and that all live values be released at the row boundary. In the supported chain the expected event count is

\[
N_{\mathrm{events}}=2M\sum_{e=1}^{L-1}N_e.
\]

Exceeding the declared verification budget produces **BOUND_EXCEEDED**, not a certificate. The checker records an event-stream digest and a conservative simultaneously-live value peak. It does **not** prove the generated C++ itself semantically equivalent to the graph, nor does it formally verify Vitis HLS or RTL. Instead, separate host execution checks test whether generated source exhibits matching temporal event order and \(\Phi\) iteration coordinates.

## 3. Compiler realization

### 3.1 Resolution and validation

Functional models and independent operator architecture requests are validated and resolved first. The compiler creates typed inter-operator exchange objects only for supported tensor relationships; a column-major exchange must be a nondegenerate 2D MatMul/Gemm intermediate and all its consumers must use the recognized left-hand operand semantics. The earlier row-lifetime path supports sequential MatMul chains and source-compatible row tiling. A **separate bounded R2f/R2h fanout path** supports a MatMul producer with two to four independently mapped MatMul readers followed by a fixed Add chain and Relu, including per-value last-reader source scratch aliasing. Neither path supports general arbitrary DAG composition, concurrent port semantics, hardware streaming or mixed-precision lowering.

### 3.2 Architecture-preserving lowering

The normal compiler entry point and default generated source remain unchanged. An opt-in path chooses whole tensor, alternate layout, row-lifetime two-operator, or row-chain mode. For alternate layout it changes producer and consumer physical address expressions while preserving logical element names and operator \(\Phi\). For row-chain mode it emits an outer row loop, followed by each operator's PE/SIMD/reuse-aware reduction nests and one static scratch array per intermediate tensor.

### 3.3 Multiple independent checkers

There are intentionally different evidence categories:

- **E0:** functional/architectural/lowered-IR contracts, per-op \(\Phi\), tensor address bijections, finite row-lifetime certificates, and explicit unsupported-case diagnostics.
- **E1:** generated C++ hash, successfully compiled host-float C++ numerical tests against a separate NumPy reference, actual executed per-iteration \(\Phi\) trace and row-ready/row-consumed event traces.
- **E2/E3:** original vendor Vitis fixed-point C simulation and HLS synthesis. The initial separately scoped twelve-source package below remains prospective; the later original **R2i 72-case fanout experiment** has independently audited 72/72 CSim E2 and 48/72 CSynth E3 qualifications.
- **E4+:** placement/routing timing, physical resources and live FPGA execution. The later original **R2j/R2l six-pair OOC study** has twelve warning-disclosed routed observations but **zero strict clean-DRC E4 cases**. Live-board/bitstream evidence is absent.

Host float is not equivalent to the vendor `ap_fixed` numerical implementation. The original R2i Vitis CSim testbench uses **absolute tolerance 0.1** with `ap_fixed<16,6,AP_RND,AP_SAT>` (2^−10 per fractional LSB); this is roughly **102 LSB**, a coarse numerical gate and **not arm-to-arm bit identity or proof of every fixed-point value**. A separate tighter, bit-level testbench is required for strong numerical equivalence, without retroactively changing original E2 evidence. An HLS cycle estimate is not a measured achieved clock. Runtime completion of a vendor tool is not by itself numerical or timing success.

## 4. Experimental method

The earlier chain/layout E0/E1 revision was `6282fcea6e87cf0dc84cad28f4479dec0404da63`; its historical regression figures are archived and not presented as current results. The 72-case native R2i cohort is pinned to `b2d5dcc4f18fad7cb1cd2c342101dd33185e817f`. The later original source C3 revisions are R2n `6296f99eb5d52371bffb9755989a340aadbe7ff6`, R2o `ce8083c0ad5b827232e1e67322811b25edd3ac79` and R2p `dac395598f2ab96bc9b7a41255b0529c72b22593`. All local tests require no GitHub Actions or tokens. Different source populations remain separately attested.

Three restricted same-compiler studies compare actual emitted source paths:

**A. Array-layout configurations.** The controlled campaign generates 96 cases across 24 distinct Functional graphs. Each graph has two effective architecture-mapping signatures; source generation includes matched row-major/column-major variants. The report contains 48 matched layout pairs, 96 distinct C++ hashes, eight successful host-numerical cases, and no Vitis execution.

**B. Two-operator row lifetime.** Eighteen pairs compare whole-tensor materialization against row-epoch scratch reuse with identical functional and local architecture choices. Four selected pairs are host-executed and all pass. The maximum reduction in declared intermediate C++ elements is fivefold for the sampled row count; this is not a synthesized resource result.

**C. Multiple operators.** Twelve generated 3-/4-MatMul chain cases contain eight three-operator and four four-operator graphs. All twelve have independent bounded lifetime witnesses; four selected cases pass host numerical comparison and two execute \(\Phi\)/row-epoch traces successfully. The maximum source scratch-declaration ratio in this study is threefold.

Across all three studies, 16 of 16 selected host-numerical qualifications passed. **Do not interpret this selected subset as a statistical generalization to arbitrary FPGA networks**, unsupported operators, or all architecture combinations.

The CPU study artifacts are kept separate from old MLP/CNN/vision results in earlier work; those earlier Vitis studies cannot be substituted for native validation of this new CST transformation.

### 4.1 Prepared native HLS comparison

A source-pinned twelve-case package is available for a prospective original-Vitis campaign: four groups, each with matched whole-tensor, alternate-layout and row-lifetime HLS source/testbench configurations on the same graph and architecture choices. The target in the prepared protocol is AMD Kria KV260, part xck26-sfvc784-2LV-c; the requested period and uncertainty are explicit in the package. No original-Vitis cases have yet been run in this new experiment. The read-only scorecard correctly reports zero qualified native pairs and eight unmeasured comparisons. That initial package remains separate from the subsequently completed R2i native 72-case and R2j/R2l OOC route experiments. No source or evidence should be substituted between cohorts.

## 5. Results

### 5.1 Independent R2h/R2i/R2l completed matched fanout experiment (2026-11-03)

The source-pinned original R2h cohort contains **36 matched disjoint versus per-value-optimal scratch allocations**, with all **72 original GCC host programs executed**, original source-address certificates and independently checked semantic/numeric results. All 36 pairs reduced **source C++ committed-value scratch scalar counts against a no-reuse disjoint arm**, not against a tensor-granular reuse planner or hypothetical fused accumulators. R2h also declares separate full-sized reader accumulator arrays, which the unit-slot metric excludes. This is bounded E1 source evidence, **not** an FPGA BRAM claim or a proof of a novel allocation algorithm.

The immutable R2i original native Vitis cohort at source revision `b2d5dcc4f18fad7cb1cd2c342101dd33185e817f` comprises **72/72 CSim-qualified E2** cases, **48/72 CSynth-qualified E3** cases, **24/36 paired E3 qualified**, and **12/36 E3 unknown** because native HLS `latency_best` is undefined. On the full 36-pair denominator, the qualified HLS estimate shows **6 BRAM_18K decreases, 18 ties and 0 estimated regressions**; 12 remain unknown. **HLS is an estimate, not routed resource use.**

After the original E3 HLS outcomes were known, the six HLS-estimated BRAM-positive pairs were selected **conditional on that post-HLS result, before their physical measurements**, for source-attributed Vivado OOC routing: 20, 23, 28, 31, 32, 35. Pair20 was routed in the original R2j study and the five remaining pairs in a separate create-only R2l follow-up. All twelve routed arms met requested-clock timing and were independently source/report-hash audited, but **all twelve had noncritical DRC warnings**; **0/6 matched pairs qualify for strict clean-DRC E4**.

| Pair | Cases | HLS BRAM_18K saving (estimated) | Routed BRAM tiles | Routed LUT | Routed FF | Routed DSP | OOC WNS (ns) |
|---|---|---:|---|---|---|---|---|
| 20 | 040/041 | 1 | 1.5 → 1.5 | 649 → 684 | 396 → 407 | 7 → 7 | +26.356 → +27.304 |
| 23 | 046/047 | 1 | 1.5 → 1.5 | 656 → 693 | 396 → 407 | 7 → 7 | +26.941 → +27.008 |
| 28 | 056/057 | 1 | 1.5 → 1.5 | 870 → 889 | 501 → 503 | 8 → 8 | +25.491 → +27.133 |
| 31 | 062/063 | 1 | 1.5 → 1.5 | 878 → 819 | 497 → 494 | 8 → 8 | +26.359 → +27.054 |
| 32 | 064/065 | 1 | 1.5 → 1.5 | 883 → 880 | 504 → 489 | 8 → 8 | +27.205 → +26.863 |
| 35 | 070/071 | 1 | 1.5 → 1.5 | 869 → 858 | 504 → 499 | 8 → 8 | +27.119 → +26.121 |

Each cell reports **disjoint → optimized**. Both arms were routed with the same source-pinned target part and requested timing constraints. The original 72-case/36-pair cohort is the full denominator; these six routes are the entire stratum selected **after E3** for positive *HLS-estimated* BRAM effects. All twelve routes disclose noncritical DRC warnings, and the table is **descriptive OOC evidence**, not strict clean-DRC E4.

**Physical translation finding.** All six routed BRAM results are **ties at 1.5 tiles per arm**, whereas each matched pair had an HLS-estimated reduction of one BRAM_18K. Routed DSP also tied in **6/6** pairs (7 per arm for two pairs; 8 per arm for four). The optimized arm increased LUT and FF in **3/6** pairs and decreased each in **3/6** pairs. Mean optimized-minus-disjoint utilization was **+3.000 LUT** and **+0.167 FF** across six related, unreplicated pairs with changes in both directions; neither demonstrates a consistent area advantage. Descriptive OOC WNS improved in **4/6** pairs and worsened in **2/6**, with mean difference **+0.335333 ns** and median **+0.381 ns**. Paired WNS differences span **−0.998 to +1.642 ns** at a loose **32 ns requested clock**. These are one-run, warning-disclosed OOC slack observations, not achieved maximum Fmax or evidence of a causal frequency increase. All these resource and timing results are **warning-disclosed descriptive OOC observations**, not strict clean-DRC E4.

This is a **negative result restricted to the six prespecified HLS-estimate-positive MatMul fanout configurations** under their original device, timing, numeric and tool settings. **30/36** original physical pairs were not routed and remain UNKNOWN, not ties or regressions. The complete original source cohort and qualifying native study results remain valid within their respective tiers. The external comparator/firstness gate remains unqualified.

The independent R2m paper-table exporter was verified twice on the user workstation: **11/11 focused and 1,447/1,447 full pytest** passed in both runs. The corrected v2 output was created at `build/r2m_original_six_physical_paper_v2/` without new vendor calls; `results.json` includes each routed arm's utilization, DRC rule counts and original four Vivado report SHA-256 fingerprints, and `all_36_pairs.csv` explicitly preserves the 30 unobserved physical pairs. The prior v1 export is retained, and original R2i/R2j/R2l experimental evidence remains frozen. See [R2m final export protocol](R2M_FINAL_ORIGINAL_PHYSICAL_PAPER_EXPORT_20261103.md).

### 5.2 Original source-only experiments and separately pending native studies

| Study | Generated cases or pairs | Host-numeric checks | Other independently checked evidence | Native Vitis |
| --- | ---: | ---: | --- | --- |
| Array layouts | 96 configurations / 48 pairs | 8/8 pass | 24 functional graphs; 96 distinct C++ source hashes | Not run |
| Two-MatMul row lifetime | 18 paired cases | 4/4 pass | Real producer/consumer row-loop source; up to 5× fewer declared C++ intermediate elements | Not run |
| Multi-operator row chains | 12 chain cases | 4/4 pass | 12/12 bounded liveness, 2/2 executed \(\Phi\)/row-epoch traces | Not run |
| Original native campaign | 12 HLS sources prepared | — | Source/testbench/Tcl digests, dry-run and read-only audit | Zero qualified cases |

The source-only studies substantiate **compiler implementation and selected source-level semantic conformance**. The separately completed R2i native fanout study additionally provides conditional HLS-estimated cycles/resources and the R2j/R2l study provides warned OOC routed resource/WNS comparisons; these datasets are **not interchangeable**. None demonstrates physical BRAM savings, power reduction, achieved maximum Fmax, deployed performance or superiority over another compiler.

### 5.3 What a future strict E4 and external comparison must establish

The completed R2i/R2j/R2l experiment establishes controlled source/HLS/routed evidence and a negative physical BRAM translation result. A stronger **physical implementation advantage** claim would require a *new and separately preregistered* matched source-controlled experiment with complete routing and **zero DRC violations in both arms**, appropriate repeated experimental controls and a clear requested-clock versus achieved-Fmax distinction. A claim of **external performance superiority** would additionally require a matched external baseline; this is **not required** merely to publish a bounded IR semantics and verification contribution. The contribution must nonetheless be positioned precisely against related work, without unsupported claims of novelty firstness.

### 5.4 Original R2r generated-source dynamic shadow mutation study (2026-11-03)

The earlier E0 event-mutation adequacy tests challenged checker logic, while R2h source E1 prints the claimed address next to each actual scratch access. An independent review correctly noted that these may share assumptions. We therefore **instrumented actual generated R2h C++ scratch LOAD and STORE sites** with a second dynamic shadow occupant/remaining-read checker based on graph topology, not on the generated program's printed `FPGAI_SLOT` or typed event log. The original R2h source-event and independent NumPy numerical checks run as a separate validation layer.

At the original clean source revision `fbdd34ea363647dbea700223f923d0dad8547732`, local tests passed (**22/22 focused, 1,531/1,531 full pytest**), followed by **six** original unmutated GCC-qualified generated-C++ baselines and **216 intentionally changed source C++ programs** (six graphs × nine fault families × four payloads). Across **222 original GCC attempts**, the full mutant denominator was **192 caught by dynamic source shadow occupancy/read counts, 24 caught by the original R2h source/numerical checker, 0 survivors, 0 compile-invalid, 0 preflight-invalid and 0 unknown runtime failures**. A separately run **read-only source/mutant SHA audit** agreed on all populations and hashes but **did not independently recompile or execute any GCC case**.

These results support **detection of these selected source-operation/address/numeric faults in this bounded fanout template**, not a theorem of general source-refinement, an independent binary-level shadow mechanism, arbitrary-input or RTL correctness, hardware memory-bank concurrency or physical QoR. The injected shadow uses static source graph/shape obligations and a generated instrumentation wrapper. [Complete original R2r protocol](R2R_EMITTED_SOURCE_SHADOW_MUTATION_PROTOCOL_20261103.md).

### 5.5 Original R2q adversarial source-memory baselines (2026-11-03)

An independent reviewer observed that the original 36 matched R2h allocation pairs use a **disjoint no-reuse baseline**, and that the reader-output lifetime model excludes the period during which a complete, separately declared `acc` array is being accumulated. We implemented a **new, source-pinned E0-only** baseline sensitivity study over the same 36 graph/mapping configurations; original revision `7890c6f363cc9dd61c8bfa29ae121107ee3d86de` passed **17/17 focused, 1,516/1,516 full pytest**. The distinct, read-only source/revision/CSV audit independently recomputed all 36 rows and agreed. No additional original GCC, Vitis, Vivado or trained RL runs occurred.

| Source-accounting comparison | Observed denominator |
|---|---:|
| Committed-value element-exact scratch beats **disjoint no reuse** | **36/36** |
| Feasible, replay-verified **tensor-granular first-fit** beats disjoint | **32/36** |
| Committed-value element-exact scratch beats **feasible tensor first-fit** | **36/36** |
| Committed-value element-exact scratch beats **optimistic tensor concurrency lower bound** | **36/36** |

The tensor concurrency number is a **lower bound**, not an achieved tensor-granular layout; the feasible first-fit case is a separate validated source-address assignment. The element allocation optimum is classical interval coloring for an explicitly **fixed total source event order**, not a new general allocator. The current source emitter contains full reader `acc` arrays and separate counters: their declared elements are reported separately from unit scratch slots.

We then compared three non-scalar Phi policies with the scalar policy for each of nine graph/shape configurations (**27 distinct, not independently sampled, mapping interventions**):

| Reader-output lifetime convention | Reductions | Ties | Increases | Largest absolute Phi capacity change |
|---|---:|---:|---:|---:|
| Current **COMMIT→last READ** source scratch | **18** | **9** | **0** | **2 scalar slots** |
| Hypothetical **first MAC→last READ** in-place reader output | **0** | **27** | **0** | **0 scalar slots** |

The second line is a **different counterfactual memory system**, not the current emitted R2h C++ and not independent GCC, RTL or FPGA hardware evidence. It nevertheless **falsifies lifetime-convention-invariant Phi scratch benefits** in the studied domain. The strongest surviving C1 statement is a bounded, explicit **source issue order and final-reader/release relation** under declared PE/SIMD and buffer accounting assumptions—not that Phi produces a general FPGA memory saving. Source event-address conformance C2 remains an independently testable hypothesis; generated-source mutation with independent shadow memory and stronger peer capability comparison remain publication gates.

### 5.6 Original R2s bounded Phi scaling and alternative lifetime study (2026-11-03)

To challenge the bounded nature and memory-convention sensitivity of the committed-value storage reduction, we executed a separate prespecified, original **source-only E0** study covering **147 complete configurations**: seven shapes (up to `(m,k,n,q)=(16,2,16,16)`), two to four independently mapped reader MatMuls, and seven scalar/PE2/PE4 policies. The original clean Git revision `f38a98cb96a96097174204773462a53b8e6c02e2` passed **19/19 focused and 1,543/1,543 full regression tests**. The create-only 147-case study and separate, read-only same-revision **source recomputation** audit agreed on **126 scalar-controlled non-scalar mapping contrasts**, with no original GCC/Vitis/Vivado or RL training calls.

| 126 controlled Phi contrasts versus scalar | Committed-value source slots | Hypothetical first-MAC in-place peak |
|---|---:|---:|
| Reductions | **102** | **0** |
| Ties | **24** | **126** |
| Increases | **0** | **0** |
| Maximum reduction in source slots | **12** | **0** |

All **126/126** measured committed-value contrasts satisfied the reviewer-proposed inequality `S_Phi <= R*(max_reader_PE−1)` under the **fixed serial source issue policy**. No violation was observed, but this bounded sample **does not prove the inequality for general source schedules, tensor dimensions or hardware implementations**. Across the study, the largest absolute committed-value saving was **5.882% of complete declared mutable source array elements**; for the largest shape `(16,2,16,16)` it was **0.389%** (four readers, all PE4, **12** source scratch slots saved from **1036** to **1024**, versus **3084** scalar-baseline total declared mutable elements). Therefore even the largest observed scratch-slot difference is not necessarily significant at the complete source-memory-system level, let alone FPGA BRAM. The hypothetical first-MAC in-place model is **not an emitted or GCC-executed implementation**. Both source-only accounting conventions and the exact denominator must be preserved; the original evidence is pinned and frozen. See [R2s original source study protocol](R2S_PHI_SCALING_SOURCE_PROTOCOL_20261103.md).

### 5.7 Original R2t exact tensor and R2u native HLS mechanism studies (2026-11-03)

**R2t — strongest feasible whole-tensor source comparator.** We ran the newly preregistered reference on the original 36 graph/Phi source configurations at clean source revision `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`; local focused **29/29** and full **1,565/1,565** regression tests passed. The original **36/36 contiguous whole-tensor global minima were proven**, with **0 UNKNOWN_BUDGET cases**, and the separate source-recomputing read-only audit agreed. **12/36 optimal whole-tensor layouts saved source slots compared with the feasible greedy first-fit heuristic**. All **36/36 globally optimal contiguous tensor layouts equaled the concurrency lower bound** in this selected cohort. The earlier lower bound was therefore attainable for these exact graphs and fixed lifetime intervals—but attainability is not a theorem for arbitrary tensor intervals. The **committed-value scalar element-exact** assignment used fewer source slots than the newly proven feasible whole-tensor optimum in **36/36** configurations. This is the expected strength of classical element-level interval coloring when the tensor-block comparator requires each tensor to occupy contiguous memory through its full first-COMMIT→last-READ envelope. It **does not** establish a novel allocator, a fused first-MAC in-place implementation, FPGA BRAM savings or area/timing superiority.

**R2u — source-attested factorial in native Vitis HLS.** At the same original pinned source revision, we independently prepared and audited all **32 packages** (two fixed fanout graphs, (2^4) factor arms: scratch allocation, address-table versus switch lookup, automatic versus requested `ram_2p` BRAM binding, and exact versus 64-aligned declared array capacity). The full original **32-case opt-in Vitis invocation completed with successful tool exits**. **32/32 original CSim cases qualified E2**, with **30/32** satisfying the stricter numerical, resource and effective-clock **E3** gate. **Two actual source cases, `case_004` and `case_012`, failed the effective HLS clock budget**, even though they passed CSim and Vitis exited successfully; their resource deltas must not be scored as zero or as ties. The original requested clock was **8 ns** with **1.5 ns uncertainty**, and the accepted numeric threshold was **0.01 absolute**, not bit-exact fixed-point equivalence. Raw HLS estimated BRAM_18K for the qualified cases varied with graph, requested binding and padding; source scratch savings therefore do not directly determine estimated BRAM usage. The **64 planned same-graph conditional two-arm contrasts** include comparisons adjacent to the two unqualified cases; the new read-only **R2v** analysis, which reports qualified versus UNKNOWN comparisons without rerunning any vendor jobs, is **implemented but has not yet been executed against the original native directory**, so a full independently audited factor-effect table is pending.

These measurements are **HLS E3 estimates, not physical Vivado E4**. No new R2u Vivado place/route, clean DRC, achieved routed Fmax, physical RAM primitive binding, power or on-board measurement has been obtained. The `bind_storage` pragma documents a request; unused padding may be optimized away, and neither implies a physically implemented memory primitive. The older original six selected, warning-disclosed Vivado OOC pairs remain **six routed BRAM ties and 0 strict clean-DRC E4-qualified comparisons**. Claims of an explained routing discrepancy remain **open** pending actual RTL/RAM instance report inspection and an independently qualified routed study. See [R2t/R2u protocol](R2T_R2U_ALLOCATION_AND_HLS_MECHANISM_PROTOCOL_20261103.md) and the new [R2v read-only factorial review](R2V_READONLY_NATIVE_FACTORS_PROTOCOL_20261103.md).

### 5.8 Formal source event refinement scope

For a fixed total sequence of unique value COMMITs and subsequent READs, a source scratch map is interval-safe when simultaneously live logical tensor elements never occupy the same source slot. An elementary induction then establishes that the sequential scratch machine returns the last committed value at every later read, **conditional on exact generated-source adherence to the same event/address semantics**. This is not independently a novel mathematical theorem or a proof of all-input compiler correctness. The classical unit-value fixed-order slot optimum is the peak live-value count (interval graph coloring); the R2t whole-tensor optimum adds contiguous block placement but remains classical fixed-lifetime allocation. The observed R2s inequality `S_Phi <= R(max_reader_PE−1)` is still **empirical and unproved beyond its finite 126-case source cohort**. The E0→E1→E2/E3→E4 evidence limits and full proof obligations are separated in [formal source event obligations](FORMAL_SOURCE_EVENT_REFINEMENT_AND_TENSOR_BASELINE_20261103.md).

### 5.9 Original R2v independently re-audited HLS factor contrasts (2026-11-03)

After all 32 R2u native vendor runs had completed, an additional **read-only, source-pinned R2v analysis** verified all original case/source/Tcl/testbench report hashes, all **32 SHA-attested CSynth XML files**, the **32/32 CSim E2** passes and **30/32 effective-clock-qualified CSynth E3** cases. The postprocessing source at revision `23e998a0553e3eec6f05c5d3cd09a9b4a81af83e` passed **26/26 focused and 1,572/1,572 full pytest**, and the new create-only original analysis `build/r2v_r2u_native_factorial_review_v1/` completed without any further Vitis/Vivado calls. Exactly **57/64** prespecified conditional two-arm factor contrasts had E3-qualified endpoints; the other **7/64** remain UNKNOWN because `case_004` and `case_012` failed effective clock qualification. Their original CSynth XML files were nevertheless SHA-attested, and neither timing failure is silently imputed as a tie or resource saving.

| Factor intervention (intervention − control) | Estimated BRAM decreased | Tied | Increased | E3 UNKNOWN |
|---|---:|---:|---:|---:|
| Disjoint → element-exact scratch | **2** | **13** | 0 | 1 |
| Static address table → switch-case lookup | 0 | **14** | 0 | 2 |
| Automatic memory inference → requested BRAM `ram_2p` | 0 | **10** | **4** | 2 |
| Exact source declaration → padded to next 64 elements | 0 | **10** | **4** | 2 |
| **Total** | **2** | **47** | **8** | **7** |

Storage-layout-induced HLS BRAM reductions were absent in all **7 eligible** two-reader graph pairs but present in **2/8** four-reader graph pairs; hence no single general physical savings statement follows. At the HLS *estimate* level, 4/14 binding pairs and 4/14 padding pairs consumed more estimated BRAM under their intervention, while the 14 eligible table-to-switch lookup pairs tied in estimated BRAM. Crucially, two **switch/automatic/exact** source arms (`case_004`, `case_012`) did **not** qualify at the effective 8 ns requested clock, so an apparent constant BRAM count cannot be interpreted as equivalent performance. HLS resource counts do not reveal instantiated FPGA physical RAM cells or explain the older routed BRAM ties.

A further **R2w read-only RTL/ROM provenance and memory-identifier inventory** is implemented but **not yet executed locally**. It will inspect original HDL/ROM SHA maps and source-attested CSynth textual reports, retaining unknown files and using only bounded, explicitly lexical memory-name evidence. A string such as `RAMB18E2` in generated RTL might occur in a comment or unused module and is **not** routed primitive inference. No new original physical implementation has been made. See [R2v completed factorial protocol](R2V_READONLY_NATIVE_FACTORS_PROTOCOL_20261103.md) and [R2w prospective RTL forensic protocol](R2W_SOURCE_ATTESTED_RTL_MEMORY_FORENSICS_20261103.md).

### 5.10 Original R2w attested RTL inventory and prospective R2x static module analysis

A separate original **read-only R2w** study at analysis revision `bb98983a438efe9e65c0204041895353730f964e` passed **22/22 focused and 1,580/1,580 full regression tests** and verified the SHA-256 of **every generated HDL/ROM inventory in all 32 original Vitis HLS cases** against the preserved `0c6d85f189062df8d5157237f3c8ac74ac8da4b8` vendor receipts. All **64 conditional RTL pairs** had original RTL SHA-attested evidence, irrespective of the **seven E3-UNKNOWN HLS pairs**, and **no Vitis/Vivado processes were launched**.

| Original Vitis graph | Static address table: RTL files / memory-named filenames | Source switch lookup: RTL files / memory-named filenames | Scratch-named files (both modes) |
|---|---|---|---:|
| Two-reader fanout | **23 / 10** | **24 / 6** | **7** |
| Four-reader fanout | **30 / 16** | **32 / 8** | **9** |

The source address-representation choice thus coincides with measurable differences in the original RTL **file organization** even when the qualifying HLS BRAM estimate remains tied in all **14/14** table-versus-switch pairs. These are **filename/token observations**; an emitted module need not be instantiated or survive native logic optimization. Therefore counts of files or names must not be treated as physical RAM instance counts, memory-port designs or routed BRAM.

A prospective, separate **R2x read-only static Verilog hierarchy study** is implemented but **not yet workstation tested**. R2x will distinguish module definitions and a bounded recognized subset of type-named instantiations, calculate provisional static reachability from `fpgai_model`, and compare scratch/memory-named module-type candidates across all 32 source-attested cases and 64 source-attested factor pairs. Unsupported Verilog, missing top modules, ambiguous instantiation or macros will remain explicit **UNKNOWN**, not “no RAM.” Even a reachable `RAMB18E2` *text token* cannot establish an implemented physical RAM cell. See [R2w completed protocol](R2W_SOURCE_ATTESTED_RTL_MEMORY_FORENSICS_20261103.md) and [R2x planned static hierarchy protocol](R2X_STATIC_HLS_VERILOG_HIERARCHY_PROTOCOL_20261103.md). Native RTL synthesis/placement, independent peers and the emitted first-MAC in-place comparator remain unresolved.

### 5.11 Original R2x static hierarchy and new R2y routed physical pilot (2026-11-03)

The source-pinned original **R2x** read-only static Verilog analysis at revision `e5ea6a14d279be786492f338bdf1e51ba684ba73` passed **23/23 focused and 1,588/1,588 full pytest**. All **32/32** generated HLS RTL cases and **64/64** matched source pairs were SHA-attested; all 32 cases had a static `fpgai_model` top and no parser-unknown statuses on the tested Verilog subset. For the two-reader graph, table-address source produced **10 recognized reachable memory-named module types and 10 recognized instances**, whereas switch-address source produced **6 and 6**; for the four-reader graph the corresponding values were **16 and 16** versus **8 and 8**. The contrast held across both layouts and binding/padding settings, and **does not establish implemented RAM cells** because the source graph is not elaborated, optimized or physically placed. All **14 eligible HLS table/switch BRAM comparisons remained estimates ties**. The generated HDL organization can change independently of HLS-estimated BRAM.

To cross the remaining source→physical evidence boundary without selection after routing, a **prospective nine-case, seven-contrast R2y Vivado OOC pilot** is now implemented but **NOT tested or executed on the workstation**. It reuses nine original R2u cases with E3-qualified 8 ns HLS source: `case_000,001,002,008,016,017,018,020,024`; pairs `000→008` storage, `000→002` binding, `000→001` padding, `016→024` storage, `016→020` address, `016→018` binding and `016→017` padding. The source and numerical/Tcl controls are matched by construction; the two E3 clock-rejected switch cases `004` and `012` are excluded rather than misrepresented as valid physical input. Original RTL, source ROM, Vitis invocation and Vivado Tcl hashes are independently preserved in a new create-only output. Native Vivado execution requires **two explicit opt-in flags** and never re-runs Vitis.

The prespecified R2y qualification rule is strict: **fully routed, WNS nonnegative, complete source-attested resource/timing/DRC reports and ZERO verified DRC violations and critical warnings for BOTH matched arms**. Otherwise, the physical E4 pair remains UNKNOWN; any separately valid but DRC-warned routed observations are published only as descriptive warning evidence. In particular, the `016→024` pair is a preregistered test of an original **one-unit estimated HLS BRAM decrease**, but its physical resource difference is **currently unknown**. No new R2y routing, physical BRAM, achieved Fmax, board measurements or hardware correctness claims are reported. See [R2x verified protocol](R2X_STATIC_HLS_VERILOG_HIERARCHY_PROTOCOL_20261103.md) and [R2y controlled E4 pilot protocol](R2Y_NINE_CASE_STRICT_E4_PHYSICAL_PILOT_20261103.md).

### 5.12 Mixed CNN-to-Dense Model IR implementation (R2z, prospective)

The general-purpose operator source emitter already implemented convolutional and fully connected neural operations separately, but a bounded unified **Conv2D → pooled/activated 4D network → Flatten → multiple rank-2 Dense layers** compiler with explicit cross-rank interface contracts, complete stage identity preservation and end-to-end actual intermediate-value evidence was missing. R2z implements this **engineering extension without creating independent second Conv/MatMul emitters**. Its typed `NeuralChainIR` and `compile_neural_chain` admit static NCHW non-grouped, undilated Conv2D with optional constant bias, ReLU/MaxPool/GlobalAveragePool, exactly one axis-1 Flatten, constant-weight MatMul or biased Gemm with independent per-operator PE/SIMD/tile requests, and rank-2 ReLU/Sigmoid/Softmax. Every tensor boundary is explicitly **full materialization before its downstream consumer**; there is no implicit in-place accumulation, streaming or inferred physical memory claim. Branching and unsupported operators fail closed.

The new `verify_neural_chain_gcc` executes the **actual canonical generated C++** with `FPGAI_TRACE_TENSORS` and checks **every emitted intermediate element**, not merely the final prediction, against independently evaluated NumPy Model IR values. Tests cover three prespecified mixed CNN topologies, scalar and independently mapped architectures, input/bias/geometry/fixed-point-format failure modes, source-weight fault mutation, source SHA guards, and actual strict ONNX Conv→Relu→Flatten→Gemm import. A separate **R2z source-only original experiment (not yet run)** plans **six source packages = three topologies × two mappings**, two deterministic runtime inputs each = **12 original GCC E1 intermediate-value executions**, and **six Vitis HLS C++/testbench/Tcl native-ready but unexecuted packages** at 10 ns/1 ns uncertainty with 0.01 absolute CSim tolerance. All source tests and this experiment **remain workstation-unverified** at writing; no original numerical pass count, HLS/RTL resource or physical speedup can yet be claimed.

This establishes a defensible bounded end-to-end **CNN + multilayer fully connected source coverage** objective and stage-level falsifiability, not a generally novel neural architecture, standard Conv implementation or new allocation theorem. Prior R2u–R2x source/native originals remain immutable, and **the first new R2y Vivado case was already routed only with disclosed DRC, not strict clean E4**. See [R2z reproducible source experiment protocol](R2Z_MIXED_CONV_DENSE_NEURAL_NETWORK_PROTOCOL_20261103.md).

### 5.13 Prospective native CNN-to-Dense HLS validation: R3a (source-only code, no native results)

The R2z E1 source cohort is designed to create three deterministic full mixed CNN/Dense networks under two individually controlled architecture policies (scalar and independently mapped Conv/Dense), yielding **6 source packages and 12 original GCC intermediate-tensor input traces**. A new R3a opt-in native HLS runner/auditor is implemented but **NOT workstation verified or run**. Its three preregistered matched source comparisons are `case_000→001`, `002→003` and `004→005`, holding each complete original network's functional graph SHA, testbench, test inputs, part, requested 10 ns HLS clock and 1 ns source Tcl uncertainty fixed. It **reuses the source-attributed original CSim E2 and effective-clock-qualified Vitis CSynth E3 auditor**: zero fixed-point CSim failures, exactly two samples/full output count, original vendor identity and report SHA, HLS schedule feasibility and complete used-resource and worst-cycle fields. Only fully qualified two-arm E3 pairs can report estimated BRAM_18K/LUT/FF/DSP/cycle differences. Missing, failed or incomplete native results remain **UNKNOWN**, not equal. No R3a Vitis or Vivado processes have been run and **no E3/E4 benefit is claimed**. See [R3a prospective native protocol](R3A_CNN_DENSE_ORIGINAL_HLS_NATIVE_PROTOCOL_20261103.md).

## 6. Related work

**Spatial and temporal accelerator mappings.** Timeloop, MAESTRO, AutoSA and other polyhedral/data-centric frameworks already formalize iteration placement and temporal reuse. FPGAI's \(\Phi\) builds on this established concept and cannot be called the first such mapping.

**Composable HLS and hierarchical dataflow.** Allo (PLDI 2024, DOI 10.1145/3656401) decouples and composes compute, memory, communication and datatype customization. HIDA (ASPLOS 2024, DOI 10.1145/3617232.3624850) introduces hierarchical functional/structural dataflow compilation. Their achievements overlap with a generic claim of compositional hardware IR.

**Streaming and tensor layout types.** StreamTensor (MICRO 2025, DOI 10.1145/3725843.3762817; arXiv 2509.13694) represents stream layouts using iterative tensor types and supports inter-kernel fusion and memory/buffer optimization. Merely encoding column-major or row-epoch layout cannot establish new scientific priority.

**Recent FPGA dataflow compilers.** ST-Flow (PACT 2026, DOI 10.1145/3838684.3846865) maps PyTorch/MLIR workloads into spatial, temporal or hybrid FPGA dataflow, supports cross-kernel sharing and evaluates Vitis HLS artifacts. This is a particularly important direct comparison. CODO (ISCA 2026) addresses dataflow violation detection, scheduling and data movement; FuseFlow (ASPLOS 2026, DOI 10.1145/3779212.3790165) addresses sparse fusion-centric accelerator compilation. Wavelet (2026, DOI 10.1145/3808263) offers verified asynchronous dataflow compilation. FPGAI does not claim to originate any of these broad ideas.

**Additional current neural and dataflow baselines.** hls4ml (ACM TRETS 2026, DOI 10.1145/3801979) already provides end-to-end neural model→HLS compilation and multi-vendor acceleration. SODA/ML4SODA (GLSVLSI 2025, DOI 10.1145/3716368.3735223) includes an MLIR neural HLS flow and exploration of convolutional/fully connected kernels with temporary-buffer optimizations. Filament (PLDI 2023, arXiv 2304.10646) already has modular temporal timeline types, while Dahlia (PLDI 2020, DOI 10.1145/3385412.3385974) covers time-sensitive affine resource typing. **Thus neither Conv/Dense source coverage, typed memory/layout composition, hardware timing types nor independent operator optimization is independently novel.** See [public-source-scoped 2026 related-work mechanism matrix](RELATED_WORK_COMPILER_IR_NOV2026_VERIFIED_MATRIX.md). These public sources establish overlaps, not experimentally verified claims that any rival lacks our precise source liveness checker.

**Precise open question.** The defensible distinction, still to be demonstrated against independently inspected artifacts, is whether an explicit *operator \(\Phi\) + per-value tensor lifetime* intermediate representation with a separately computed finite source-level liveness certificate provides useful guarantees or automation that existing systems do not supply for the same neural-network design. A restricted ablation within FPGAI alone does not settle that comparison.

## 7. Limitations and threats to validity

The validated row-chain lowering is limited to static fixed-shape MatMul/Gemm MLP chains, while a distinct R2f/R2h path supports bounded **consumer-specific last-reader scratch aliasing** on two-to-four-reader MatMul fanout templates. The newly implemented but **not yet workstation-verified R2z** path adds *full materialized tensor* end-to-end sequential NCHW Conv→pool/activation→Flatten→Dense source compilation, reusing the general Conv/MatMul/Gemm source emitter. It **does not** implement the fanout path's consumer-specific scratch reuse for these CNNs, nor arbitrary branched graphs, physical streaming, port-concurrent memories, dynamic shapes, mixed numerical formats without casts or multi-clock execution. The larger generic emitter's operator inventory must not be mistaken for whole-network artifact-backed source/native validation.

The bounded liveness witness is an interpreter of the prescribed symbolic schedule; it is **not a verified theorem about an external RTL implementation**. Host-float C++ tests and trace macros establish correctness for selected generated-source executions, not bit-exact fixed-point equivalence. The E0 bound excludes arbitrarily large symbolic tensor domains. The completed R2i original study has conditional E3 Vitis estimates and the selected R2j/R2l original study has six warning-only OOC routed matched comparisons, **not strict clean-DRC E4**. In the six HLS-positive-selected, warning-disclosed physical observations, routed BRAM **did not decrease**; 30 other matched pairs have unknown physical outcomes and no mechanism for the six ties is established.

The study explores only a limited set of models, widths and architecture choices. The E2 gate uses 0.1 absolute numerical tolerance and the emitted HLS source includes runtime iteration decoding and slot-address lookup tables; storage capacity, address-generation cost and HLS automatic memory binding are confounded. The 12 E3-unknown pairs and 30 physically unrouted pairs cannot be called ties. A matched same-compiler ablation shows concrete expressibility but does not prove broader superiority over Allo, HIDA, StreamTensor, ST-Flow or CODO. Related-work artifact reproduction, independent external comparison and new native Vitis evaluation are conditions for strong publication-level claims.

### C3. Bounded learning-compatible compiler IR interface

The **source-attested, bounded decision interface** is implemented in
`model_ir/decision_interface.py`: immutable complete design requests;
typed action patches; `transition` that invokes the ordinary compiler
resolver/lowerer and returns either the new source-bound certificate
or a concrete rejection; `graph_observation` for variable-size
operator/tensor/edge graphs; and `certified_action_mask` returning
`BOUNDED_MAPPED_VALUE`, `REJECTED` or `UNKNOWN`.
UNKNOWN is never presented as certified legal. The generic
`action_mask` tests source acceptance only and must not be confused
with the stronger mapped-value E0 mask.

R1 source action/catalog tests and a restricted independent oracle
are already in the decision-interface campaign. The prospective,
newly integrated `cst_rl_interface_study` is a bounded, create-only
R3/R4 study over three static chain depths (2,3,4) and two controls:
renaming anonymous graph observations, and replaying identical
complete actions from several proposer labels through exactly the
same compiler/certificate path. Its finite action replay and renaming outcomes should be interpreted as **API regression evidence**, not independent RL optimization, general graph-invariance, mapping-sensitive legal-action discrimination or physical HLS proof.

We therefore report **an implemented RL-usable, policy-independent bounded compiler API**, as a useful secondary research-enabling interface rather than a separately validated novel RL algorithm or mapping-sensitive action mask. Learned performance impact and transfer remain unmeasured.
The interface is an application of the CST certificate semantics,
not a replacement for the primary CST research question.

**New original C3 source experiment, R2n (workstation verified 2026-11-03):**
the `source_executed_action_mask` passes each complete-design action
through the ordinary compiler transition and distinguishes actual
`E1_GCC_SOURCE_QUALIFIED` from compiler `REJECTED`,
uncertified `UNKNOWN_E0_CERTIFICATE`, or source execution failure.
The new source-pinned experiment and independent read-only audit covered
**three static MatMul chains (operator depths 2, 3, 4), each with three
prespecified actions (9/9 total)**. Across the full denominator,
**3 actions were independently GCC-executed E1-qualified**, **3 were
rejected**, **3 remained E0 unknown**, and **none of the three attempted
GCC executions failed**. At original Git revision
`6296f99eb5d52371bffb9755989a340aadbe7ff6`, the workstation
reported **37/37 focused** and **1,464/1,464 full pytest passes**.
This is bounded source/GCC host-float evidence only, not a learned RL
performance comparison or hardware-feasibility mask. See the
[R2n exact original protocol](R2N_C3_E1_QUALIFIED_DECISION_IR_20261103.md).

**Original R2o C3 multi-reader source experiment (workstation verified
2026-11-03):** a separate `FanoutDecisionState` adapter shares the typed
Functional graph, architecture requests and `DecisionAction` with C3,
but reuses the *existing* R2f multi-reader `Phi`/last-reader E0
certificate and R2h actual generated C++ GCC scratch-address/numeric
verifier. On an original source-pinned three-graph × four-action study
covering **2, 3 and 4 independent MatMul readers**, the full
**12/12 action denominator** consisted of **nine actually executed
`E1_GCC_MULTI_READER_QUALIFIED` actions**, **three rejected producer
PE edits**, **zero E1 execution failures**, and **zero uncertified
unknown outcomes**. The original generated C++ checks recorded
**102, 210 and 372 original source-address witnesses**, respectively,
per qualified action at the three fanout degrees. A separate read-only
source, manifest, stdout SHA and exact population audit agreed with
the original reported outcomes, but was **not a second GCC/numerical
replay**. Original clean Git source revision
`ce8083c0ad5b827232e1e67322811b25edd3ac79` passed **27/27 focused**
and **1,484/1,484 full pytest** locally. All native vendor calls and
trained RL policies in R2o were **zero**. This result is *bounded
single-input source E1*, not all-input semantics, arbitrary-DAG action
legality, hardware bank/timing feasibility or RL learning evidence.
See [R2o source/receipt protocol](R2O_C3_MULTI_READER_DECISION_SOURCE_PROTOCOL_20261103.md).

**Original R2p common C3 decision facade (workstation verified
November 3, 2026):** a new typed `unified_decision_interface`
provides one action/transition/status/weight-private-graph observation
vocabulary for both bounded sequential and R2o multi-reader designs.
It dispatches to the *existing original compiler paths* and refuses
to promote source-admitted E0-UNKNOWN or failed GCC actions to certified
E1. A **six-action, two-domain** controlled GCC experiment and a
read-only population/source-hash audit were executed in the clean original revision `dac395598f2ab96bc9b7a41255b0529c72b22593`: **six actions** produced **three actual GCC E1 qualifications, two compiler rejections, and one E0 UNKNOWN**, with a matching independent read-only receipt audit. The local focused regression passed **29/29** and the full suite **1,506/1,506**. This is interface consolidation, not proof
that the two source domains have identical semantics, a new RL policy
or universal graph certification. See
[R2p scientific protocol](R2P_UNIFIED_C3_SOURCE_INTERFACE_PROTOCOL_20261103.md).

### Historical separate v6 native cohort (not the completed R2i/R2j/R2l population)

The older original 12-case Vitis explicit-BRAM HLS CSim/CSynth study is a
source-attested E3 observation in a separately immutable case
population. It did NOT demonstrate universal BRAM savings; the six
paired BRAM_18K estimates were five ties and one regression.
New original Vivado v6 `case_000` passed the independently audited
**warning-disclosed routed observation** tier after one foreground
native call, while `strict E4 cases=0`, `matched routed pairs=0`,
`case_001` through `case_011` remain unrun. It cannot be used to
claim a causal physical savings effect. The original v6 case's
numerical route/timing/DRC report must be read before publishing
any numeric result.

## 8. Conclusion

We implemented a compositional space–time tensor-lifetime research extension to the FPGAI Model IR. The typed intermediate contracts couple operator-local execution mappings to concrete tensor-address and row-epoch lifetime relations. Supported transformations alter actual generated HLS C++, and independent finite liveness and host-executed checks validate selected semantic properties. Source-pinned CPU experiments pass their selected numerical qualifications across array, two-operator and multi-operator cases. This demonstrates a functioning research compiler mechanism, while preserving distinct status for model-level legality, host-source evidence, native HLS synthesis and physical FPGA realization.

The subsequently completed original fanout experiments extend this work beyond CPU-only source checks: **72/72 Vitis CSim**, **24/36 paired source-attributed CSynth E3 qualifications**, and six selected source-pinned OOC routed comparisons with **six routed BRAM ties and mixed LUT/FF/WNS changes**. None of the six routed pairs passes the **strict clean-DRC E4** gate, and no deployed hardware, power, achieved Fmax, or uniform physical resource benefit is claimed. The candidate primary contribution is a bounded executable source-event contract for permitted reader mappings, element readiness, all-reader last use, and source address conformance. Its **scientific distinctness remains unresolved** until stronger tensor-granular baselines, actual accumulator-lifetime accounting, checker mutation and prior-art capability analysis are completed. The R2n/R2o/R2p C3 interface is useful but no learned RL or mapping-dependent legality advantage has been demonstrated. The physical BRAM/LUT/FF/DSP/WNS results remain truthful, narrowly scoped negative translation evidence.

## References (verified core bibliography)

1. H. Chen et al., **Allo: A Programming Model for Composable Accelerator Design**, PLDI 2024. DOI: https://doi.org/10.1145/3656401.
2. H. Ye, H. Jun, D. Chen, **HIDA: A Hierarchical Dataflow Compiler for High-Level Synthesis**, ASPLOS 2024. DOI: https://doi.org/10.1145/3617232.3624850.
3. H. Ye, D. Chen, **StreamTensor: Make Tensors Stream in Dataflow Accelerators for LLMs**, MICRO 2025. https://arxiv.org/abs/2509.13694; DOI: https://doi.org/10.1145/3725843.3762817.
4. S. Basalama, S. Pouget, J. Cong, **ST-Flow: A Hardware Compiler for Automating Spatial-Temporal Dataflow Acceleration**, PACT 2026. DOI: https://doi.org/10.1145/3838684.3846865; artifact: https://github.com/UCLA-VAST/ST-Flow.
5. R. Lacouture et al., **FuseFlow: A Fusion-Centric Compilation Framework for Sparse Deep Learning on Streaming Dataflow**, ASPLOS 2026. DOI: https://doi.org/10.1145/3779212.3790165.
6. Z. Lin, Y. Cai, M. Surbatovich, **Let it Flow: A Formally Verified Compilation Framework for Asynchronous Dataflow**, 2026. DOI: https://doi.org/10.1145/3808263.
7. **CODO: An Automated Compiler for Comprehensive Dataflow Optimization**, ISCA 2026. Public artifact: https://github.com/sjtu-zhao-lab/codo-artifact. Bibliography author list and DOI require final cross-check.
8. Original FPGAI progressive-Model-IR draft and its citations to FINN, hls4ml, Timeloop, MAESTRO, AutoSA, ScaleHLS, Halide and TVM: maintain as related foundations rather than reproduce unverified bibliographic entries here.

## Release and submission status

**Paper draft complete as a source-grounded technical narrative, not a submission-qualified manuscript.** Do not submit without reviewing ownership/IP, authorship and relevant original RVO/RU research affiliation constraints; original native Vitis numerical/HLS performance experiments; external prior-art evidence; and final editorial review. Unmeasured claims remain clearly unmeasured.
