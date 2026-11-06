# Contract-Aware Model IR for Policy-Independent FPGA Architecture Exploration

**Active scientific working manuscript, updated 6 November 2026. Two-contribution compiler paper; MAJOR REVISION, not submission-ready.** [Evidence-to-claim and submission lock](FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md).

**Scope decision.** This paper studies **C1**, a bounded compositional, operator-architecture-aware source-event contract, and **C2**, a policy-independent, learning-compatible compiler search interface. Training a GNN/RL agent, reward optimization, HLS learning curves, and performance advantages for trained agents are outside this paper and reserved for a separate study.

## Abstract

FPGA architecture exploration exposes choices in numerical precision, operator spatial mapping, source issue order, and intermediate storage that are not captured by a neural-network graph alone. We study whether a bounded compiler intermediate representation can connect complete operator-local architecture assignments with executable tensor-value ownership and last-reader obligations, and expose those *same* compiler decisions to independent search policies. Our first contribution (C1) is a finite typed source-event contract covering contribution completion, committed value availability, multi-reader reads, scratch-slot reuse and operator issue-order conformance under explicit static-domain assumptions. Our second contribution (C2) is a policy-independent interface for complete-design transitions, a weight-private typed graph observation and three-valued source admissibility (`CERTIFIED_E0`, `REJECTED`, `UNKNOWN_E0`), deliberately separate from native hardware feasibility.

Original bounded source experiments agree with a separate exhaustive fixed-order slot oracle on **5/5** small cases and reject **25/25** selected adverse mutations; a separately implemented finite event interpreter agreed with the source verifier on **133/133** tested traces. Across six variable-sized source graphs, **21/48** compiler actions were certified, **24/48** rejected, and **3/48** remained unknown. A precompiled 48-action candidate catalog enabled selection of previously certified states without *additional* compiler calls, while still requiring **48 upfront compiler transitions**. A subsequent independently source-revision-pinned seeded metamorphic study covered **54 additional bounded fanout configurations / 486 selected event traces**, with **zero observed independent-oracle disagreements**. A matched source-only CPU benchmark recorded **324 direct/lazy/eager method observations** with identical certified output identities, but measured time superiority remains **unanalyzed**. These tests support finite-domain correctness and interface determinism, not unrestricted semantic soundness, superior search efficiency, or trained-agent performance.

Source-attested Vitis C simulation qualified **6/6** neural-network variants, with **4/6** meeting the stricter HLS synthesis timing/resource gate; the two qualified two-arm mapping contrasts increased estimated cycles with no HLS-estimated BRAM reduction. In a separate, source-schedule-matched 32/6 storage experiment, HLS estimated BRAM tied at **13 versus 13**, while actual completed Vivado OOC reports showed **7 versus 4 mapped BRAM tiles** and **24 versus 21 DSPs**, offset by increased LUT/FF usage and worse routed WNS (**−2.170ns versus −3.718ns** at a 12ns requested period). Both physical implementations missed timing and had DRC warnings, so neither qualified for strict clean-DRC physical evidence, and **physical compute and end-to-end latency remain unknown**. The work provides an auditable bounded compiler contract and decision interface with explicit negative performance findings. Distinctiveness relative to established compositional/dataflow compilers is under independent evaluation; no firstness, general FPGA speedup, achieved frequency, or learned-policy benefit is claimed.

## 1. Introduction

Conventional accelerator architecture exploration can conflate a functional graph, design decisions, generated source, and downstream hardware synthesis feedback. A search algorithm may propose a source-illegal action or one whose tensor ownership and last-reader assumptions cannot be certified. Treating an action as simply a numeric knob vector also obscures which operator-specific architectural choice changed a tensor's source access obligations.

The proposed representation keeps an immutable typed functional model separate from per-operator architecture requests and explicit supported tensor exchange policies. Unlike an opaque reward-only environment, every proposed complete-design action traverses the same source compiler and returns a certificate or a concrete failure. This is a compiler interface suitable for *future* learning-based agents; its suitability does not establish an agent performance benefit.

Two research questions guide the paper:

- **C1:** Within well-specified static source-event domains, can operator-local architecture decisions and typed tensor-element ownership be verified against actual source address usage, preventing invalid early reads and reuse before all readers are finished?
- **C2:** Can those exact, already implemented compiler decisions be exposed as a deterministic policy-independent transition, a three-valued finite action mask, and a weight-private variable-size graph representation, while preserving UNKNOWN rather than inventing hardware legality?

Neither standalone CNN coverage nor the existence of spatial/temporal representations is a priority claim.

## 2. State, actions, and evidence tiers

A complete source design state is `S = (F, A, X, L)`: typed functional graph `F`, one complete architecture request `A[o]` for every operator, a domain-supported tensor exchange/layout choice `X`, and explicit static target limits `L`. An action `a = (scope, target, field, value)` edits one supported choice. Its outcome `T(S,a)` is `REJECTED`, `UNKNOWN_E0`, or `BOUNDED_CERTIFIED_E0` after the **ordinary compiler** executes. Rejection leaves `S` unchanged; acceptance produces a new state and a source-bound receipt.

`UNKNOWN_E0` means the source emitter admitted a design but the supported bounded event/ownership certificate is not proven. The mask bit `M(S,a)=1` only when `T(S,a)` is `BOUNDED_CERTIFIED_E0`. This is an admissibility result for one proposed complete source state, *not* a Vitis-qualified resource/timing prediction.

We retain evidence tiers separately: E0 bounded model/source event contracts; E1 emitted C++ and, where explicitly executed, GCC source-level numerical/trace conformance; E2 actual source-attested Vitis CSim; E3 original CSynth cycle/resource **estimates** passing strict effective clock qualification; E4 a separately qualified clean-DRC routed implementation. A successful E0/E1 does not imply E3/E4.

## 3. C1: finite value ownership and source-event semantics

### 3.1 Source event model

For supported static MatMul source models, a finite trace contains `CONTRIBUTE` events, `COMMIT` for completed tensor elements, and consumer `READ` events. A typed value key identifies a tensor element; the producer maps reductions into spatial owners derived from its effective `Phi`, and consumers carry corresponding bounded source-issue/owner metadata.

The event checker rejects (a) a read before all required contributions and commit, (b) a duplicate or invalid contribution/reader, (c) a wrong source `Phi` owner, (d) a commit before required input reads, and (e) a scratch slot overwrite before all of the value's required readers have consumed it. A distinct issue-order check validates conformance to the claimed `Phi` source issue order. **An otherwise readiness-safe read permutation may still violate the claimed issue order.**

### 3.2 Scope of the bounded safety argument

For a supported finite event population `E`, a verifier that accepts only traces whose every prefix satisfies its producer-completeness, committed-read and live-slot invariants establishes those properties **within its model**. This is a direct invariant of the finite verifier's transition checks. It does not prove external RTL behavior, bank-port scheduling, clock-cycle timing, general arbitrary graphs or all-input numerical correctness.

Likewise, source scratch allocation can be checked for a **fixed total order** by enumerating the intervals from commit to final required read and disallowing overlap on identical unit slots. The existing independent small-population exhaustive color oracle provides a separate bound on how many source unit slots are necessary for that fixed order. This is not an original interval-coloring theorem nor a physical FPGA BRAM-optimality result.

### 3.3 Current implementation and independent tests

Existing R2f/R2h source compilers generate per-element fanout traces and actual C++ load/store address witnesses; R2r selected generated-source mutants test adversarial cases. R2q original source memory comparisons are sensitive to whether last-use is measured at committed values or at hypothetical first in-place MAC. Both accounting assumptions must be disclosed, including the zero-reduction result for the hypothetical first-MAC convention.

**Bounded C1 original evidence table (different frozen cohorts, not pooled independent trials):**

| Observation | Original evaluated population | Observed outcome | Claim boundary |
|---|---:|---|---|
| Independent fixed-source-order unit-slot exhaustive oracle | 5 small source cases | 5/5 agreement | Source-slot minimum under fixed order only |
| Selected semantic mutations | 25 (five types × five cases) | 25/25 rejected | Only these prespecified negative controls |
| Independently coded finite event interpreter vs production verifier | 133 trace variants, five source cases | 0/133 status disagreements | Both share admitted static model; no general proof |
| Generated-source selected mutation study (separate R2r population) | 216 selected source mutations | 216/216 detected | Selected source mutations, not all possible incorrect C++ or RTL |

**Workstation-verified R3e study (original source revision \`fc40369f9493d93cc78c4886ae2e32c5d1f64093\`):** exactly five tiny bounded source cases across 2–4 readers and a changed reader-PE control, with an *independent finite exhaustive minimum-slot oracle* and five targeted negative mutations per case: early consumer read, producer commit without its reduction, wrong `Phi` owner, source-safe but wrong `Phi` issue order, and source overwrite of a simultaneously live value. Original planned population: **5 source cases × 5 mutations = 25 mutation trials**, with no allowed post-hoc exclusions. A rejected mutation must report its expected semantic reason rather than any incidental compiler crash.

### 3.4 Explicit bounded operational soundness obligations

The [finite semantics and proof sketch](C1_FINITE_EVENT_SEMANTICS_AND_PROOF_SKETCH_20261104.md) specifies prefix-operational states for producer contributions, committed values, seen consumer reads and occupied source slots. Conditional propositions P1–P3 express no premature consumer reads, no early slot overwrite, and standard fixed-order interval-packing capacity; P4 separates source-value safety from `Phi` issue-order conformance. The argument holds **only if the bounded graph's expected producer/consumer obligations have been enumerated correctly**. The proof sketch is not mechanically verified and does not establish RTL refinement or arbitration of physical memory ports.

The R3g differential checker implements the same bounded problem using a separately coded operational interpreter and forward-rank issue enumerator; **133/133 original tested trace variants agreed with the compiler (0 disagreements)**, with the author-reported source SHA and read-only receipt in [the frozen R3g/R3h checkpoint](R3G_R3H_ORIGINAL_C1_C2_WORKSTATION_RESULTS_20261104.md). This is finite observed agreement, **not** independent specification completeness because both interpreters share the admitted static model.

### 3.5 Matched issue-order predicate necessity control

To isolate the contribution of the operator-local `Phi` issue-order predicate from value readiness and scratch-lifetime safety, we evaluated **24 bounded static MatMul fanout configurations** (2–4 readers, four held-out static shapes, two operator-local mapping policies). Each had a canonical valid trace and eight prespecified adverse traces, giving **216 source trace variants**. Of the 192 deliberately *full-contract-nonconformant* variants, **24** were source-value-safe but violated the specifically declared `Phi` source READ order; the other **168** violated source-value/ownership safety. **The first 24 are not thereby functionally incorrect**: they may be legitimate alternative source schedules, rejected only because they violate the particular operator-local schedule being certified. Using the same traces and candidate policies:

| Predicate-derived admission | Full-contract-conformant controls accepted (of 24) | Selected full-contract-nonconformant controls accepted (of 192) | Total accepted (of 216) |
|---|---:|---:|---:|
| Value readiness, source ownership and scratch lifetime **without `Phi` issue-order check** | 24 | **24** | 48 |
| Above plus **operator-local `Phi` READ issue-order check** | 24 | **0** | 24 |

The author ran **43/43 focused source-only tests** and the original create-only ablation report, recording exact source revision `205364f5aea74170a5d4b0bb8bdaaf6a338d5e38`. Its 216-row CSV and summary file passed a separate SHA-256 inventory/denominator consistency check; **no Vitis/Vivado calls** were made. This observation demonstrates that a **value-safe trace need not conform to a chosen operator-local issue order** for the tested finite domain; it is a separation between *functional/data-lifetime safety* and *declared schedule conformance*, not evidence that all alternative safe orders are bugs.

**Design and causal inference:** The original R3j study evaluates `reference_event_oracle`, `verify_fanout_events` and `verify_fanout_issue` for every trace, rejecting disagreements between oracle and compiler predicates. The ablation exporter then **recombines these two existing independently checked verdicts**: baseline admission = `safe`; full admission = `safe AND phi_issue_conformant`. These are matched, explicitly selected negative controls and a **post-hoc predicate necessity comparison**, **not two independently executed compiler builds**, a randomized causal experiment, proof of correctness on arbitrary graphs, an external compiler baseline or any hardware performance measurement. The selected controls and acceptance assertions were fixed by source; the outcome should not be treated as an unbiased estimate of real-world error prevalence. Hardware scientific value further requires a verified link from the source order being certified to the emitted implementation's actual issue behavior and memory/port/timing constraints; **that link remains a research gap**. [Source-pinned evidence and limitations](FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md).

## 4. C2: policy-independent source search interface

The R2n/R2o/R2p compiler decision interfaces unify two bounded source domains: sequential MatMul chains and 2–4-reader fanout. The graph observation includes typed operators/tensors, shapes, supported mapping probes and source domain metadata, without raw tensor constants or prior HLS reward. The policy may come from a person, enumeration, heuristic or future agent; the compiler does not inspect that policy's identity.

For an explicit finite candidate catalog `C`, the mask calls the actual compiler once for every proposed complete-design action. Therefore compiler-mask agreement with independently invoked `T(S,a)` is a **deterministic implementation invariant**, not evidence that expensive source compilation has been eliminated. The source-only mask must retain rejected and uncertified outcomes; an E1 GCC execution, E3 qualified HLS result, and physical E4 remain different gates.

### 4.1 Causal ablation controls

- **Unmasked catalog:** every candidate is submitted to the same compiler for acceptance/failure feedback.
- **Source-admitted filter:** candidates whose source is accepted may include unknown E0 ownership.
- **E0-certified filter:** only candidates meeting the source event/ownership contract are eligible.
- **Explicit screening cost:** every compiler call used to build any mask is counted; no improvement in total cost or RL performance is inferred simply from removing invalid proposals.

All arms use the *same candidates and the same compiler*. Report qualified, unknown, rejected, elapsed CPU checking time and full denominators, not just success fractions. No action is considered hardware feasible from E0 alone.

### 4.2 Prepared finite experiments

**R3d — workstation-verified two-domain control:** six fixed candidate actions (3 E0-certified, 2 rejected, 1 E0 unknown) across two compiler source families, with an independently specified *restricted* expected-status catalog, direct compiler replay and a deterministic rename/order-invariant hash of publicly visible typed graph features. The hash is a feature fingerprint, **not** a general graph-isomorphism decision procedure.

**R3f — workstation-verified variable-size coverage:** six additional source graph cohorts: MatMul chains of depths 2, 3 and 4, and fanout of degrees 2, 3 and 4. Exactly **eight predeclared actions per graph (48 total)** undergo compiler-based mask and direct replay, with invalid/no-op controls. This evaluates action classification and graph-size variation, not trained-agent sample efficiency. The author's workstation passed **42/42 focused and 1,705/1,705 full Python tests**, then generated and independently audited R3d/R3e/R3f create-only original source reports on revision \`fc40369f9493d93cc78c4886ae2e32c5d1f64093\`. R3f classified **21/48 certified, 24/48 rejected, 3/48 UNKNOWN_E0**, with the complete denominator preserved; R3e recorded **5/5 independent finite slot-oracle agreements and 25/25 targeted mutant rejections**. R3g and R3h were subsequently tested on the author's workstation at source revision `3309076814714e61906363e21e3d50b932519f9d`: **64/64 focused, 1,751/1,751 full regressions**, 133 trace variants with 0 independent semantic-oracle disagreements, and 48 compiled-once catalog actions with 0 extra selected-source compiler invocations. All were independently re-audited as saved original receipts.

### 4.2.1 Original C2 mask population and cost

| Predeclared source observation | Certified E0 | Rejected | Unknown E0 | Compiler source transitions |
|---|---:|---:|---:|---:|
| R3d, six complete designs | 3 | 2 | 1 | 6 classified actions |
| R3f, six graphs × eight candidates | 21 | 24 | 3 | 48 classified actions |
| R3h, selection from precompiled R3f-size catalogs | Only previously certified states eligible | Not selected | Not selected | **48** upfront catalog transitions; **0 additional** transitions to select a cached certified state |

R3d, R3f and R3h were **distinct original source experiments** and their rows are not additive independent samples. The three-valued mask faithfully exposes the compiler's bounded transition classification, but its full-catalog CPU cost is not zero; an agent that wants only one candidate could call the compiler directly with lower total checking work. No unmeasured search speedup or training advantage is inferred.

### 4.3 Compiled-once source-certificate consumption

The new C2 `CompiledActionCatalog` preserves each **original, typed, source-verified next state** after the ordinary compiler has evaluated its candidate. A later proposer may select an E0-certified in-process catalog entry without a second source compiler run, while a rejected or E0-unknown candidate is never silently promoted. The R3h controlled study tests this property on the same six R3f graph families and eight actions each. This is **an API/call-count ablation**, not optimized exploration: producing a catalog of eight candidates still requires eight compiler transitions and may be more expensive than submitting one blind candidate. The R3h author-run original campaign verified **48 candidate source transitions across six graphs** and **zero additional transitions to select an already certified cached state**. It is an API-level implementation result; mask construction still requires all 48 source transitions in that cohort. Original read-only audit receipt and source revision are in [the frozen checkpoint](R3G_R3H_ORIGINAL_C1_C2_WORKSTATION_RESULTS_20261104.md).

## 5. Experimental design and reproducibility

Source and native original measurements use explicit Git source revisions, original generated-source SHA, input checks and independent audit receipts. R3d/R3e/R3f create only new output folders and never overwrite the earlier frozen studies. Proposed CPU experiments require a clean tracked source checkout. Host source validation, native HLS and physical placement/routing are separate phases.

All campaigns preserve the exact denominators and every negative result. The **workstation-verified R3e** finite oracle is an independent brute-force reference for unit slots under fixed source order; it is not a foreign compiler baseline. The R3d manually specified six expected statuses are not an exhaustive independent semantic oracle. Future completion requires cross-artifact capability tests and a larger domain-specific action-space sweep to establish generalizability.

### 5.1 Evidence tiers and meaningful latency

All results are typed by evaluation level: E0 finite static source event/certificate checks; E1 independently executed GCC C++ and numerical/source traces where recorded; E2 original Vitis C simulation; E3 **qualified HLS-estimated** worst cycles/resources/effective timing; and strict E4 **timing-closed, clean-DRC** Vivado OOC routing. Raw routed mapping and WNS from warning-bearing or timing-failing designs are retained as **descriptive physical observations**, not strict E4. The R4 checkpoint report-only follow-up is a secondary forensic confirmation, not a rerun of physical implementation.

Physical compute latency requires compatible first-result cycle counts and a **physically feasible, qualified clock period** on the same design, `latency_compute = cycles × period`; pipelined throughput additionally requires a measured or qualified initiation interval. For streamed/batched kernels, first-output latency is not reciprocal steady-state throughput. End-to-end board inference latency further includes DMA, DDR, buffering, host/device synchronization and any clock-domain transfer costs. An E3 HLS cycle count times an **unsatisfied requested** 12ns period is NOT qualified physical runtime. In the R4 original pair, both routed WNS values are negative, so real compute latency, achieved Fmax, real throughput, power, and board end-to-end latency are **UNKNOWN**. All trade-off tables must show both favorable and unfavorable axes and these missing outcomes rather than turning an unqualified quantity into zero or a presumed clock.

### 5.2 Non-independence and original study populations

The source checker, oracle/negative controls, search-interface and hardware studies use different frozen source revisions, candidate catalogs and numerical domains. We report the **complete numerator/denominator within each original cohort**, not an artificial pooled accuracy or a random population mean. Direct C2 action replay and action-mask classification use the same ordinary compiler: agreement is partly a designed implementation property; the R3h cache result saves a post-catalog selection call but not its complete catalog construction cost. Hardware paired comparisons are not automatically evidence of causal memory or timing improvement unless source schedule, precision, clock, device and executable source are matched and numerical/timing gates are satisfied. Raw Vivado OOC values do not demonstrate board throughput.

## 6. Existing source and native results

Earlier original bounded source/contract experiments report selected generated-source mutation detection, finite multi-reader/last-reader event validation, and GCC-generated C++ numerical/address-trace checks. The particular verified original R2r study identified **216/216 selected mutations**, which should be stated as selected-family detection only. Original R2q studies tested **27 `Phi` contrasts**; **18** showed source scratch reductions under committed-value accounting, with **9** ties, but no reductions under a competing hypothetical first-MAC in-place accounting convention. This sensitivity is an important negative result.

The completed original R2z/R3a full-network evaluation uses six fixed, source-attributed CNN/Dense variants: **6/6 original E2 qualified**, **4/6 strict E3**, two genuine effective-clock E3 rejections, and **2/3 original matched E3 pairs qualified**. The R3c read-only results export passed **36/36 focused and 1,663/1,663 full Python tests** and created `build/r3c_cnn_original_publication_tables_v1` on the author's workstation with **zero new Vitis calls**.

| Original CNN case pair | HLS worst-case cycles: scalar → independent `Phi` | Estimated BRAM18K: scalar → independent | Strict paired E3 |
|---|---:|---|---|
| CNN + Pool + Dense (`000→001`) | **624→911** | **1→1** | yes |
| Strided CNN + Dense (`002→003`) | **NULL** | **NULL** | no; both fail effective 9 ns scheduling budget |
| Two Conv + Softmax (`004→005`) | **2365→5436** | **0→0** | yes |

Both E3-qualified nonuniform mappings increase estimated DSP/FF/LUT usage and worst HLS cycles, with no estimated BRAM reduction. HLS resource estimates are **not** routed FPGA measurements. These results show bounded whole-network support but **do not** demonstrate that C1/C2 is fully supported across those CNN networks, nor that independent mappings are universally detrimental.

### 6.1 Added source-only validation: R3g and R3h

The author completed two new independent, create-only source studies after R3d–R3f at original code revision `3309076814714e61906363e21e3d50b932519f9d`. The regression suite passed **1,751/1,751**. **R3g compared 133 original finite typed event sequences** using a distinct reference interpreter and the production source verifier with **zero observed status disagreements**; this is limited to the five tested static model configurations and shared assumed source semantics. **R3h constructed 48/48 source-compiled typed action candidates** across six graph families and demonstrated **zero additional compiler transitions** to select a cached E0-certified next state. The unavoidable upfront mask cost of 48 compiler transitions is explicitly retained. The original R3g read-only receipt SHA-256 is `847efcd501860bb5e58757766a8ed64440e14f558dac7c666f42096e96ab542d`; R3h is `510746a5b78cc4b0b4c247b15d24510081f4685a2dd0cd2db064edec3f7dc72f`. See [frozen revision-pinned source results](R3G_R3H_ORIGINAL_C1_C2_WORKSTATION_RESULTS_20261104.md).

These results increase **compiler-internal confidence**; they do not establish statistical advantage, E1 source-binary equivalence for all inputs, hardware performance or a new result against any external compiler.

### 6.2 Controlled storage ablation: HLS cycles versus routed resource trade-off

A separate, frozen 32/6 numeric-format source experiment holds the source-level reader/issue schedule fixed while changing intermediate scratch materialization between `full_storage` and `row_storage`. Both R4k variants passed real host/GCC E1 conformance; the separately executed original R4l Vitis cases passed fixed-point CSim E2 and strict HLS synthesis E3. Their HLS worst-case cycles decreased **22,954→22,602** (−352, approximately −1.53%), and HLS BRAM18K estimates tied **13→13**; HLS-estimated DSP/LUT/FF were lower for row storage. Their separately recorded **HLS-estimated clock periods** were **9.857ns versus 9.500ns**, not achieved physical periods. If we multiply each separately estimated period by its corresponding HLS worst-cycle estimate, we obtain **226.258µs versus 214.719µs** (a derived **11.539µs** reduction). Using a common **12ns requested target** instead gives **275.448µs versus 271.224µs** (a derived **4.224µs** reduction). Both comparisons are clearly **E3-only design-time proxies under different clock assumptions**, NOT measured/physically feasible inference latencies: both later Vivado implementations failed the requested 12ns target. Neither proxy establishes real hardware runtime. A matching original Vivado OOC physical-routing pair subsequently reported:

| Original routed OOC result | Full storage | Row storage | Row − full |
|---|---:|---:|---:|
| BRAM tiles | 7 | 4 | −3 (−42.9%) |
| DSP | 24 | 21 | −3 (−12.5%) |
| LUT | 1,282 | 1,710 | +428 (+33.4%) |
| FF | 883 | 1,022 | +139 (+15.7%) |
| WNS at **requested 12ns** | −2.170ns | −3.718ns | −1.548ns (worse) |
| DRC warnings (zero critical) | 74 | 71 | −3 |
| Strict clean-DRC timing-qualified E4 | **NO** | **NO** | No qualified comparison |
| Physical inference latency / achieved Fmax | **UNKNOWN** | **UNKNOWN** | Not a physical latency result |

Both implementations were fully routed, and the original source/ROM/native receipt, four report files per arm and read-only mapped-resource summaries were SHA-attested. The source-level matched issue order does not imply equivalent synthesized pipeline or identical RTL scheduling. **Both fail the requested clock constraint and clean DRC**, so this result is an informative **descriptive mapping trade-off**, not a better or faster final implementation. It illustrates why a compiler search interface must treat BRAM, logic, timing, latency and numerical correctness as separate constrained objectives. The later one-time report-only checkpoint follow-up independently reproduced original WNS/DRC counts but yielded no complete path witnesses through the current lexical parser; no critical path cause was established. [R4l source-matched HLS](R4L_SCHEDULE_MATCHED_STORAGE_NATIVE_GATE_20261106.md), [source-attested routed ledger](R4R_SIGNED_OOC_TRADEOFF_LEDGER_20261106.md), [R4u report-only qualification](R4U_COPIED_DCP_REPORT_ONLY_NATIVE_GATE_20261106.md).

## 7. Closest work and scientific distinction to establish

The initial pinned Allo artifact review has since confirmed `reuse_at`, `partition` and `buffer_at` memory primitives, use-def and schedule composition, and real **positive and negative** `allo.verify` equivalence tests. Therefore, we do not treat "memory reuse", "composition" or "source verifier" as absent from Allo. A **matched artifact-executed test of all-reader release plus per-operator source issue order** is required before any exclusivity statement. See the [expanded pinned source analysis](C1_C2_PINNED_PRIOR_ART_RECONNAISSANCE_20261103.md).



**Allo (PLDI 2024)** already provides bottom-up type-safe composition of compute, memory, communication and datatype customizations; its public `allo/verify.py` includes schedule/source verification using PAST. **StreamTensor (MICRO 2025)** introduces stream-layout tensor types for fusion, buffer allocation and memory optimization. **Filament (PLDI 2023)** encodes hardware timing/structural constraints with timeline types. **CODO (ISCA 2026)** checks FPGA dataflow violations and optimizes communication/scheduling. **ST-Flow (PACT 2026)** compiles spatial-temporal dataflow. Their substantial overlapping contributions rule out generic claims of the *first typed compositional FPGA IR*, *first temporal source checker* or *first optimization-ready graph*. Public references and pinned revision audit are recorded in [the independent prior-art reconnaissance](C1_C2_PINNED_PRIOR_ART_RECONNAISSANCE_20261103.md).

The still-unresolved novelty claim is **a concrete, testable combination of bounded operator-local source issue semantics, multi-reader last-use scratch safety and an exactly same-compiler three-valued action interface**. Establish that distinction through executed, pinned competitor artifacts with matching positive and negative source designs and explicit semantics. Do not infer absence of a rival feature merely from a short abstract or one file.

### 7.1 Prior-art comparison with measured and unknown dimensions

| Related system | Established overlapping capability | What our current evidence supports (and does not) |
|---|---|---|
| Allo (PLDI 2024) | Composable compute, memory, communication and datatype schedule customization, plus source/schedule equivalence checks | FPGAI checks its **particular bounded multi-reader value-release and operator issue-order relation**; a directly executed semantically matched Allo positive/negative comparison has **not** been completed, so difference/absence in Allo is UNKNOWN |
| CODO (ISCA 2026) | Fine- and coarse-grained dataflow violation checking, data-movement optimization, scheduling and published latency comparisons | A bounded event/ownership contract plus policy-neutral action mask is the *proposed* focus, not a demonstrated faster accelerator or a claim to invent dataflow violation checking |
| ST-Flow (PACT 2026) | Spatial/temporal/hybrid dataflow compilation and resource reuse with explicit design-space exploration | FPGAI has only bounded static source actions and mixed HLS/routed observations; cross-compiler resource/latency superiority is not shown |

This classification draws on the authors' public papers and pinned source reconnaissance, not a matched executable capability experiment. To claim a truly **novel semantic obligation**, define the exact trace/reuse counterexample and show that the same workload and property can be represented and tested (or document a concrete expressivity limitation) in the pinned competitor artifact. Missing imports, tool setup failures or unsupported API forms are **not** evidence of a competitor's incorrect verification.

### 7.2 Original seeded C1 and fair C2 source-only evidence — verified

At exact source revision **`637f2745721b32d6f5a7e02c4659b0623bc9eee7`**, the author ran **72/72** focused tests (2.28s) and executed two independent, create-only source-only studies. Both preserved immutable original report SHA-256 identities and passed separate read-only SHA/population audits. Neither ran Vitis/Vivado.

**C1, in-family held-out metamorphic:** Three predeclared seeds, 2–4 fanout readers, scalar/mixed-`Phi` operator mappings and three distinct newly sampled held-out shapes per group yielded **54 unique static configurations and 486 prespecified typed source-event variants**. The independent reference-event oracle and production source value/issue-order checkers had **zero observed classification disagreements**. Each canonical trace passed the full contract; each deliberately value-safe alternative read ordering was identified as conformant to value-safety but nonconformant to the *declared* `Phi` source issue schedule; selected other negative controls were rejected by the source value/ownership predicate. The result extends *shape and mapping coverage within the same admitted MatMul fanout topology*, not independent graph-family generality or proof of generated C++/RTL semantics. Report SHA-256: `8fa3848fcb0993e44a83391f595861943d75e3f09b669bd93b1b27e0222f3978`.

**C2, matched source-screening cost dataset:** On the **same six existing R3f graph cohorts**, direct-per-query source recompilation, on-demand lazy caching and eager eight-candidate catalog construction produced **324 measured method observations** under fixed three seeds, request counts 1/8/32 and two certified-request streams. Every compared source decision identity matched, including resulting source/C++ and certificate hashes. An original **48 source-compiler transition preflight** to identify certified actions was **excluded from all three measured method runtimes** and must not be hidden. SHA-256: `be2adaa4acf30196a2ccb17114f2caa0af3e6679dc2685872e27cb086ca86ec1`. **The author's original report summary did not show the per-method timing values**, so the manuscript makes **no CPU speedup claim** pending read-only paired cost interpretation. A separate new source-only SHA-pinned analysis script can summarize this original file without rerunning it.

**Pre-existing theoretical caveat:** For one fixed eight-action parent state, eager catalog construction always requires eight source-compiler transitions, whereas a correct lazy cache requires only the number of distinct actions queried (at most eight). Eager cannot claim fewer source compiler calls than lazy for this experimental setup; any CPU wall-time differences are empirical and workload-specific. These source-only results do not measure hardware latency, learned policy search efficiency or an independent competitor capability.

See [falsification-first original protocol and complete provenance](SCIENTIFIC_STRENGTHENING_C1_C2_PROTOCOL_20261106.md) and the [evidence-to-claim publication gate](FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md).

### 7.3 Pending direct generated-source C1 refinement (not measured)

Earlier, source-pinned R2r results exercised actual emitted C++ and rejected **216/216** selected generated-source mutations, on a different six-graph bounded fanout cohort, with distinct independent shadow-memory and numeric/source checks. This E1 evidence is more implementation-grounded than Python event-predicate agreement but does not prove all-input behavioral refinement or a clock-aware RTL schedule. To directly connect the new 54-profile C1 study to emitted source, a separate two-layout-per-profile **actual GCC executable** protocol has been implemented, pairing each source event SHA with the author's immutable original E0 study. Its prospective 108 GCC runs have **NOT** been executed on the workstation at this source revision; do not promote the planned denominator or a successful preliminary synthetic test into observations. See [E0-to-E1 original source bridge protocol](C1_E0_E1_HELDOUT_CPP_BRIDGE_GATE_20261106.md).

### 7.4 External scientific novelty decision: output equivalence versus requested temporal contract

The [adversarial claim matrix](SCIENTIFIC_REVIEWER_ADVERSARIAL_CLAIM_MATRIX_20261106.md) preregisters exact positive/negative semantic controls: a canonical valid source; a value-safe reader reorder that preserves functionality but disagrees with the originally declared operator-local `Phi` schedule; early read/overwrite hazards; invalid mapping witness; a newly declared alternative valid schedule; and an unsupported graph/precision. The **scientifically interesting distinction is not that equivalent scheduling can produce identical results or that incorrect FIFO access orders can be caught**: Allo's PLDI 2024 Section 5.2/Figure 6 already detects a **producer/consumer MatMul FIFO access-order mismatch** (https://www.csl.cornell.edu/~zhiruz/pdfs/allo-pldi2024.pdf). Allo also verifies equivalent but structurally different schedules, and its official `allo.verify` tutorial provides both positive reordered/buffered GEMM and negative incorrect reduction examples (https://cornell-zhang.github.io/allo/gallery/dive_05_verifier.html). The testable question is whether a compiler can preserve a *requested* source event/order/memory-ownership certificate with actual emitted implementation witnesses alongside functional equivalence. This external semantic comparison has **not** been executed. CODO (ISCA 2026, https://arxiv.org/abs/2604.12618) independently addresses coarse/fine dataflow violations and performance-aware mapping, so no exclusivity or firstness claim is warranted.

**Publication decision rule:** If a representable Allo/CODO program already supplies the same typed source/value-read/order contract, explicitly record overlap and revise the paper. If C2's eager catalog is weaker than a lazy cache and no independently useful policy-neutral certificate benefit appears, make C2 **supporting engineering infrastructure** and present a single scientifically substantiated C1 contribution rather than falsely balancing the title around two breakthroughs. An unresolved external benchmark is an open P0 blocker, not evidence of competitor incapability.

## 8. Threats to validity

**Semantic and specification validity.** Finite event admission proves transition invariants only under correctly enumerated supported static reductions and reader obligations. The separate interpreter shares the admitted model; 133 matching traces are not a formal proof of implementation completeness. Fixed-order scratch oracle minima are not global schedule-optimal FPGA storage. Actual generated C++ access refinement is tested only on bounded selected programs/mutations, and source correctness does not prove numerical agreement on every input or hardware memory-port legality.

**Action-interface utility and cost.** A policy-independent three-valued action mask is built using the same source compiler as direct transitions; its correctness is partly by construction. The 48-action precomputation requires 48 compiler evaluations, despite zero additional calls for later cached selection. A new author-verified 324-row direct/lazy/eager original source-only dataset exists, but its per-method timing metrics have **not yet been independently summarized**. Its 48 preflight compiler transitions were excluded from all methods. The six source graphs reuse the R3f cohort rather than independently held-out applications. There is no trained RL agent or demonstrated search-cost, sample-efficiency, physical performance, or cross-model transfer gain. Fair blind-policy acceptance, independent source-legality, and equal-budget comparison remain necessary for stronger utility claims.

**Hardware representation and latency.** Supported source-level chains/fanouts do not establish general CNN tensor-lifetime safety or all operator types. Native HLS estimates are not realized physical timing, power, or BRAM mapping. The single R4 pair's **negative** routed WNS and DRC warnings preclude a strict E4 comparison or achieved physical latency/throughput; same-source-level issue order does not ensure equal synthesized schedule. A smaller BRAM count can coexist with worse LUT, FF and WNS. No design-wide Pareto superiority or board-level end-to-end inference latency follows.

**Novelty, external validity and artifact rights.** Allo already provides composable schedules, memory and source equivalence, and CODO/ST-Flow address overlapping dataflow correctness and optimization. The exact C1/C2 joint combination has not yet passed a matched independently executed competitor test, nor independent novelty review. Static bounded source cases, small action catalogs, and single-point hardware runs cannot establish broad generalization. Any public artifact or vendor log must separately satisfy ownership, licensing and rights-cleared distribution review.

## 9. Conclusion and remaining scientific gates

We have implemented and independently exercised bounded source-event ownership and operator issue-order checks (C1) and a policy-independent, three-valued complete-design compiler action interface (C2), including deliberately adverse and unknown outcomes. These are **finite compiler semantics and API observations**, not full formal proof, trained-agent advantage, unrestricted graph coverage, or FPGA performance superiority. The qualified HLS experiments include negative results; the source-matched routed memory experiment demonstrates a BRAM/DSP trade-off alongside worse logic and timing without establishing actual physical compute latency.

To make the manuscript **submission-ready**, the remaining priority is an independent matched competitor capability/semantics analysis (especially Allo), explicit generated-source refinement limitations, a CPU-cost-fair C2 screening comparison or narrowed utility claim, and an independently audited artifact/manuscript release with exact original revisions, denominators and licensing approvals. Further R4 Vivado path forensics should not displace these scientific tasks. The submission decision remains **MAJOR REVISION / NOT SUBMISSION READY**. No new native study, RL training or board experiment is implied by this manuscript revision. [Canonical evidence-to-claim and submission gate](FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md).

### 9.1 Unfinished post-evaluation source studies and final evidence export

The separate R3j held-out experiment is designed to evaluate **24 prespecified combinations of fanout degree, shape and independent `Phi` mappings** over **216 finite original event traces**, including unsafe address aliasing and source-safe/issue-order-invalid controls. R3k is a matched **12 source-model-pair / 192 typed-action** intervention on explicit source parallelism limits. The current manuscript does **not** import numerical results from these studies without separately verifying their original source-revision-specific receipts and audit records. An implementation, planned population or read-only index is not sufficient to claim an observed trial; only fully audited original runs may enter the results tables.

The new R3l source-only publication release is designed to re-audit six separate original receipts (R3c through R3h), verify full source revision identities and R3c-derived file digests, forbid imputed metrics for failed E3 pairs, and create provenance tables in a **new output folder**. Its mandatory `submission_ready=false` is intentional: paper novelty against external compiler artifacts remains unresolved. Optional R3j/R3k can be included only when their original experiment folders exist.

## References and reproducibility index

1. Allo, PLDI 2024: https://doi.org/10.1145/3656401; https://github.com/cornell-zhang/allo.
2. StreamTensor, MICRO 2025: https://doi.org/10.1145/3725843.3762817; https://arxiv.org/abs/2509.13694.
3. Filament, PLDI 2023: https://doi.org/10.1145/3591234.
4. CODO, ISCA 2026: https://doi.org/10.1109/ISCA66397.2026.00018.
5. ST-Flow, PACT 2026: https://doi.org/10.1145/3838684.3846865; https://github.com/UCLA-VAST/ST-Flow (artifact tag `PACT26`).
6. [Research-scope/acceptance gates](C1_C2_COMPILER_IR_FINAL_SCOPE_AND_ACCEPTANCE_GATES_20261103.md).
7. [Original R3c six-case HLS data and read-only exporter](R3C_COMPLETE_CNN_HLS_PUBLICATION_RESULTS_20261103.md).
8. [Pinned external-artifact reconnaissance](C1_C2_PINNED_PRIOR_ART_RECONNAISSANCE_20261103.md).
9. [Original R3g/R3h source results, SHA digests and scientific limits](R3G_R3H_ORIGINAL_C1_C2_WORKSTATION_RESULTS_20261104.md).
10. [Formal bounded event semantics](C1_FINITE_EVENT_SEMANTICS_AND_PROOF_SKETCH_20261104.md).
