# Contract-Aware Model IR for Policy-Independent FPGA Architecture Exploration

**Scientific working manuscript — 3 November 2026. Proposed two-contribution compiler paper, not submission-ready.**

**Scope decision.** This paper studies **C1**, a bounded compositional, operator-architecture-aware source-event contract, and **C2**, a policy-independent, learning-compatible compiler search interface. Training a GNN/RL agent, reward optimization, HLS learning curves, and performance advantages for trained agents are outside this paper and reserved for a separate study.

## Abstract

Optimizing heterogeneous FPGA accelerators requires representing not only operators and tensors, but also architecture-specific execution decisions and the lifetime conditions under which produced values may be read and reused. We investigate a bounded Model IR that connects static functional graphs, complete operator-local architectural requests, typed source-level events, and compiler-generated source identities. A finite contract checker distinguishes completion of producer reductions, committed tensor-element availability, multi-reader last use, legal scratch reuse, and conformance to supported source issue orders. We expose those compiler decisions through immutable complete-design states, typed action transitions, explicit accepted/rejected/unknown outcomes, and weight-private typed graph observations, without depending on the identity of an optimization policy.

The evidence comprises original bounded source-event/counterexample and GCC-generated-source studies, together with original source-attested Vitis HLS experiments for complete CNN–Dense chains. In the completed six-case neural cohort, all six numerical Vitis C-simulations qualified and four met the strict HLS timing/resource qualification; two source-attributed clock failures were retained. Among two HLS-qualified two-arm comparisons, independently selected operator mappings increased HLS cycle and logic-resource estimates without reducing estimated BRAM. These negative results limit claims about hardware quality but do not negate the representational or source-legality research question. Additional original finite experiments are now workstation-verified: 6/6 complete-design actions classified in R3d (3 E0-certified, 2 rejected, 1 source-admitted but E0 unknown); R3e agreed with a separate exhaustive fixed-order scratch oracle on 5/5 small cases and rejected 25/25 selected mutations; and R3f classified 48/48 actions over six variable-size graphs (21 certified, 24 rejected, 3 unknown). These are bounded source decisions, not a trained-RL or hardware benefit. The later R3g independent bounded-event oracle compared **133/133** source traces with **zero disagreements**, and R3h compiled **48/48** complete-design candidates with **zero additional compiler calls for selecting cached certified states**. Both passed read-only audit on their distinct source revision. These results describe compiler source semantics, not learned agent performance. The work makes no claims of improved trained-agent performance, physical FPGA QoR or general formal proof; its scientific distinctness relative to competing typed hardware/dataflow compilers remains to be independently established.

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

**Workstation-verified R3e study (original source revision \`fc40369f9493d93cc78c4886ae2e32c5d1f64093\`):** exactly five tiny bounded source cases across 2–4 readers and a changed reader-PE control, with an *independent finite exhaustive minimum-slot oracle* and five targeted negative mutations per case: early consumer read, producer commit without its reduction, wrong `Phi` owner, source-safe but wrong `Phi` issue order, and source overwrite of a simultaneously live value. Original planned population: **5 source cases × 5 mutations = 25 mutation trials**, with no allowed post-hoc exclusions. A rejected mutation must report its expected semantic reason rather than any incidental compiler crash.

### 3.4 Explicit bounded operational soundness obligations

The [finite semantics and proof sketch](C1_FINITE_EVENT_SEMANTICS_AND_PROOF_SKETCH_20261104.md) specifies prefix-operational states for producer contributions, committed values, seen consumer reads and occupied source slots. Conditional propositions P1–P3 express no premature consumer reads, no early slot overwrite, and standard fixed-order interval-packing capacity; P4 separates source-value safety from `Phi` issue-order conformance. The argument holds **only if the bounded graph's expected producer/consumer obligations have been enumerated correctly**. The proof sketch is not mechanically verified and does not establish RTL refinement or arbitration of physical memory ports.

The R3g differential checker implements the same bounded problem using a separately coded operational interpreter and forward-rank issue enumerator; **133/133 original tested trace variants agreed with the compiler (0 disagreements)**, with the author-reported source SHA and read-only receipt in [the frozen R3g/R3h checkpoint](R3G_R3H_ORIGINAL_C1_C2_WORKSTATION_RESULTS_20261104.md). This is finite observed agreement, **not** independent specification completeness because both interpreters share the admitted static model.

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

### 4.3 Compiled-once source-certificate consumption

The new C2 `CompiledActionCatalog` preserves each **original, typed, source-verified next state** after the ordinary compiler has evaluated its candidate. A later proposer may select an E0-certified in-process catalog entry without a second source compiler run, while a rejected or E0-unknown candidate is never silently promoted. The R3h controlled study tests this property on the same six R3f graph families and eight actions each. This is **an API/call-count ablation**, not optimized exploration: producing a catalog of eight candidates still requires eight compiler transitions and may be more expensive than submitting one blind candidate. The R3h author-run original campaign verified **48 candidate source transitions across six graphs** and **zero additional transitions to select an already certified cached state**. It is an API-level implementation result; mask construction still requires all 48 source transitions in that cohort. Original read-only audit receipt and source revision are in [the frozen checkpoint](R3G_R3H_ORIGINAL_C1_C2_WORKSTATION_RESULTS_20261104.md).

## 5. Experimental design and reproducibility

Source and native original measurements use explicit Git source revisions, original generated-source SHA, input checks and independent audit receipts. R3d/R3e/R3f create only new output folders and never overwrite the earlier frozen studies. Proposed CPU experiments require a clean tracked source checkout. Host source validation, native HLS and physical placement/routing are separate phases.

All campaigns preserve the exact denominators and every negative result. The planned R3e finite oracle is an independent brute-force reference for unit slots under fixed source order; it is not a foreign compiler baseline. The R3d manually specified six expected statuses are not an exhaustive independent semantic oracle. Future completion requires cross-artifact capability tests and a larger domain-specific action-space sweep to establish generalizability.

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

## 7. Closest work and scientific distinction to establish

The initial pinned Allo artifact review has since confirmed `reuse_at`, `partition` and `buffer_at` memory primitives, use-def and schedule composition, and real **positive and negative** `allo.verify` equivalence tests. Therefore, we do not treat "memory reuse", "composition" or "source verifier" as absent from Allo. A **matched artifact-executed test of all-reader release plus per-operator source issue order** is required before any exclusivity statement. See the [expanded pinned source analysis](C1_C2_PINNED_PRIOR_ART_RECONNAISSANCE_20261103.md).



**Allo (PLDI 2024)** already provides bottom-up type-safe composition of compute, memory, communication and datatype customizations; its public `allo/verify.py` includes schedule/source verification using PAST. **StreamTensor (MICRO 2025)** introduces stream-layout tensor types for fusion, buffer allocation and memory optimization. **Filament (PLDI 2023)** encodes hardware timing/structural constraints with timeline types. **CODO (ISCA 2026)** checks FPGA dataflow violations and optimizes communication/scheduling. **ST-Flow (PACT 2026)** compiles spatial-temporal dataflow. Their substantial overlapping contributions rule out generic claims of the *first typed compositional FPGA IR*, *first temporal source checker* or *first optimization-ready graph*. Public references and pinned revision audit are recorded in [the independent prior-art reconnaissance](C1_C2_PINNED_PRIOR_ART_RECONNAISSANCE_20261103.md).

The still-unresolved novelty claim is **a concrete, testable combination of bounded operator-local source issue semantics, multi-reader last-use scratch safety and an exactly same-compiler three-valued action interface**. Establish that distinction through executed, pinned competitor artifacts with matching positive and negative source designs and explicit semantics. Do not infer absence of a rival feature merely from a short abstract or one file.

## 8. Threats to validity

Bounded MatMul chain/fanout source semantics do not imply generic CNN fanout, dynamic graphs, cross-clock temporal semantics, physical concurrent bank-port safety, complete NN training or hardware power/latency. A CPU/GCC source trace is not an all-input fixed-point theorem; E3 can fail timing after E0/E1 succeeds. A generated `Phi` spatial probe is not an RTL or source cycle schedule. The mask can increase CPU cost because it validates every candidate in advance; no measured agent advantage is established. The original HLS negative results must remain visible. Independent prior-art and held-out graph-family generalization tests are not yet completed.

## 9. Conclusion and remaining scientific gates

We present an implemented bounded source-contract and policy-independent interface as candidates for two scientific contributions. Existing source and native results demonstrate concrete supported functionality and important limits. R3d/R3e/R3f and now R3g/R3h finite C1/C2 source experiments have been workstation-verified. An independently executed external-artifact equivalence/lifetime comparison, broader source-graph generality and peer assessment remain open. To turn this into a submission-qualified research paper, we must complete those original source experiments, independently inspect and execute a fair competing-artifact capability comparison, establish whether any semantic distinction survives that comparison, and finalize the formal scope, artifacts and independent review. **No trained agent is required for the intended compiler paper, and none has been evaluated here.**

### 9.1 Unfinished post-evaluation source studies and final evidence export

The separate R3j held-out experiment is designed to evaluate **24 prespecified combinations of fanout degree, shape and independent `Phi` mappings** over **216 finite original event traces**, including unsafe address aliasing and source-safe/issue-order-invalid controls. R3k is a matched **12 source-model-pair / 192 typed-action** intervention on explicit source parallelism limits. These studies are **not yet workstation-verified** and should not be counted as observed results.

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
