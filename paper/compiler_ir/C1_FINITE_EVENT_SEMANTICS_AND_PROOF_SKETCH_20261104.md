# C1 finite trace semantics and proof obligations — compiler paper

**Research status:** bounded proof sketch and executable falsification program, NOT mechanized formal verification or a priority claim. Source revision of previously measured R3d–R3f results: `fc40369f9493d93cc78c4886ae2e32c5d1f64093`. The new R3g differential semantics and R3h compiled-once catalog are **not yet workstation-verified**.

## 1. Explicit finite source model

Let the supported static design contain typed tensors, a fixed total source event order `E = [e0, ..., e(T-1)]`, a complete operator-local architecture assignment `Phi`, and scratch address function `addr(v)` defined for each intermediate tensor element. Supported events are:

- `CONTRIBUTE(v,k)`: a producer reduction contribution to value `v`, with source/weight index evidence and a unique in-range reduction index `k`;
- `COMMIT(v)`: complete the value `v` once all required producer input obligations hold and occupy its assigned scratch slot;
- `READ(v,c)`: consumer use `c` of value `v`, only after commit and while that exact value occupies its scratch slot. The last required consumer read releases the slot.

Each value has an explicitly enumerated finite expected consumer set `Use(v)`. A producer has an expected finite reduction set `Terms(v)`. For output of a reader or merge, its `Terms` obligation becomes the required incoming `READ` tokens. The supported source models do **not** admit inferred unspecified consumers. Every event prefix determines a deterministic operational state:

`(Done, Contributions, SeenReads, LiveSlots)`.

The transition relation has *partial* semantics: an event is permitted only when its producer completeness, typed consumer identity, unique use, source `Phi` owner, and live-slot conditions hold. Rejections are accompanied by a concrete first failure.

## 2. Bounded safety propositions

**Proposition P1 (no premature consumer read, within the finite model).** Suppose (i) `Use(v)` and `Terms(v)` accurately enumerate every source obligation in a supported static graph, (ii) event decoding and type/shape admission are correct, and (iii) the verifier admits a complete trace only when every prefix follows its checked operational transition relation. Then every admitted `READ(v,c)` occurs after all `Terms(v)` and `COMMIT(v)` and before slot release.

*Proof sketch.* Induct on event prefixes. Initially no values are committed or live. A `CONTRIBUTE` adds a unique permitted term; a `COMMIT` is admitted only with all terms already observed, no previous commit and an available address; a `READ` requires a prior commit and exact live occupant. After the final required read, the occupant is released. Thus an admitted read could not be uncommitted or stale under the assumed source operational model.

**Proposition P2 (no early slot reuse, within the same fixed event order).** If every `COMMIT(v)` refuses an occupied address and an address is released only after `SeenReads(v) = Use(v)`, no two simultaneously live values occupy the same scalar scratch slot. Reusing a slot before the last required reader violates a prefix check.

*Proof sketch.* By induction, the live-address map is initially empty. Only a commit adds a live occupant after checking vacancy; only a final permitted read removes one. Therefore the map remains a function from addresses to one live value. This does not prove that physical single/dual-port BRAM accesses can be issued in parallel.

**Proposition P3 (fixed-order unit-slot capacity).** Under a *known total source order*, equal-size scalar slots and no banking/port constraints, the least number of slots required by all value intervals `[COMMIT(v), LastRead(v)]` equals the maximum simultaneous overlap count. The interval property is a standard result; the present contribution is explicit source-event generation and source-address checking, **not** a new graph-coloring theorem.

**Proposition P4 (Phi issue conformance is independent of safety).** Value-readiness and last-use safety alone do not imply `Phi` issue-order conformance. In the supported 2–4-reader fanout model, swapping adjacent reader uses of an already committed H value can preserve all source-value safety obligations while violating the per-reader source ordinal rank prescribed by `Phi`. A complete compiler claim therefore needs two distinct checks.

**Proposition P5 (finite mask soundness relative to the certified transition).** For an explicitly supplied finite complete-design action catalog `C`, define `Mask_E0(S,C)[i]` as true exactly when the ordinary compiler action transition for `C[i]` emits an accepted bounded E0 certificate. Under deterministic source/compiler execution and certificate integrity, every true bit corresponds to that source-certified transition. This is a definitional, implementation-relative soundness claim, **not** HLS/RTL/FPGA feasibility or a novel RL sample-efficiency theorem.

## 3. Proof dependencies and falsification targets

Each proposition relies on assumptions that must remain explicit:

| Dependency | Finite verification program | Missing stronger evidence |
|---|---|---|
| Complete `Use(v)` / producer terms | R2f/R3e original fanout event traces, selected C1 mutants | Independent arbitrary graph shape and fanout semantics |
| Source address uses conform to typed events | R2h/R2o original GCC emitted C++ witness checks | Formal translation/refinement of arbitrary emitted C++ or RTL |
| No arbitrary early address reuse | R2c independent exhaustive interval-coloring oracle, R3e 5/5 observed oracle agreements | Banking/port/clock and arbitrary interleavings |
| Independent source-value and Phi issue properties | R3e verified wrong-issue vs source-safe mutant; R3g newly prepared independent reference differential oracle | Proof for dynamic graphs / pipeline execution |
| Exact bounded source action legality | R2p unified facade; R3d 6, R3f 48 observed action receipts | External complete legality oracle, all CNN/NN graphs, hardware viability |
| Cached source-state selection without recompilation | R3h newly prepared compiled-once experiment | Any advantage over one unmasked action or trained agent |
| Novelty against existing compiler IR | External pinned Allo/CODO/ST-Flow source reconnaissance | Actually executed equivalent-capability artifact study |

The independent R3g reference oracle shares the **declared supported static graph and source semantics** with production. Its construction is independent only in event interpretation and `Phi` forward-rank enumeration; it cannot by itself falsify omissions that both designs share. The R3g adjacent-swap corpus adds systematic perturbations but is not exhaustive over all typed event traces.

## 4. Distinguish correctness, completeness, and scientific novelty

**Verifier-relative soundness** means that for the bounded *formalized* event language, every accepted trace satisfies the stated operational properties, given correctly enumerated obligations.

**Completeness** would require every legal source schedule representable in that language to be accepted. Our compiler also requires a particular `Phi` issue order, and its fixed-shape admission excludes many safe programs, so no arbitrary-schedule completeness claim is available.

**Native execution correctness** needs emitted C++ numerical/trace tests (E1), actual Vitis CSim (E2), and any stronger translation proof if claimed. E0 alone is insufficient.

**Scientific novelty** requires comparing the precise representation/checker combination to independently implemented prior-art mechanisms. The proofs P1–P5 do not establish priority or originality; Allo, Filament, StreamTensor, CODO and others supply important overlapping approaches.

## 5. Paper-ready acceptance gate

C1 can be claimed only at its proven static domain if the oracle agrees, negative counterexamples fail for the claimed reasons, emitted-source address witnesses remain valid, and sensitivity to alternate in-place scratch accounting is fully disclosed. C2 can be claimed as a policy-independent interface if state/action invariance, immutable typed transitions, complete candidate masks, explicit UNKNOWN and screening costs pass. **Neither result justifies claiming improved RL learning: trained agents are a different paper.**

This is the mathematical story to review with a hardware-compiler or programming-languages expert before submission, not a substitute for peer-reviewed proof.
