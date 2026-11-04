# Formal source-event obligations, soundness theorem and exact baseline

**Scope:** This is a **mathematical statement about one finite, sequential source-event trace**, not a machine-checked proof, not a Vitis/Vivado hardware schedule, and not an originality claim. The source-event model and interval allocation are standard. The exact relationship between the compiler's typed Model IR, its independently selected operator Phi mappings, the source-event witness and actual C++ emitted operations is the bounded scientific question.

**Status (2026-11-03):** Source-event proof obligations and earlier R2s/R2r source controls are recorded. The R2t whole-tensor optimum and R2u native HLS factorial **have now passed workstation verification at pinned original revision `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`**: full regression 1,565/1,565, R2t 36/36 proven feasible global whole-tensor minima (12/36 improve on greedy; 36/36 equal tensor peak lower bound), R2u 32/32 E2 CSim and 30/32 E3 CSynth qualified, with two original effective-clock-budget failures. **This does not turn the source theorem into all-input/compiler equivalence or hardware E4.** Proof obligations below are stated explicitly to prevent E0 source results being promoted to RTL or physical claims.

## Model, assumptions and observable semantics

Let a supported finite functional graph have a finite set of tensor values

[
V={(mathrm{tensor},mathrm{element})}.
]

Fix one concrete, **total, single-threaded** source event order produced by the current bounded R2f/R2h compiler:

[
E=(e_0,e_1,ldots,e_{T-1}).
]

Each source event is a **COMMIT** of one logical tensor element or one **READ** of that element. For every `v` in `V`, assume:

1. `v` has exactly **one** COMMIT event at ordinal `c(v)`;
2. `v` has at least **one** READ, all with ordinal greater than `c(v)`; define `l(v)` as its **last actual READ ordinal**;
3. source scalar values occupy unit-sized slots; a scalar COMMIT writes exactly one slot and READ returns the value currently stored at that slot;
4. a new COMMIT may overwrite a slot only **after** the previous logical value's last READ (closed lifetime interval `[c(v),l(v)]`);
5. an explicit source-to-runtime refinement assumption holds: the actual generated C++ program executes scratch COMMIT and READ operations corresponding to this event order, using the physical addresses stated in its allocated map, without undeclared writes, speculative memory effects, simultaneous ports or other unmodeled concurrency.

A map `a:V→{0,...,K−1}` is **interval-safe** if all values with overlapping closed lifetimes map to **different source slot addresses**:

[
[c(u),l(u)]cap[c(v),l(v)]
eemptyset
quadLongrightarrowquad a(u)
e a(v).
]

In the code, `element_scratch_allocation.extract_intervals` checks (1),(2); `verify_allocation` checks the exact trace against `a` and capacity; `allocate_minimum_slots` is an independently checkable standard interval-coloring implementation for the **fixed total event order**. R2r adds bounded **actual source-operation fault injection**, but does not prove assumption (5) for all possible emitted source programs.

## Theorem 1 — Source scratch refinement under explicit assumptions

**Claim.** For any finite event order satisfying assumptions (1)–(5) and any interval-safe map `a`, a sequential scratch machine initially empty and executing `COMMIT(v,x)` as `scratch[a(v)]=x` and `READ(v)` as `scratch[a(v)]` returns the most recently COMMITted value of `v` on **every READ(v)** for **every sequence of scalar COMMIT values**.

**Proof (ordinary mathematical induction on source event position).** Immediately after `COMMIT(v,x)`, slot `a(v)` stores `x`. At a later ordinal `t` before or at `l(v)`, any different value `u` that could write `a(v)` must start a COMMIT after `c(v)` but not later than `t`. Then `t≤l(v)`, so the closed intervals of `u` and `v` overlap, contradicting interval safety if `a(u)=a(v)`. Thus no such write can alter `a(v)` between its unique COMMIT and any of its READs. Induction through the total event sequence gives the claimed per-value readback. After `l(v)`, the slot may be reused without affecting any future read of `v`, because none exists. □

**Important limitations.** This is a source *memory-value preservation* statement, not proof of **correct arithmetic**, equality to the functional Model IR for arbitrary inputs, hardware FIFO/dataflow legality, inferred BRAM ports, HLS pipeline concurrency, achieved clock or power. The code has a checker and repeated original host GCC source executions, **not** a proof that arbitrary generated C++ refines the event order. A single-input numerical pass proves only that input; R2r's 216 mutants are an exact chosen-fault-family adequacy result rather than a universal soundness or compiler-correctness theorem.

## Proposition 2 — Minimum scalar slots for a fixed total source event order

For unit scalar slots, the fixed-order interval overlap graph is an **interval graph**; its chromatic number equals its maximum clique size, which is the **maximum number of simultaneously live scalar values**. A standard earliest-release coloring procedure uses exactly this many slots. Therefore the current `allocate_minimum_slots` returns a minimum `K` **provided assumptions (1)–(4) and the exact interval replay constraints hold**.

This is **classical interval coloring**, not a new compiler result or evidence that a layout reduces on-chip FPGA RAM after HLS. If computation/clock order changes, the interval family changes and the optimum may change; if memory ports/banks and simultaneous accesses matter, this source-only scalar coloring bound need not be implementable as one on-chip RAM primitive.

## Proposition 3 — Global optimum for **whole-tensor contiguous blocks**

R2t uses a different conservative model. Each tensor `t` has length `size(t)` and a single interval from the **first scalar COMMIT** to the **last scalar READ** of *any* element. All elements of one tensor occupy consecutive source addresses beginning at `base(t)` throughout this interval. Only temporally overlapping whole tensors must use disjoint address ranges.

For any feasible integer-base assignment, sort the tensor blocks by nondecreasing physical base (tie-break consistently), giving a total **spatial ordering** `π`. Any two temporally overlapping tensors already have disjoint source address intervals, so their order by `base` satisfies a lower-to-higher disjointness constraint. Under a fixed spatial order, place a block as far left as possible subject to all earlier, **temporally conflicting** blocks:

[
base_pi(t)=maxleft({base_pi(u)+size(u):uprec_pi t, I_ucap I_t
eemptyset}cup{0}ight).
]

Induction over the spatial order shows this **left-justified** assignment is no farther right than the original feasible assignment, and therefore does not increase the required overall capacity. Enumerating **all spatial permutations** and retaining the smallest resulting capacity therefore attains the optimum for the specified fixed set of whole tensor lifetimes and contiguous source block constraints, **when the full search completes**. An independent bounded reference enumerates all integer bases for small counterexamples. The original source event replay then verifies that the returned base map is *actually feasible*.

If R2t exhausts its explicit search-node budget, return **UNKNOWN_BUDGET**, not `OPTIMAL_PROVED`. The tensor concurrency peak `max_t sum(size(t))` over overlapping intervals is a valid **lower bound**, but may be below the true contiguous-block optimum. The earlier R2q greedy first-fit is a **feasible upper bound**. The conventional scalar element-exact coloring optimum remains at or below any feasible whole-tensor layout under the same committed-value interval semantics.

None of this mathematical framing is new by itself; the scientific value is a **strong, reachable comparator** against which to judge whether the compiler's per-element source-lifetime contract adds useful expressiveness and what remains after removing baseline artifacts.

## Proposed Phi-saving inequality — **unproved** general statement

The independent reviewer proposed, and R2s tested on **126 finite paired source configurations**, the inequality

[
S_Phile Rleft(max_i PE_i-1ight).
]

All **126/126** original R2s comparisons satisfied it, with no violation. These observations do **not** establish this bound for arbitrary tensor dimensions, arbitrary PE/SIMD mappings, all valid schedules or hardware. In particular the values **102 reductions under COMMIT→last-READ source semantics and 0 reductions under hypothetical first-MAC in-place semantics** show that even the definition of `S_Phi` must be explicit. Do not promote this empirical inequality to a theorem without a proof from the compiler's exact `generate_fanout_events` ordinal-to-coordinate mapping, including partial final PE tiles and multi-reader last-use closure.

## E0, E1, E2/E3 and E4 boundaries

| Tier | Evidence that may qualify | What it cannot establish |
|---|---|---|
| **E0** | Finite typed graph, per-operator mapping request, exact source-event trace, interval-safe addressing certificate, source peak and feasible memory assignment, frozen SHA | Runtime execution, RTL cycle order, RAM ports, hardware QoR |
| **E1** | Actual source generated and GCC-executed on fixed float inputs; independent reference intermediate snapshots; dynamic source shadow fault detection | Universal arithmetic correctness, bit-exact vendor fixed point or inferred BRAM |
| **E2/E3** | Original source-attested Vitis CSim with numerical criteria; qualified source-attested CSynth latency and used-resource estimates at explicit clock/uncertainty | Routed BRAM, achieved Fmax, deployed power |
| **E4** | Source-attested Vivado place/route, raw reports, **clean critical DRC**, timing/constraint qualification, reproducible tool version/seed | Unmeasured boards/workloads, other process variations |

The legacy R2m physical observation has six routed BRAM ties, DRC warnings and **zero strict clean-DRC E4**. New R2u must not be described as E4 merely because Vitis estimates or a memory binding pragma are present. Later numerical tests should separately measure exact/fixed-point LSB errors rather than relying only on 0.1 absolute tolerance in historical E2.

## Remaining falsification and related-work gates

Before submission, audit overlapping mechanisms in **Allo, HIDA, StreamTensor, ST-Flow, CODO, Filament, MODeL, Timeloop and previous polyhedral/storage-optimization work** at the level of actual expressible value readiness, per-reader last-use schedules, admitted independently mapped PE/SIMD decisions, and **emitted source witness checks**. Absent an independently demonstrable capability gap and original validated external baseline, claim a **bounded model/source-verification case study**, not firstness of a new memory allocator or space–time IR.

The two original R2t/R2u study programs and command protocol are in [R2t/R2u study instructions](R2T_R2U_ALLOCATION_AND_HLS_MECHANISM_PROTOCOL_20261103.md). All original R2s and older results are frozen. No GitHub Actions or RL training is required.
