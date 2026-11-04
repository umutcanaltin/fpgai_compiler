# FPGAI CST-Model IR — scientific study protocol

**Date:** 2026-11-02. **Status:** preregistered hypotheses and research procedures; not a firstness or hardware-performance claim. This is the compiler research track; frozen RL/Gym remains untouched.

## Primary research question

Does coupling an operator-local spatial/temporal mapping Phi with an independently checkable **tensor-element availability and physical scratch-lifetime relation** provide an executable, diagnostically useful compiler representation beyond the existing per-operator FPGAI IR?

This is deliberately narrower than inventing scheduling, layout typing, row buffering, multilevel IR, or FPGA compilation.

## Falsifiable hypotheses

**H1 — representational and executable choice.** Hold Functional IR, numeric policy and each operator's resolved Phi constant. B=1, B=2 and B=M schedules should generate different real HLS C++ programs with independently accepted lifetime contracts and numerically equivalent outputs. Reject H1 if only manifests/hashes change without actual addressing and execution changes.

**H2 — incremental legality diagnostics.** A candidate schedule checker must reject illegal production/consumption orders and scratch aliasing with the first concrete row/stage/edge/address counterexample, while passing legal controls. The existing per-operator Phi refinement is only a *scope ablation*; it has no proposed schedule as input, and we do not call this an unfair performance victory over other schedule checkers.

**H3 — actual source fidelity.** On prespecified supported 3-/4-MatMul graphs and odd row/tensor tails, emitted GCC-host C++ must pass an independent NumPy numerical oracle; original Phi iteration traces and independently expected row-ready/consumed traces must agree. Failures remain in denominators.

**H4 — HLS effect (not yet measured).** In a paired original licensed Vitis campaign, B=1/2/M may induce latency/resource/feasibility trade-offs; do **not** assume improvements. Measure actual original fixed-point CSim, CSynth cycles/resources/timing and explicit rejection reasons. Source scratch length is not proof of physical BRAM.

**H5 — research novelty (not established).** A feature-level, reproducible comparison must demonstrate what the *exact* Phi-to-element lifetime contract and certificate provide beyond the published semantics/implementations of Allo (PLDI 2024), HIDA (ASPLOS 2024), StreamTensor (MICRO 2025), and ST-Flow (PACT 2026); otherwise narrow the claim or change thesis.

## Prespecified CPU experiment design

Catalog: deterministic 2, 3, or 5 row graphs, 3- and 4-operator MatMul chains, odd widths, both globally uniform and independently mapped PE/SIMD/reuse settings. For each graph, deduplicate row block B from 1, min(2,M), min(3,M), M.

Positive controls: genuine B-row wavefronts with exactly B physical row slots per intermediate tensor. Negative interventions, all specified *before observing results*:

1. Consumer executes before producer — READ_BEFORE_PRODUCE_OR_STALE_ROW.
2. Duplicate an operator/row execution — DUPLICATE_STAGE_EXECUTION.
3. Delete the terminal execution — MISSING_STAGE_EXECUTION.
4. Inject an out-of-domain stage — INVALID_STAGE_OR_ROW.
5. For B>1, provide only one scratch row slot — OVERWRITE_UNCONSUMED_VALUE.

Record every trial (including failures): functional graph, mapping policy, proposed event sequence/row block, legal/invalid a priori label, returned status, first counterexample, event budget, and CPU checker time. The intentionally induced faults establish detection of *these designed classes*, not generalized real-world mutation recall. Do not cherry-pick passing cases.

For selected valid programs, execute generated host C++ against NumPy. Check actual Phi coordinates and temporal row event traces separately. A bounded E0 schedule proof is not a formal proof about generated RTL. A host-float E1 test is not Vitis fixed-point bit exactness.

### Hardware campaign (requires original licensed Vitis)

Family A: four matched groups, each whole tensor vs column-major vs one-row lifetime. Family B: 3-/4-operator graphs compiled with row block 1, 2 and M, the same graph/Phi, identical fixed-point policy/part/clock per triplet. Both are create-only, source-hashed. Family B uses cst_block_native_prepare.py and cst_block_native_report.py. Native runs are opt-in and auditable. Without original Vitis output, report all timings and physical resource fields as **unmeasured**.

### Statistical and causal interpretation

Primary E0 outcomes: correct positive accepts, correct negative rejections by fault type, false accepts, false rejects, first-counterexample class, peak live elements, checker runtime distributions. E1: actual GCC numerical pass fraction and error, actual Phi trace, row-event trace. E2/E3: original CSim qualification, CSynth cycles/resources/timing feasibility and missing cases. Do not treat small synthetic graphs as broad DNN hardware performance.

Comparisons must be source- and model-matched. Same-compiler restricted ablations demonstrate expressiveness only, not superiority to Allo/ST-Flow. No generic novelty or performance claims until direct artifact-level comparison and hardware results.

## Research software and reproducibility

- model_ir/temporal_schedule.py: candidate schedule semantics and concrete counterexamples.
- codegen/hls/row_chain.py and compiler/row_chain.py: opt-in B-row source-level lowering, with independent schedule checking required.
- validation/cpp_chain_trace.py: executed host row-event temporal agreement.
- experiments/cst_schedule_ablation.py: controlled legal/invalid schedule campaign.
- experiments/cst_block_native_prepare.py: create-only matched native source packages.
- experiments/cst_block_native_report.py: read-only paired HLS report with missing fields.
- .github/workflows/compiler-ir-cpu.yml: CPU/GCC tests plus prespecified source experiment artifacts.
- .github/workflows/compiler-ir-native-vitis.yml: explicit dispatch on a licensed, dedicated self-hosted runner; never starts automatically.

**Go/no-go:** H1–H3 are necessary for a new executable IR mechanism; H4–H5 determine whether the claim is publishable as a distinct compiler research contribution. If prior art has the same semantics, do not present a renamed method as new. Preserve RL code and earlier experimental sources without mixing evidence.


## Publication-critical acceptance experiment, locked 2026-11-02

This compiler-first contribution is **not** the introduction of type-safe
accelerator composition or spatial/temporal dataflow in general. Allo,
HIDA, StreamTensor and ST-Flow already study key overlapping mechanisms.
Wavelet/Let It Flow presents Lean-verified asynchronous dataflow
compilation. Treat the precise mapping-conditioned source certificate as
a falsifiable, narrower hypothesis, pending direct peer evidence.

### Primary scientific endpoints — not just hardware engineering

1. **True mapping-conditioned discrimination:** for the SAME Functional
   graph, numeric configuration, scratch capacity and proposed early
   consumer schedule, change ONE legal producer/consumer operator-local
   Phi mapping. Derive all contributing iterations, complete-reduction
   COMMIT and every consumer READ. A verifier only using a Phi SHA or
   whole-operator completion fails this gate. Demand concrete,
   independently reconstructed first counterexample and actual emitted
   source-level READ/COMMIT evidence; not fabricated abstract timestamps.
2. **Bounded soundness and completeness within stated scope:** independent
   finite event oracle over prespecified small schedules, legal/illegal
   address reuse, multiple readers, reductions and read-after-overwrite.
   Report the full confusion matrix, denominators, first witness,
   bounds-exceeded and UNKNOWN separately. Zero designed false accepts
   is a useful local result, not a universal formal proof.
3. **Constructive alternative:** compared with a conservative whole-tensor
   barrier and source-matched restricted per-operator Phi-only compiler,
   generate actual different C++ memory addresses/execution; independently
   reexecute numerical and event reference. If only metadata changes,
   the new IR's central mechanism is not yet established.
4. **Ablation attribution:** independently remove Phi-conditioned
   completion, multi-reader final release and address/port constraints
   while keeping the same graph/inputs/clock/numeric choices. Expose
   which invalid examples each restricted variant misses; do not compare
   against deliberately broken baseline implementations.
5. **External semantic parity:** run an explicitly pinned artifact from
   Allo, ST-Flow (PACT26), StreamTensor and, where meaningful, Wavelet.
   Record the *exact task and semantics* each can/cannot express, version,
   environment and limitations. Unsupported or unexecuted is UNKNOWN;
   neither missing code nor a paper's silence is evidence of superiority.

### Secondary HLS/physical endpoints

Prespecified NCHW evidence is 12 original graph/Phi/storage cases
(six disjoint/shared matched pairs), and a 2x2 `auto` vs
explicit `bram_2p` memory-policy design (24 cells) when the
corresponding original Vitis studies are fully qualified.
Do not substitute source buffer elements for inferred BRAM. The
existing explicit-binding original HLS data do **not** establish
monotonic improvement; mixed cycle directions and nondecreasing
estimated BRAM are scientifically important negative results.

Report each original case's CSim numeric qualification, effective HLS
clock feasibility, exact resource counts, reported worst cycles,
original requested timing, manifest/source/RTL/ROM hashes, tool
identity, vendor failures and UNKNOWN. For post-route E4 demand:
exact FPGA part, 32ns requested clock, original **both source and
working-directory** ROM hashes, fresh vendor receipts, proven complete
routing, clean DRC, WNS and parsed LUT/FF/DSP/BRAM. A Vivado exit=0
alone is insufficient. No OOC result establishes a board-deployed
clock/Fmax or power.

A pair effect is computed **only** if both original cases are independently
qualified, including timing and DRC. Report missing, rejected and
unattempted cases in the original denominator; do not delete them,
impute zero, average incomparable graph topologies or choose the
best-looking Phi after observing outcomes. Separate exploratory
toolchain repairs from frozen primary scientific comparisons.

### Paper stop/go

- GO for a **bounded semantic/certificate compiler paper** only if the
  actual mapping-conditioned counterexample and source-refined
  constructive execution pass independent tests AND a direct artifact
  comparison identifies a defensible difference.
- GO for **hardware trade-off claims** only on original fully paired
  E3/E4-qualified real vendor reports with complete provenance.
- If native HLS/physical BRAM does not improve, report that negative
  finding; no search or memory-optimality claim follows from smaller
  C++ arrays.
- If the same certificate capability already exists in prior art, narrow
  or revise the paper thesis rather than asserting firstness.
- Original trained-network external-validity experiments, release/IP,
  bibliography and independent artifact reproduction are additional
  submission requirements; compiler-only synthetic examples cannot
  substitute for all of them.


## 2026-11-02 addendum: prospective consumed-readmem physical provenance

The v5 original exit-zero Vivado log **explicitly** reports that the
source-pinned `$readmem` asset was read successfully and emitted a
routed checkpoint. The file was missing from the original vendor cwd
after execution, even after switching its project `add_files`
registration. The physical study remains unqualified; no hardware
QoR effect or E4 timing statement is established.

Prospective v6 uses a separate, strictly fail-closed,
`ATTESTED_VENDOR_READ_POST_RUN_ABSENCE_V1` receipt/witness
protocol, anchored in the NEW source-pinned original Tcl. It preserves
the SHA-256-matched bytes **before** execution, original RTL/ROM
source stability and original hash-matched vendor-read-success line,
and records the ROM's post-run disappearance as a distinct evidence
mode. Source-only tests and simulated vendor logs are not original
synthesis observations. The entire E4 route/timing/DRC/report gate
remains mandatory. Legacy v4/v5 outputs are never retrospectively
promoted; records in other evidence modes are not silently pooled.
Refer to `NCHW_NATIVE_BINDING_AND_VIVADO_OOC.md`.

This prospective provenance regime changes **how a legitimate
readmem-then-absent-file case is audited**, not the mathematical model,
the mapped hardware architecture or any scientific outcome metric.
Therefore it is a toolchain-method repair, **not compiler-IR scientific
novelty**. The preregistered causal Phi/certificate counterexample and
external peer-artifact comparison remain uncompleted.

## 2026-11-02 prospective secondary physical experiment and limitations

The original v5 `case_000` hash-matched Vivado report has complete
routing, +20.724 ns WNS at the original 32 ns target,
4995 LUTs, 2783 FFs, 25 DSPs, 2.5 Block RAM tiles, 0 URAM;
its 55 DRC warnings are `DPIP-2(16)`, `DPOP-3(19)`,
`DPOP-4(19)`, `RTSTAT-10(1)`. The vendor-confirmed ROM read
is not retroactive E4 qualification and the original v5 still
has `routed_qualified=false`. The 54 DSP-pipeline warnings imply
design-specific delay/power optimization tradeoffs; do not insert
a pipeline register in only one compared implementation.

The new source-pinned physical study uses two separately counted
tiers. **Primary:** fully routed, timing-met **clean-DRC E4**
with all original vendor/source/ROM/report hashes. **Exploratory:**
the *same* gates except that a nonzero warning-only DRC report
is permitted in a frozen, explicit pilot-informed allowlist of
`DPIP-2, DPOP-3, DPOP-4, RTSTAT-10`. This is labelled
`OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS`, never
`QUALIFIED_VIVADO_OOC_ROUTE_TIMING_E4`, and never demonstrates
deployable correctness or a power/frequency improvement.
Unknown/severe/unrecognized DRC faults, incomplete routing,
missing/forged reports, or nonnegative-WNS failures exclude a
case from **both** tiers. Publish case-specific DRC counts,
evidence regime and zero/unknown fields explicitly.

**Selection-bias warning:** the allowlist is based on the original
v5 exploratory observation. Do not present it as a fully independent
pre-registered primary physical study, and never blend its
measured differences into clean-DRC E4 counts. Retain the
original denominators and use matched disjoint/shared source/part/
Phi/clock pairs before any exploratory difference. The novel claim
remains certificate-level, falsifiable source-refined semantics
versus prior art; HLS/physical evidence is secondary.


## 2026-11-03 three-contribution research protocol

The publication is organized around **C1 CST Model IR**, **C2
bounded independently checkable compiler certificates**, and
**C3 policy-independent RL-compatible design exploration**.
These are three connected contributions of ONE CST research
compiler, not three unrelated systems.

- C1: for supported static MatMul/NCHW DAGs, materialize producer
  completion, all mapped consumer reads, last-reader release
  and actual generated scratch addressing under independent
  per-operator Phi. Report legal/illegal counterexamples and
  source-checked numerical outcomes.
- C2: compare original E0 logical contracts, E1 executed GCC
  events, E2/E3 original fixed-point Vitis and strict DRC-clean
  E4 implementation. A `warning-observed` physical case
  may inform descriptive QoR but is NOT strict E4.
- C3: use one immutable state/action/transition semantics for
  policy-neutral human, random, heuristic or future RL
  proposers. Verify R1 restricted independent legality oracles,
  R2 causal mapping dependence, R3 anonymous variable-graph
  observation and R4 certificate equality. Conservative mask
  UNKNOWN is disallowed; source-legal is NOT physically feasible.

Primary C1/C2 **scientific distinctness remains falsifiable** by
upstream ST-Flow, Allo, HIDA, StreamTensor and Wavelet capabilities.
R5 actual RL benefits are NOT part of this compiler-paper
acceptance unless independently trained and evaluated. The new
R3/R4 code was not yet run on the workstation at this edit;
no manuscript numbers may be copied from unit fixture values.

The single-command CPU/GCC series now includes both
`rl_compatible_ir_r3_r4` and `two_phi_counterfactual_e1`.
The latter's exact original GCC CSV hash is independently
checked by the source evidence index to prevent mixing an
unrelated experiment or revised original case rows.
