# Archived manuscript development and evidence notes — 2026-11-03

These dated notes were moved **verbatim** from the beginning of `MANUSCRIPT_DRAFT.md` after an independent reviewer noted that approximately 330 historical lines preceded the scientific paper title. This is a presentation/organization change only; original vendor/source artifacts and their scientific evidence tiers are not modified.

The active draft begins at [MANUSCRIPT_DRAFT.md](MANUSCRIPT_DRAFT.md). Historical claims below may be outdated or superseded; the active methods/results and current scientific status control.

---

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

> **Newest source-pinned compositional E0 result (2026-11-03,
> original `98676f7786687eddace0cab9124868448f9bdc24`):**
> The R2f 2–4 reader certificate family passed **27 focused
> and 1,357 full Python tests**, then **36/36 E0
> producer/contributor, reader/Phi, Add and final-value
> readiness certificates**. Of **27 non-scalar graph/Phi
> interventions against the fixed scalar mapping**,
> **18 changed the last H-reader rank and reduced
> the fixed-order minimum logical scalar scratch slots**,
> **9 tied**, **0 regressed** in the prespecified cohort.
> The experiment used **0 original GCC/Vitis/Vivado calls**.
> It neither establishes executed source-memory savings,
> real FPGA RAM ports nor BRAM/timing improvements.
>
> **New prospective, not yet workstation-verified R2g
> follow-on:** A 36-case source-pinned GCC C++ runtime-loop
> emitter and independent read-only audit now test actual
> producer contributions, independently mapped readers,
> Add-chain correctness and **every intermediate** NumPy
> value. The emitter deliberately uses **disjoint source
> scratch**, NOT the previously measured E0-optimal unit
> layout. Consequently R2g source correctness cannot
> be advertised as 18 physical-memory improvements.
> [R2g scientific protocol](R2G_RUNTIME_FANOUT_E1_PROTOCOL.md).
> Scientific firstness remains unproven because
> schedule-dependent storage optimization has mature
> prior art; native hardware and external peer comparisons
> are still required for submission.
>
> **Latest original R2b E1 workstation evidence — 2026-11-03.**
> At original source SHA
> `5d449ca595dbdfdf7a837ce862eb5fa8f89ce9ce`,
> `pytest` passed **1,253/1,253** tests and the
> prospective separately named E1 R2b campaign
> compiled/executed **24/24 real original GCC
> C++ sources**, all numerical and typed source
> event checks passing. Its independent, READ-ONLY
> original source/trace SHA auditor accepted **24/24**.
> Across 18 layout-scoped Phi interventions,
> **18/18** changed the global last-read event index,
> **12/18** changed final H-only READ rank or
> final consumer/output, **6/18** changed final
> consumer identity; **6/18** H-only
> comparisons tied. Two scratch layout arms
> are controls for each identical graph/Phi,
> NOT independent samples: this corresponds to
> **nine unique graph/Phi interventions,
> six changed final H-only READ ordering,
> three changed final consumer identity,
> three with unchanged H-only ordering**.
>
> This proves only a **bounded executed host C++
> source-consumption-order effect** under the new,
> explicitly fixed Phi-conditioned PE-group
> source scheduler. The historical fixed serial
> source study's **0/18** lifetime effects
> remain an independent valid negative control.
> **Phi-dependent source memory capacity savings
> were NOT established**, nor native HLS/RTL
> timing, external comparator novelty or RL
> policy learning benefits. Publication
> readiness remains **FALSE**. Full original
> [R2b E1 results and follow-on independent
> audit](R2B_E1_ORIGINAL_RESULTS_20261103.md).
>
> **2026-11-03 final six-pair quantitative update (supersedes
> first-pair-only interpretation):** Original `v6` auditor
> reported **12/12** fully routed/timing-met cases and **6/6**
> matched descriptive routed pairs, all with verified
> noncritical DRC warnings, and **0 strict clean-DRC
> E4** pairs. Disjoint-minus-shared vendor-reported
> BRAM differences by predeclared pair were
> `[0, +8, 0, 0, 0, -2.5]`: one 25% reduction
> (32→24), four ties and one regression
> (3.5→6). LUT `[-155,+57,-4,0,+9,-350]`;
> FF `[-10,-234,-120,0,-507,-21]`;
> DSP `[+3,0,0,0,0,0]`; WNS ns
> `[+0.195,-1.470,+0.537,-1.182,+0.707,-1.046]`.
> Positive resource differences favor shared storage;
> positive WNS differences favor disjoint slack.
> This is a **mixed descriptive non-clean-DRC OOC
> result**, not universal physical improvement,
> achieved Fmax or an external compiler comparison.
>
> **The stronger R2/C1 mapping-induced lifetime claim was
> NOT demonstrated:** 18/18 effective mapped reader
> interventions but 0/18 actual source last-use changes
> and 0/18 capacity changes for the fixed serial
> emitter. This is a published-paper-worthy negative
> finding about the tested static execution schedule,
> not a reason to modify original evidence or claim
> `Phi`-dependent lifetime firstness. RL-compatible
> interface C3 remains a bounded API claim: 9
> proposals, 3 certified legal, three accepted-by-source
> but conservative UNKNOWN and zero trained RL agents.
> See [complete original six-pair evidence](ORIGINAL_SIX_PAIR_PHYSICAL_RESULTS_20261103.md).
>

> **LATEST ORIGINAL RESEARCH DATA — 2026-11-03, not a
> submission-qualified paper.** Latest verified workstation
> full regression **1,236/1,236 passed** and original
> CPU/GCC `paper` research campaign **40/40 passed**
> at `1cefdd8566658cb6b69e12444c9b5ce9ba6e4a31`.
> After the original vendor run, the separately source-pinned
> Vivado v6 **12/12 case cohort is complete**: all twelve
> original OOC cases are fully routed/timing-met with disclosed
> allowlisted DRC warnings and **ZERO strict clean-DRC E4**
> cases. All six preplanned matched disjoint/shared pairs can
> now be reported **descriptively** after the complete original
> exporter is executed. Original numeric values are available
> only for the first pair in the uploaded transcript:
> `case_000` disjoint 4995 LUT/2783 FF/25 DSP/2.5 BRAM,
> WNS 20.724 ns; `case_001` shared
> 5150 LUT/2793 FF/22 DSP/2.5 BRAM, WNS 20.529 ns.
> Thus shared changes by **−3 DSP, +155 LUT, +10 FF,
> 0 BRAM, −0.195 ns WNS**. This is NOT a physical
> BRAM benefit or board-qualified timing/frequency result.
>
> **Negative R2 original finding:** In 24 GCC cases with
> 18 effective counterfactual reader Phi changes,
> original CPP/event SHA changed in all 18, but
> source-observed **last-use and scratch-capacity changed
> in ZERO**. The current serial-stage source emitter
> does not demonstrate Phi-induced final-reader lifetime
> changes; do not claim it does. See
> [the full falsification analysis](CST_R2_NEGATIVE_FINDING_AND_NEXT_FALSIFICATION.md).
> The stronger CST mapping-dependent lifetime thesis is
> **NOT qualified** by this experiment. The bounded
> source/certificate and policy-neutral RL-compatible
> interface remain genuine implemented contributions,
> but RL advantage and external firstness are unproven.
> The older tables below use their own historical data
> denominators and may be superseded only by final
> independent original six-pair effect export.
>
> **2026-11-03 scope and original evidence correction.** This is a
> **CST Model IR research paper**, whose three proposed contributions are
> (C1) mapping-conditioned tensor-element production/consumption/lifetime
> representation, (C2) bounded independently checkable architectural
> legality/source-refinement certificates, and (C3) a policy-independent,
> **RL-compatible** observation/action/transition/mask interface. C3 is NOT
> an RL training result and does not claim reduced RL sample complexity.
>
> The latest original **workstation-verified** main revision
> `98ee6097aa74b9dd9b1f16a26bea43b650f33bc1` passed **110 focused
> tests and 1,221 full tests**. An original create-only 12-case/6-pair
> Vivado v6 study was prepared. Its FIRST original `case_000` returned
> exit code 0 after 102 seconds, and the independent auditor classified
> it as **OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS**.
> It is NOT strict E4 clean DRC; eleven cases remain `PREPARED_NOT_RUN`
> and **zero matched routed pairs** exist. The original terminal excerpt
> did not include v6 numeric WNS/utilization/DRC counts, and they must not
> be borrowed from the historical v5 preview. The twelve-case earlier
> source-pinned original Vitis HLS E3 evidence is a separate experiment,
> and does not by itself establish FPGA physical BRAM savings.
>
> New R3/R4 studies and manuscript integration code added afterward have
> **not** yet been independently executed on the workstation. Full
> original external peer-compiler parity, trained network generality,
> and Phi-induced true lifetime differences remain unresolved. The
> author-facing current contribution contract is
> [PAPER_CONTRIBUTIONS_AND_RL_INTERFACE.md](PAPER_CONTRIBUTIONS_AND_RL_INTERFACE.md);
> the exact source-pinned launch is in
> [FINAL_RESEARCH_EXECUTION.md](FINAL_RESEARCH_EXECUTION.md).
>
> **Scientific revision gate — 2026-11-02.** This draft begins with
> earlier 6282fcea E0/E1 results and is **not** the latest complete
> experimental manuscript. The proposed stronger paper contribution is
> `Phi -> per-tensor-element full-reduction COMMIT -> multi-consumer READ ->
> last-reader RELEASE -> constrained physical address/port -> independently
> source-checked certificate`. Individual row scratch and column-major
> storage are classical mechanisms, not independent novelty. New source-observed
> NCHW GCC experiments and 12 original qualified Vitis BRAM_2P CSim/CSynth
> cases exist in separate source-pinned studies; their precise case-level
> values must be imported ONLY after independent evidence validation,
> not pasted as presumed results. The current physical study v4 has one
> Vivado 2023.2 process returncode 0 but an **unresolved ROM inventory
> provenance rejection**; physical results remain UNKNOWN. The historical
> explicit BRAM pragma did not consistently improve HLS estimated BRAM
> or cycles. Do not claim positive physical QoR, firstness, RTL proof or
> measured Fmax. The manuscript is NOT submission-ready until the
> `SCIENTIFIC_STUDY_PROTOCOL.md` preregistered mapping-conditioned
> discrimination, restricted-IR causal ablations, original fixed-point
> native qualification and direct prior-art artifact parity are complete.
> Preserve the prior E0/E1 tables as historical evidence with their exact
> revision rather than silently updating their denominators.
>
