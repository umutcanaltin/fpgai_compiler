# Compiler IR research and scientific evidence

This directory contains the extended compiler-paper research history, experiment protocols, proof sketches and manuscript work. It is **not the compiler user guide**. For installation and usage start at [FPGAI Research](../../README.md) and [Using FPGAI](../../docs/USING_FPGAI.md).

**Active first-paper work (6 November 2026):** [C1/C2 active manuscript](C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md) and **canonical [evidence-to-claim / submission-lock ledger](FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md)**. Consult the [scientific completion and experimental history](FIRST_PAPER_SCIENTIFIC_COMPLETION_20261106.md) for original study provenance, not as the current manuscript. The RL branch remains paused and trained-agent benefits are **outside this first paper**. Evidence is substantial but independent competitor novelty, generated-source refinement and C2 cost claims remain unresolved; **MAJOR REVISION / NOT SUBMISSION READY**.

## Scientific scope

The current compiler-paper work investigates two related questions:

- **Bounded compositional source-event semantics:** explicit tensor-element ownership, production readiness, all-reader last use, source scratch reuse and operator-local issue order in supported static MatMul families.
- **Policy-independent compiler decisions:** typed complete-design architecture actions, source certification/rejection/unknown states, and stable graph observations suitable for future optimization methods.

Neural-network HLS source emission and separately source-attributed CNN/Dense HLS studies examine implementation behavior. Earlier independent MNIST MLP/CNN physical routing work supplies a **distinct** E3/E4 observation cohort; do not combine physical timing/resource fields with the later convolution–dense HLS-only population.

Source certification is not formal proof of arbitrary graphs and does not imply physical feasibility. The separately routed R4 matched-storage study has **descriptive** BRAM tiles 7→4 but adverse LUT/FF/WNS effects and two timing-failed, nonclean-DRC designs; physical compute latency, real throughput and board end-to-end latency remain **UNKNOWN**. [Resource and WNS source ledger](R4R_SIGNED_OOC_TRADEOFF_LEDGER_20261106.md). A trained-RL performance contribution is outside this compiler paper.

## Scientific documentation

| Purpose | Entry point |
|---|---|
| **Current evidence-to-claim matrix and publication stop rules** | [First-paper submission lock](FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md) |
| Scientific questions and what remains unproven | [Final scope and acceptance gates](C1_C2_COMPILER_IR_FINAL_SCOPE_AND_ACCEPTANCE_GATES_20261103.md) |
| Draft scientific argument | [Two-contribution working manuscript](C1_C2_TWO_CONTRIBUTION_MANUSCRIPT_DRAFT_20261103.md) |
| Bounded finite operational semantics and proof obligations | [Source-event semantics](C1_FINITE_EVENT_SEMANTICS_AND_PROOF_SKETCH_20261104.md) |
| Independently inspected compiler prior art | [Pinned prior-art reconnaissance](C1_C2_PINNED_PRIOR_ART_RECONNAISSANCE_20261103.md) |
| Source-event and action classification original receipts | [Original source observations](R3G_R3H_ORIGINAL_C1_C2_WORKSTATION_RESULTS_20261104.md) |
| Isolated external compiler comparison protocol | [Allo capability runbook](R3I_ISOLATED_ALLO_CAPABILITY_RUNBOOK_20261104.md) |
| Scientific publication status | [Submission gates](SUBMISSION_DECISION_GATE_20261103.md) |
| Safe reproducibility terminology | [Public scientific reproducibility guide](../../docs/SCIENTIFIC_REPRODUCIBILITY.md) |

## Latest internally reported observations

In the author's 4 November 2026 original workstation report, the current regression suite passed **1,814** tests. A held-out bounded event semantics study evaluated **24 architecture/model profiles and 216 source-event variants** with **zero observed disagreements** against its separate reference interpreter. A target-limit intervention used **12 matched static model pairs, 192 classified source proposals and 24 classification changes**. The read-only source/native evidence aggregator successfully inspected six previously saved study cohorts, while retaining `independent_external_compiler_parity=NOT_ESTABLISHED` and `submission_ready=false`.

These are **separate, finite source studies**; they are not proof of correctness on every possible graph, RTL schedule or physical device.

**6 November 2026 original source-predicate verification:** The author ran **43/43** focused CPU-only tests and created the source-pinned `build/first_paper_contract_predicate_ablation_20261106_01` for **24 bounded source profiles / 216 prespecified traces**; independent local SHA/row-count inspection confirmed the saved files. In the matched R3j-derived predicate comparison, **24** deliberately issue-order-invalid but value-safe controls pass the value/lifetime-only test; **zero** pass when the `Phi` issue-order predicate is conjunctively required. This demonstrates a **bounded rule necessity control**, not separately compiled A/B versions of the compiler or an overall performance benefit. Full source/version limitations are in the [canonical paper audit](FIRST_PAPER_EVIDENCE_CLAIMS_AND_SUBMISSION_LOCK_20261106.md). Neither Vitis nor Vivado ran.
**Verified original C1/C2 scientific strengthening (6 November 2026):** [Falsification-first protocol and experiment provenance](SCIENTIFIC_STRENGTHENING_C1_C2_PROTOCOL_20261106.md). At exact source revision `637f2745721b32d6f5a7e02c4659b0623bc9eee7`, the author passed **72/72 focused tests** and executed create-only studies: **54 new bounded fanout profiles / 486 trace checks / 0 independent source-event oracle disagreements** (original report SHA `8fa3848fcb0993e44a83391f595861943d75e3f09b669bd93b1b27e0222f3978`), and **324 matched direct/lazy/eager source-compiler method observations / 48 excluded preflight compiler calls** (original report SHA `be2adaa4acf30196a2ccb17114f2caa0af3e6679dc2685872e27cb086ca86ec1`). Both passed author-run read-only source/SHA audits; no vendor calls. These observations expand *in-family* C1 evidence and establish a C2 CPU-cost **dataset**, but the per-method times have **not yet been summarized**, and neither proves general soundness, source-to-RTL refinement, optimization/learning speedup, or qualified physical latency. A **new, untested** hash-pinned read-only C2 timing-analysis module has been added. The paper must distinguish source schedule **conformance** from functional safety and compiler decision echoes from independently established novelty.
**Next falsification gate — not yet executed:** [Pinned C1 held-out event-to-generated-GCC refinement protocol](C1_E0_E1_HELDOUT_CPP_BRIDGE_GATE_20261106.md). Source-only bridge to the existing real-GCC R2h scratch/numerical verifier, original 54-profile E0 SHA population, two actual source layouts per profile (108 planned GCC executions), clean tracked Git preflight, immutable new attempt output. This is distinct from the previously completed 486 E0 source-event checks and from earlier independent R2r E1 mutations; no new C1 E1 results have been measured. The repo also includes [a static file ownership/import inventory tool](../../scripts/audit_source_inventory.py) to reduce duplicate implementations and assess change impacts without deleting original experiments.

**Scientific novelty and rejection controls:** [Adversarial reviewer claim matrix](SCIENTIFIC_REVIEWER_ADVERSARIAL_CLAIM_MATRIX_20261106.md) distinguishes functional equality, declared source-order conformity, real emitted GCC correctness, source/physical performance, and executed exact external Allo/CODO equivalence. **This is a preregistration, not an observed study**. The paper must not claim to be the first schedule verifier or hide the C2 eager-vs-lazy non-superiority bound.


**Verified new original C1 E1 and C2 CPU outcome (6 November 2026):** [Frozen original C2 six-condition source-cost results and C1 two-profile real GCC E1 smoke](C1_E1_SMOKE_C2_SOURCE_COST_ORIGINAL_RESULTS_20261106.md). Lazy beat eager in all 108 matched source CPU trials (198 vs 864 measured compiler calls), and the separately source-E0-pinned C1 smoke completed **4 genuine GCC compilations and executions** over 2 source profiles, both with positive source scratch slot saving and an 11-artifact SHA audit. The full 54-profile E1 trial remains **UNEXECUTED**; neither result provides physical FPGA clock/latency or external scientific novelty qualification.

## Archive policy

Many other files in this directory are **dated research notebooks**: experiment protocols, preliminary hypotheses, sprint plans, physical forensics, read-only audit reports and superseded manuscript drafts. They are preserved to protect the original evidentiary record and source citations. Do not interpret a proposed experiment in those files as a completed result.

Public-facing examples, contributions and programmatic interfaces belong in `docs/`, `examples/` and `CONTRIBUTING.md`, not in new research-sprint notes. Internal work logs and private native receipts must not be included in public source distributions. [Release preview](../../docs/RELEASE_PREVIEW.md) uses an allowlist that excludes `paper/` entirely.

**Rights-cleared licensed redistribution remains blocked.** The repository may already be publicly viewable, but this work did not change visibility or grant a license. History/rights review and a definitive release decision remain mandatory.
