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
## Archive policy

Many other files in this directory are **dated research notebooks**: experiment protocols, preliminary hypotheses, sprint plans, physical forensics, read-only audit reports and superseded manuscript drafts. They are preserved to protect the original evidentiary record and source citations. Do not interpret a proposed experiment in those files as a completed result.

Public-facing examples, contributions and programmatic interfaces belong in `docs/`, `examples/` and `CONTRIBUTING.md`, not in new research-sprint notes. Internal work logs and private native receipts must not be included in public source distributions. [Release preview](../../docs/RELEASE_PREVIEW.md) uses an allowlist that excludes `paper/` entirely.

**Public release and license approval are still pending.** This project has not been made public or relicensed by the documentation work.
