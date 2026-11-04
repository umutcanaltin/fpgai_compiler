## 2026-11-03 — R2n C3 ORIGINAL GCC campaign and independent audit WORKSTATION VERIFIED

At original tracked revision `6296f99eb5d52371bffb9755989a340aadbe7ff6`, user executed source validation: **37/37 focused pytest**, **1,464/1,464 full pytest** in 137.17 seconds, then new original create-only GCC `build/r2n_c3_e1_qualified_actions_v1/` for three static MatMul chain depths **2, 3, 4** with **9/9 complete candidate actions**. Each depth had **1/3 `E1_GCC_SOURCE_QUALIFIED`**, **1/3 source compiler `REJECTED`**, **1/3 `UNKNOWN_E0_CERTIFICATE`**. Across all three: **3 E1 qualified** with real GCC source lifetime/numeric checks, **3 rejected**, **3 E0 unknown**, **0 failed source E1** and **3 actual source E1 executions attempted**. User then ran `cst_r2n_e1_action_audit` as a separate read-only study (`build/r2n_c3_e1_audit_v1.json`), independently reconfirming counts, input/source SHA and evidence-tier status consistency. Audit **did not execute GCC**. Original original Vitis, Vivado and RL training calls **0**. This validates **bounded static single-reader row-chain source** decision evidence only; no general multi-reader certified action interface or learned RL gains.

Experimental source revision and original artifacts are **frozen**. Do not rerun, overwrite, or delete either R2n source or audit. Subsequent new code is a new source revision and is NOT retroactive proof for the original R2n. See [R2n original protocol](R2N_C3_E1_QUALIFIED_DECISION_IR_20261103.md) and [primary manuscript](MANUSCRIPT_DRAFT.md).

# R2n — C3 policy-independent E1-qualified architecture decision IR

**Date: 2026-11-03. Status: source and tests committed; workstation execution PENDING.** No GitHub Actions, tokens, CI, trained RL, original Vitis or Vivado required.

## Scientific purpose

The new policy-independent `source_executed_action_mask` in `src/fpgai_research/model_ir/decision_interface.py` uses one production source compiler transition per candidate and optionally performs real GCC-executed mapped-value trace and host-float numerical qualification. It retains four distinct outcomes:

| Outcome | Meaning | Limit |
|---|---|---|
| `REJECTED` | Actual compiler transition rejects typed action | Not universal hardware infeasibility |
| `UNKNOWN_E0_CERTIFICATE` | Source compiles but supported mapped-value E0 proof is absent | Must never be marked certified |
| `FAILED_E1_SOURCE_EXECUTION` | E0 qualified, but GCC source trace/host numeric execution failed or is incomplete | Do not discard negative evidence |
| `E1_GCC_SOURCE_QUALIFIED` | E0 bounded mapped-value contract plus real generated-source GCC lifetime and numerical checks on a fixed input | Not all-input proof, fixed-point HLS, FPGA timing, port legality, or RL gains |

The GCC source and original action certificate hashes must match. Contradictory source provenance is a **fatal error**, never an ordinary failure/unknown. No native vendor calls, reward, RL agent, training or learned policy are involved.

## Prospective controlled R2n experiment

`src/fpgai_research/experiments/cst_r2n_e1_action_study.py` uses three small static MatMul chains of operator depths **2, 3, 4**, each with three action *categories*: a supported first-operator PE=2 edit, an invalid PE=0 edit, and a whole-tensor exchange request whose source may compile but whose row-chain value-level E0 certificate is absent. This yields nine prespecified action cases. All outcomes, successes, rejects, unknowns, actual GCC source hashes/input digests and failures are recorded. **Do not assert prospective outcome counts until original workstation execution.** Inputs are deterministic, finite, shape-bound and SHA-pinned.

Actual selected GCC source traces use the existing source execution checker and independent NumPy host-float reference. This improves C3 source-level evidence beyond a static compile/admission mask. It does **not** establish that the existing complete-design row-chain API supports arbitrary fan-out DAGs, multi-clock HLS, RTL proofs, hardware resources or RL generalization.

`tests/test_cst_source_executed_action_mask.py` includes real GCC positive qualification plus unknown/rejected/failed-source and provenance-forgery controls. `tests/test_cst_r2n_e1_action_study.py` covers one original GCC case, all-denominator preservation, create-only outputs and native-claim rejection. Both new modules are included in both quick and paper source-only experiment profiles through `experiments/compiler_ir/series.py`; `rl_environment/` is untouched.

## Workstation tests and create-only study

Use the **existing clone**, preserving all original R2i/R2j/R2l build artifacts and original physical receipts.

```bash
cd ~/Desktop/github_projects/fpgai_research
git status --short
git pull --ff-only origin main
python -m compileall -q src/fpgai_research experiments/compiler_ir tests
python -m pytest -q tests/test_cst_source_executed_action_mask.py tests/test_cst_r2n_e1_action_study.py tests/test_cst_decision_interface.py tests/test_cst_rl_interface_study.py tests/test_compiler_ir_series.py
python -m pytest -q
```

Only after tests pass, run the new study:

```bash
python -m fpgai_research.experiments.cst_r2n_e1_action_study --output build/r2n_c3_e1_qualified_actions_v1 --graphs 3
```

The new **create-only** `build/r2n_c3_e1_qualified_actions_v1` folder contains `summary.json` and `case_evidence.json`. Do not overwrite; if output exists, use a new versioned output name. Never rerun original Vitis or Vivado experiments.

Dry-run planning of the integrated CPU workflow (read-only):

```bash
python experiments/compiler_ir/series.py --profile quick --plan --skip-onnx
```

## Separate read-only receipt integrity audit (newly implemented; workstation pending)

A second module, `src/fpgai_research/experiments/cst_r2n_e1_action_audit.py`, does **not** rerun GCC. It checks the saved `summary.json` and `case_evidence.json` against the prespecified 3-actions-per-graph cohort, Git revision, input/source SHA identities, per-action outcome tiers, and numerical/trace receipt completeness. It rejects forged E1 statuses, UNKNOWN-to-valid promotions, count manipulation, source-treatment edits and symlinked inputs. It is intentionally an **integrity re-audit**, not a second independent execution of mathematical/numeric semantics.

After the R2n source experiment succeeds, run:

```bash
python -m fpgai_research.experiments.cst_r2n_e1_action_audit \
  --study build/r2n_c3_e1_qualified_actions_v1 \
  --output build/r2n_c3_e1_audit_v1.json
```

The audit output is create-only, and both original study JSON files remain untouched. New `tests/test_cst_r2n_e1_action_audit.py` supplies six source-only integrity/falsification tests. If any E1 source execution fails, retain the failure in the study and the audit; do **not** remove it to make the result appear successful.

## Scientific limits and remaining sprints

- **C1/C2 integration gap:** general independent-Phi, multi-reader DAG architecture action edits do not yet flow through this bounded single-reader `DecisionState` action mask. This is a future source-backed correctness sprint, not an achieved result.
- **C3 semantics:** current mask is E0 bounded or E1 single-input GCC on supported complete row-chain designs. No partial `UNASSIGNED` / `STOP`, general bank/port constraints, achieved HLS timing or physical feasibility.
- **Generalization:** graph observation's anonymous typed features and bounded renaming tests do not prove arbitrary graph isomorphism equivariance, transfer or learned sample efficiency.
- **Scientific originality:** CompilerGym, RL4ReAl and optimization-driven compiler interfaces are prior art. A usable typed action graph/mask is not automatically a new RL algorithm or second scientific theorem.
- **Frozen RL:** trained shared-GNN/real-HLS R5 comparison remains a different future study. Do not change `rl_environment/` or original L0 observations now.
- **Original HLS/physical:** source-pinned 72-case R2i E2/E3 and original six routed R2j/R2l pairs with six BRAM ties and zero clean-DRC E4 remain separate and frozen.
- **Independent Claude review:** use its prospective analysis to correct hypotheses after review, without inventing experimental outcomes or retroactively redefining original cases.

**Proposed C3 manuscript wording, after verified R2n:** “We expose a bounded policy-independent compiler decision interface with explicit source-admission, mapped E0 certificate, and independently executed E1 host-source qualification status on several static MatMul graphs. It does not train an RL agent or prove HLS/physical feasibility.”

See [RL-ready IR research contract](RL_READY_IR_NOVELTY_AND_EXPERIMENTS.md), [C1/C2/C3 contribution contract](PAPER_CONTRIBUTIONS_AND_RL_INTERFACE.md), [certified decision sprints](CERTIFIED_DECISION_IR_SPRINTS.md), [Claude review](../../SCIENTIFIC_REVIEW_README.md) and [R2m physical results](R2M_FINAL_ORIGINAL_PHYSICAL_PAPER_EXPORT_20261103.md).
