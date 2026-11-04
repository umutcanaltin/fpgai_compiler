## 2026-11-03 — R2o ORIGINAL 12-action multi-reader GCC study and read-only audit WORKSTATION VERIFIED

Source-pinned checkout `ce8083c0ad5b827232e1e67322811b25edd3ac79`, clean tracked state. User ran **27/27 focused** tests and **1,484/1,484 full pytest** (138.32 s), then the original **create-only** `build/r2o_c3_multi_reader_original_gcc_v1/` experiment on **three** prespecified static MatMul fanout graphs with **2, 3 and 4 independently mapped readers** and four actions per graph. Results: **12/12 original outcomes recorded; nine E1_GCC_MULTI_READER_QUALIFIED; three REJECTED invalid producer-PE actions; zero UNKNOWN and zero failed E1 executions; nine original GCC executions**. This is **nine qualified proposals, not 12 successes**. The two-reader, three-reader and four-reader graph arms respectively recorded **102, 210 and 372** source-address witnesses per qualified GCC execution. All three original roles per graph that were E1 attempted passed their selected single fixed host-float input numerical, real source-address, and mapping/lifetime checks.

Independent *read-only* source/manifest/stdout SHA receipt auditor `build/r2o_c3_multi_reader_original_gcc_audit_v1.json` reported identical original source revision, action statuses, case and witness populations. This receipt integrity audit **does not independently replay GCC or numerical semantics**. No original Vitis/Vivado or RL training calls occurred. R2o source-only E1 is not physical memory, achieved Fmax, power, HLS bank-port feasibility, clean-DRC E4, arbitrary DAG legality, all-input mathematical equivalence, or learned RL gain.

**FREEZE** these original R2o outputs at source `ce8083c...`; do not overwrite/re-run to revise outcomes. Earlier R2n original nine-action GCC campaign and R2i/R2j/R2l/R2m results remain independently frozen. The next compiler sprint may create a policy-independent shared facade over bounded row-chain and multi-reader adapters, but must not silently call this a generalized arbitrary-DAG source compiler. [Original R2o protocol](R2O_C3_MULTI_READER_DECISION_SOURCE_PROTOCOL_20261103.md).

## 2026-11-03 — R2o first workstation smoke: 25/26 focused, 1482/1483 full; negative-test assertion corrected (RETEST PENDING)

At original source checkout `2ac4f4fb50728e81bdb45b13c93c9a4bff5f580c`, the workstation ran source compilation, focused pytest (**25 passed, 1 failed**) and full pytest (**1482 passed, 1 failed**). The sole failing test was the deliberate forgery negative control `test_forged_e1_status_without_e1_receipt_rejected`. The auditor **correctly rejected** the forged E1 action at the earlier missing-E0-certificate condition `R2o E0-admitted original action lacks certificate`, while the test incorrectly expected the later `original GCC receipt absent` message. This is a **test specificity failure**, not evidence that the forged E1 action was accepted or that actual source GCC qualification failed.

Follow-up source revision `fb805af2f0f588d256f0985389272720aa0c1b56` fixes the expectation for the original malformed-E0 forgery and adds a second negative control with syntactically populated E0 SHA fields but no GCC receipt, explicitly exercising the **distinct E1-missing-receipt** rejection. It does not change R2o production compiler or original HLS/Vivado code. These new assertions are **not yet reverified on the user's workstation**. The **original prospective R2o 12-action GCC experiment has not run** and no R2o outcome counts are yet recorded. Run the tests again before executing it, with the original source-pinned R2n/R2i/R2j/R2l studies frozen.

# R2o — Certified policy-independent decisions on bounded multi-reader graphs

**Date: 2026-11-03. Status: source implementation and tests committed, ORIGINAL workstation verification pending.** R2n original source evidence is **complete and frozen** at commit `6296f99eb5d52371bffb9755989a340aadbe7ff6` (37/37 focused, 1,464/1,464 full pytest, nine original actions: **three E1-qualified, three REJECTED, three UNKNOWN_E0_CERTIFICATE**; independent read-only audit agrees). R2o is a **new experiment and source revision**, not retroactive qualification of those original R2n observations.

## Scientific question

Can the secondary C3 RL-/search-compatible interface use the same **actual source compiler and bounded Phi/last-reader memory certificates** for a typed graph with **multiple independent consumer mappings**—without silently converting source legality into hardware feasibility?

We do **not** attempt general arbitrary DAG compilation in this sprint. The exact tested Functional topology is:

```text
X @ Wp -> H -> {H @ Wc0 -> A0, H @ Wc1 -> A1, ... H @ Wcr -> Ar}
           -> left-associative Add chain -> Relu -> Y
```

There are **2, 3 or 4 MatMul readers**, independently chosen effective reader-PE mapping parameters, a scalar producer, scalar Add and Relu, and one complete deterministic source event order. The existing fanout certificates verify producer contributions, each reader's distinct Phi issue, all reads, commits, last-use and scratch-address safety. The corresponding R2h compiler emits **actual loop-driven GCC-executed C++** whose scratch address is identical to its source witness and whose intermediates/final values are checked against independent NumPy. We reuse those existing components rather than recreate the allocator or trace verifier.

## Implementation

The new sibling adapter `src/fpgai_research/model_ir/fanout_decision_interface.py` uses the **existing typed** `FunctionalModel`, `ArchitectureRequest`, `TargetLimits` and **same `DecisionAction` type** as the earlier row-chain C3 prototype, but a new constrained `FanoutDecisionState`. This is **not yet a single unified complete-design API for arbitrary row chains and fan-out graphs**.

- `fanout_transition(state, action)` applies one independently mapped **reader `pes`** edit or a global scratch layout edit (`disjoint` / `per_element_optimal`), calls the actual `resolve`, `source_layout` and `emit_allocated_fanout_cpp` implementations, and returns `FanoutAccepted` with source-pinned E0 certificate or `FanoutRejected` with original unchanged state. Unsupported producer, reader SIMD/tile/pipeline, unknown operators and malformed edits are not legalized by a shim.
- `source_qualified_multi_reader_mask` passes each proposal through that same production compiler transition. For accepted candidates within an explicit finite host-GCC budget, it compiles and runs actual R2h source via `verify_allocated_gcc`; only matching source, architecture, slot map and original stdout hashes with valid intermediate numerical and address observations yield `E1_GCC_MULTI_READER_QUALIFIED`. When unexecuted, it reports `UNKNOWN_E1_NOT_RUN`; compiler-invalid actions report `REJECTED`; attempted but unsuccessful source tests report `FAILED_E1_SOURCE_EXECUTION`. Evidence-tier qualification, not a model-driven performance reward.
- `multi_reader_graph_observation` exposes a variable-size typed node/edge graph, per-operator effective mapping and Phi spatial probes, source scratch capacity, target limits and fan-out number—**no constant tensor payloads, trained weights, post-HLS metrics, assumed cycle times or RL rewards**.
- The output optionally pins the **exact original generated C++**, emitter JSON manifest and raw GCC stdout for each attempted E1 case into a create-only evidence directory, with SHA-256 validation.

No Vitis, Vivado, RL training or any automation token is required.

## Prespecified original R2o campaign

`src/fpgai_research/experiments/cst_r2o_multi_reader_action_study.py` defines exactly **3 static graph shapes** and **4 actions per graph**:

| Fanout degree | (M,K,N,Q) | Typed action roles |
|---|---|---|
| 2 readers | (2,2,3,4) | First reader PE=2; last reader PE=2; invalid producer PE=2; optimized scratch layout |
| 3 readers | (2,3,5,4) | Same four roles |
| 4 readers | (3,3,4,4) | Same four roles |

**Twelve planned candidate actions**, with three planned E1 attempts per graph if production E0 admits them; actual success/failure/unknown/rejection counts are **not yet known**. Preserve all outcomes, including compiler rejection and any failed GCC traces. Input is fixed and source-SHA pinned, with clean tracked Git state checked *before* starting original GCC. Original creation path is new and unused; source, manifests and stdout are stored as `graph_NNN/case_NNN_*.cpp/json/txt` plus case receipts.

The companion `cst_r2o_multi_reader_action_audit.py` separately re-reads the immutable original study folder and verifies exact cases, actions, revision, fixed input digest, recorded source/manifests/stdout SHA, result-tier denominators and status consistency. **This auditor is NOT a second GCC execution or full independent numerical proof.**

## Workstation instructions (do not run old native campaigns)

In the same existing repository, first verify code and tests:

```bash
cd ~/Desktop/github_projects/fpgai_research
git status --short
git pull --ff-only origin main
python -m compileall -q src/fpgai_research experiments/compiler_ir tests
python -m pytest -q \
  tests/test_cst_r2o_fanout_decision_interface.py \
  tests/test_cst_r2o_multi_reader_action_study.py \
  tests/test_compiler_ir_series.py
python -m pytest -q
```

Only after tests pass, and with the original R2n and all R2i/R2j/R2l outputs frozen, execute a **new** original GCC source experiment:

```bash
python -m fpgai_research.experiments.cst_r2o_multi_reader_action_study \
  --output build/r2o_c3_multi_reader_original_gcc_v1 \
  --graphs 3

python -m fpgai_research.experiments.cst_r2o_multi_reader_action_audit \
  --study build/r2o_c3_multi_reader_original_gcc_v1 \
  --output build/r2o_c3_multi_reader_original_gcc_audit_v1.json
```

The output paths are **strictly create-only**. Never overwrite original `build/r2n_c3_e1_qualified_actions_v1`, R2m original physical tables, R2i/R2j/R2l original vendor receipts or any RL experiment. The script requires a clean tracked source revision. If any candidate fails, **retain failure files** and stop rather than deleting or patching historical evidence.

The new `multi_reader_certified_actions_e1` experiment stage is part of both compiler-paper quick and paper profiles; `quick` uses one fanout degree and `paper` uses all three. Running `--plan` is read-only; there is no need to execute the whole prior paper experiment series to validate R2o.

## Evidence and remaining publication gates

- This is **bounded source E0/E1 evidence**: exact 2–4 independent MatMul reader mappings and memory aliasing with real GCC event/address witnesses. It is stronger than R2n's single-reader limitation, but is **not arbitrary DAG**, mixed Conv/Attention, asynchronous/hardware temporality, RAM-port conflict safety or all-input soundness.
- Reader PE edits may change the source issue order, last-use ranks and allocation capacity. An observed hash or source difference alone does not prove **causal memory improvement**; retain counterfactual and equal-functional-graph controls. Do not claim causal resource savings unless measured.
- A C3 interface plus a GNN-ready observation is **not automatically a second novel RL algorithm or statistically supported improvement**. Policy independence must be distinguished from learning ability and physical feasibility.
- The already-completed physical results remain **6/6 selected routed BRAM ties**, **6/6 DSP ties**, mixed LUT/FF/WNS and **0/6 strict clean-DRC E4**. R2o is GCC-only and does not change them.
- The next genuine scientific gaps are **unification** of the two source-backed decision adapters, formal statement/proof for supported graph composition and source refinement, and further independently falsifiable cross-Phi tests. Claude's adversarial review may require narrowing claims; never silently rewrite original experiment results.

See [main manuscript](MANUSCRIPT_DRAFT.md), [R2n prior C3 experiment](R2N_C3_E1_QUALIFIED_DECISION_IR_20261103.md), [source-pinned scientific status](CURRENT_SCIENTIFIC_STATUS.md), [independent review instructions](../../SCIENTIFIC_REVIEW_README.md).
