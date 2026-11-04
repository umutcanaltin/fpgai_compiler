# R2p — Common certified C3 decision interface across bounded source compiler families

**Date: 2026-11-03. Status: NEW code and tests committed; local workstation verification and original experiment NOT YET RUN.** This is *source-only* GCC and conservative E0 work; no trained RL, Vitis, Vivado, hardware DRC or GitHub Actions.

## Prior evidence preserved

- **R2n:** original revision `6296f99eb5d52371bffb9755989a340aadbe7ff6`, **9** action candidates across three sequential MatMul chains, **3** actual GCC E1-qualified, **3** compiler rejected, **3** source-admitted but E0 certificate UNKNOWN, independent saved-receipt audit agreed. Original user workstation full `1,464/1,464` passed.
- **R2o:** original revision `ce8083c0ad5b827232e1e67322811b25edd3ac79`, **12** action candidates across 2-, 3-, 4-reader MatMul fanouts, **9** actual GCC E1-qualified, **3** compiler rejected, **0** UNKNOWN/failed. Source/GCC event-address witnesses `102`, `210`, `372` per E1 execution for each graph respectively; independently checked immutable original source, raw stdout and receipt integrity audit. User workstation focused `27/27` and full `1,484/1,484` tests passed.
- **Native original R2i/R2j/R2l/R2m:** separate source-pinned 72-case HLS CSim/CSynth and selected six routed BRAM/LUT/FF/DSP/WNS comparisons (six BRAM ties, 0 strict clean-DRC E4). They are frozen and **not re-run or reinterpreted** by R2p.

## R2p implementation — one facade, two actual compiler domains

`src/fpgai_research/model_ir/unified_decision_interface.py` supports explicit discriminated complete-design states:

| Domain identifier | Underlying production path | Bounds/limitations |
|---|---|---|
| `sequential` | Existing `DecisionState` + `transition` + `graph_observation` + `source_executed_action_mask` | Supported complete static MatMul chain; source action may be accepted while E0 mapped-value proof remains UNKNOWN |
| `multi_reader_fanout` | Existing `FanoutDecisionState` + `fanout_transition` + `multi_reader_graph_observation` + `source_qualified_multi_reader_mask` | Exact static 2–4 reader MatMul fanout with left Add chain + Relu, reader PE and disjoint/optimal scratch edits |

**There are no new source compilers, allocators, architecture oracles or numerical/GCC checkers.** The facade delegates each typed `DecisionAction` once to its actual native compiler transition, checks state/CPP/certificate source hashes, and reports:

- `REJECTED`: typed edit refused, retains exact original complete state.
- `UNKNOWN_E0`: source compiler accepted, but bounded mapped-value E0 proof is absent. **Not certified and not legal merely because it compiled.**
- `BOUNDED_CERTIFIED_E0`: original domain compiler's supported E0 mapped-value/reader last-use memory certificate is present.
- `E1_GCC_QUALIFIED`: the original checker executed generated GCC source on one explicit finite host-float input, with original reference numerical and source-value/address observations.
- `FAILED_E1` and `UNKNOWN_E1_NOT_RUN`: no false E1 success promotion; original backend statuses are preserved in the canonical receipt.

`unified_graph_observation` converts the two existing **weight-private typed graphs** to a shared node/edge vocabulary, common `mapping`/`phi_spatial_probe`, tensor shape/reader count, and explicit domain-specific source memory or row-policy features. It drops original raw source SHA from policy feature outputs; `unified_state_hash` and decision receipts expose that identity **separately**. Graph identities and feature distributions are **not claimed equivalent across unrelated domains**. No actual cycle-level hardware scheduling or RL performance is inferred from spatial Phi probes.

`unified_action_mask` and `unified_source_mask` enforce a small common finite budget (**at most eight actions per batch; source E1 budget must cover all supplied candidates**), deliberately preventing the two existing backend-specific partial-budget conventions from being conflated. No candidate is silently dropped and no unexecuted action is called E1 qualified. The original domain-specific APIs remain unchanged.

## R2p prospective original GCC cross-domain check

The experiment module `src/fpgai_research/experiments/cst_r2p_unified_interface_study.py` uses a **new clean Git source revision**, create-only output and exactly:

1. One static **three-operator MatMul chain** with three typed edits: supported `mm0.pes=2`, invalid `mm0.pes=0`, and source-admitted but E0-uncertified `whole_tensor` policy.
2. One static **two-reader fanout** with three edits: supported `reader0.pes=2`, invalid `producer.pes=2`, and a certified `per_element_optimal` scratch layout.

These yield **six prespecified candidate actions**. E0 action statuses and source/CPP hashes from `unified_action_mask` must agree with the separately executed E1 mask on the same original state/action/input. Actual original E1 outcomes are **not known until the user executes** this new study. It writes `summary.json`, `sequential.json`, and `multi_reader_fanout.json` into a fresh folder, or retains `FAILED.json` on exception. Each source action includes the explicit input hash, origin revision, normalized feature observation digest and underlying original source/GCC receipt.

The module's `--study` read-only audit re-checks source revision, original fixed inputs, graph-observation hashes, six fixed action values, original source CPP hashes and all E0/E1 status denominators. **It does not replay original GCC or independently revalidate numeric semantics**. The earlier original R2n/R2o validation paths remain the original source-oracle basis.

## Local verification and prospective original execution

Use the **existing** clean research Git checkout. Do not delete, overwrite or rerun any original R2n/R2o/R2i/R2j/R2l/R2m result.

```bash
cd ~/Desktop/github_projects/fpgai_research
git status --short
git pull --ff-only origin main
python -m compileall -q src/fpgai_research experiments/compiler_ir tests

python -m pytest -q \
  tests/test_cst_r2p_unified_decision_interface.py \
  tests/test_cst_r2p_unified_interface_study.py \
  tests/test_compiler_ir_series.py

python -m pytest -q
```

If the tests pass, perform the **new** create-only source experiment and independent read-only audit:

```bash
python -m fpgai_research.experiments.cst_r2p_unified_interface_study \
  --output build/r2p_unified_c3_original_gcc_v1

python -m fpgai_research.experiments.cst_r2p_unified_interface_study \
  --study build/r2p_unified_c3_original_gcc_v1 \
  --audit-output build/r2p_unified_c3_original_gcc_audit_v1.json
```

All output paths are create-only; choose a new `v2` path if a prior path exists, **never erase original artifacts**. R2p is also in quick and paper compiler-research plans as `unified_policy_interface_e1` after both source C3 predecessors. Neither local run invokes `rl_environment/`, vendor tools, or GitHub Actions/tokens.

## Acceptance criteria / scientific limits

This research sprint succeeds in *interface integration* only if source identities, E0 source legality, GCC E1 outcomes and source graph features remain consistent with the independently validated native compiler pathways. Its empirical study must report the full six-action denominator and all UNKNOWN, REJECTED and failed cases **without hiding them**.

It does **not** prove arbitrary independently mapped DAGs, cross-clock time semantics, hardware-port feasibility, unconditional exact allocation optimality for all schedules, end-to-end numerical correctness on all inputs, or RL algorithm/sample-efficiency improvement. A normalized API or typed graph is not itself a novel scientific theorem. The secondary C3 claim is policy-independent **reuse of an independently checkable compilation contract**, within bounded static topologies. The primary scientific contribution still requires a precise relation between `Phi`, tensor-element readiness, all readers' last use and actual source address safety, plus prior-art differentiation.

For future work, extend this shared state/action vocabulary to **additional compiler-backed domains only when their legality/host-source checkers exist**, and distinguish `UNKNOWN` from valid actions at every evidence tier. Keep the already published physical BRAM/WNS results prominent; do not pretend this source-only sprint measures physical QoR.

See [verified R2o original receipts](R2O_C3_MULTI_READER_DECISION_SOURCE_PROTOCOL_20261103.md), [verified R2n original receipts](R2N_C3_E1_QUALIFIED_DECISION_IR_20261103.md), [research manuscript](MANUSCRIPT_DRAFT.md), and [independent scientific review request](../../SCIENTIFIC_REVIEW_README.md).
