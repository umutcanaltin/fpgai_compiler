<!-- Historical preservation copy, not current community guidance. -->

# Contributing to FPGAI compiler and original-Vitis Gym research

FPGAI is a research compiler and reproducible hardware-optimization benchmark. Researchers can contribute **new DRL algorithms**, **new native-HLS Gym experiments**, **new compiler operators/layers**, tests, analysis, and hardware toolchain support. No external contributions should be described as accepted until the repository owner has approved them.

> **Distribution:** This repository has been maintained privately and does not have a grant of external redistribution rights here. Confirm intellectual-property and contribution permissions with the maintainers before submitting third-party source, datasets, weights or proprietary compiler/tool output. This guide is not a software license.

## 1. Development environment and source protection

Use a Python virtual environment, from the **actual checkout you are editing**:

```bash
cd ~/Desktop/github_projects/fpgai_research
python -m pip install --no-deps -e .
python -m rl_environment.runtime_contract
python -m compileall -q src rl_environment tests
python -m pytest -q
python -m rl_environment.gym_curriculum --list-stages
python -m rl_environment.gym_curriculum --list-experiments
python -m rl_environment.gym_curriculum --plan-all
```

The source-binding preflight checks that native Gym uses **this checkout's** `src/fpgai_research`, not an old editable installation or other checkout in `PYTHONPATH`. `rl_environment/__init__.py` places its own sibling `src/` first without running vendor tools. If the compiler is *already* loaded from a different checkout, `runtime_contract` still fails closed: restart the Python process with a clean environment. Do **not** delete existing original HLS case directories in an attempt to repair imports.

An ordinary import, dry-run, independent audit, experiment registration, or unit test **must not** invoke Vitis/Vivado. Real HLS is initiated only by explicit foreground `--execute-native`, a fresh create-only output directory, and an authorized native-call budget.

## 2. Code ownership and layout

See [Research Extension Architecture](docs/RESEARCH_EXTENSION_ARCHITECTURE.md) for the component map.

| Research area | Stable entry point | Source of truth |
|---|---|---|
| Agent algorithms | `rl_environment/agents/` | `agents/api.py`, `agents/registry.py` |
| Gym environments | `rl_environment/environments/` | `environments/registry.py` |
| Experiment protocols | `rl_environment/experiments/` | `experiments/registry.py`, `curriculum.py` |
| Trained/untrained comparisons | `rl_environment/evaluation/` | `gym_audit.py`, `gym_curriculum_report.py` |
| Native execution, code provenance | `rl_environment/infrastructure/` | `runtime_contract.py`, `post_hls_env.py`, `timing.py` |
| Compiler pipeline | `src/fpgai_research/compiler/` | `compiler/pipeline.py` |
| Model semantics | `src/fpgai_research/model_ir/` | `contracts.py`, `semantics.py`, `lowering.py` |
| Operator import & shapes | `src/fpgai_research/frontend/` | `onnx.py`, `attributes.py` |
| Hardware architecture legality | `src/fpgai_research/architecture/` | `resolver.py`, `scopes.py` |
| HLS source and native Tcl | `src/fpgai_research/codegen/hls/`, `toolchains/vitis_hls/` | `emitter.py`, `tcl.py`, `package.py` |
| Numerical reference | `src/fpgai_research/validation/` | `reference.py`, `samples.py` |

Existing public imports such as `rl_environment.gym_agent.OnlineGraphTrainer`, `rl_environment.gym_env.GymGraphEnv`, `rl_environment.gym_study.run` and `fpgai_research.toolchains.vitis_hls.package.make_vitis_tcl` remain compatible during incremental refactoring. **Don't move these to new paths or delete historical APIs in a drive-by cleanup.** New modules are structural boundaries and registries; source/test migration is incremental.

## 3. Contributing a new DRL algorithm

1. Implement a **from-scratch** algorithm under `rl_environment/agents/`, for example `ppo_graph.py`, with the contract in [agents/api.py](rl_environment/agents/api.py):
   - `choose(observation, *, training, sample)` returns `((architecture_slot, requested_clock_ns), action_tape)`. Action slots must respect the source-provided legal mask.
   - `update(action_tape, reward, *, next_obs, terminated, return_metrics, qualified)` consumes the *original* HLS transition and returns diagnostics when requested. No fabricated QoR.
   - `save_private(directory, metadata)` creates a source-hashed private checkpoint, not an overwrite; `load_private(directory, *, load_optimizer)` verifies checkpoint integrity. No external pretrained agents in this v1 curriculum.
2. Add one **explicit** source-controlled factory entry in `rl_environment/agents/registry.py`, for example:
   ```python
   def _make_ppo(*, seed: int, use_message_passing: bool):
       from .ppo_graph import GraphPPO
       return GraphPPO(seed=seed, use_message_passing=use_message_passing)

   register_agent("graph_ppo", _make_ppo)
   ```
   Do not register a stub or unimplemented class. Runtime CLI dynamically lists only names registered in **source code**; no `eval`, remote plugin installs or environment-variable-based imports.
3. Test CPU-only initialization, masked action choice, scalar update, source/checkpoint hashes, deterministic seeded restart, train/frozen mode and missing-source failures.
4. Compare your algorithm to `graph_actor_critic` with the **same** stage, seed cohorts, 100×10 train budget, original Vitis source verification, and equal-budget independent heldout test arms. Output to separate create-only study roots. Use:
   ```bash
   python -m rl_environment.gym_curriculum --stage L0 \
     --agent-algorithm graph_ppo  # after registering and testing that algorithm
   ```
   This command only prints the planned run until `--execute-native` is explicitly added. A different algorithm, hyperparameter set or pretrained strategy requires a separately source-pinned protocol and report label.

## 4. Contributing a new experimental setup

Separate **task distribution, action domain, reward, budget, and evaluation**. Do not hardcode a second parallel implementation of `post_hls_env.py` just to make a stage look successful.

- For a new variant of an existing `L0..L6` stage, register a deterministic factory in `rl_environment/experiments/registry.py`. It must return the fields required by `gym_study.plan`, including `training`, `validation`, `test`, `calls_per_network`, `seeds`, `clocks_ns`, `freeze_numeric_format`, `beginner_pilot`, and `curriculum_stage`.
- The built-in factory `experiment_config("curriculum_l0", train=100, validation=1, test=1, calls=10, seed=111)` is a working reference. A research variant can use the same trusted stage while changing a separately declared seed or number of training networks. **The full curriculum still enforces at least 100 training episodes with exactly 10 original HLS trials per episode**.
- For a genuinely *new* difficulty level, add a new, explicitly versioned stage contract in `curriculum.py`, its Functional Model IR task generator in `cohort.py`, native action-legality mask in `post_hls_env.py`, independent reward validation in `gym_audit.py`, and tests for all source hashes, heldout splits and source-case errors. Don't relabel an existing L0 benchmark to disguise a changed experiment.
- For a new Gym observation or reward, version the observation/action or reward schema; adapt the policy interface, exact source-fingerprinted evidence and independent audit **together**. If using a proxy simulator for algorithm debugging, isolate it as a **different non-publishable experiment**, never the source of original-Vitis reward or latency claims.

```bash
python -m rl_environment.gym_curriculum --list-experiments
python -m rl_environment.gym_curriculum --experiment curriculum_l0
```

Both commands are native-free. After a full independent CPU test, use a **fresh stage-specific** native output and explicit vendor-call cap. A smoke run is only a measurement validation check; it cannot substitute for 100 complete training episodes.

## 5. Contributing a new compiler layer or operator

Adding a name to a list is **not** sufficient to support a new layer. Start with [compiler/operator_contribution.py](src/fpgai_research/compiler/operator_contribution.py), then implement and verify these *actual* paths:

1. **Frontend and attributes:** extend `frontend/onnx.py` and normalized attribute/shape handlers. Reject unsupported axes, layouts, rank combinations, dynamic dimensions and unknown attributes with precise errors. Never silently lower unknown operators as identity.
2. **Functional Model IR and numerical semantics:** preserve typed `Tensor`, `Operator`, constants and exact shape/attribute semantics in `model_ir/contracts.py`; add a reference implementation in `validation/reference.py` and semantic-fingerprint tests in `model_ir/semantics.py`.
3. **Architecture resolver:** extend `architecture/resolver.py` and `architecture/scopes.py` with correct iteration coordinates, legal per-operator parallelism/PE/SIMD/memory/tile constraints and fail-closed vendor-independent requests.
4. **Lowering and refinement:** update `model_ir/lowering.py` and its refinement invariants; compile through `fpgai_research.compiler.compile_architecture` so no operator/constant/dependency is dropped.
5. **HLS emission:** implement real logic in `codegen/hls/emitter.py`, not a placeholder. Keep verified `ap_fixed` and stable source hashes; ensure generated C++ implements the same graph as the reference for all supported shapes.
6. **Native source and scientific evidence:** add numerical CSim test vectors in `validation/`, prove type/accuracy/resource/timing against original Vitis HLS, source-hashed fresh XML and testbench, and independently auditable latency. Never infer achieved physical Fmax from a requested clock.
7. **RL exposure:** only after compiler/codegen/CSim and native audit pass, expose the new operator to `rl_environment/cohort.py`, its graph features/action legality, budgets and source/reward receipt checks. Add new stage/heldout tasks as appropriate; old stage contracts remain unchanged.

For new backends, separate vendor-specific tool commands, primitives and target reports under `toolchains/`; do **not** insert vendor-dependent semantics into the canonical Model IR.

The `OperatorContributionChecklist` explicitly reports missing requirements and **does not** mark an operator implemented or native-ready on its own.

## 6. Evidence and experiment integrity

Every publishable hardware result must be reproducible from the exact recorded Git commit, model/source and testbench hashes, legal architecture choice, task/data fingerprints, executed Vitis invocation, real CSim correctness, original CSynth cycles/resources/clock report, and an independently rerunnable **read-only** audit.

- Use never-overwritten `build/` directories with unique names. **Never rewrite original native cases, generated HLS logs, private checkpoints or historical measurements.**
- Every L0–L6 **full** study requires `>=100` complete training episodes, exactly 10 native calls per training episode, plus independent validation, trained versus never-trained GNN and heuristic references.
- Before a 100×10 study, check a separately labeled 22-call `--smoke`, its `gym_audit` replay and `qualification_funnel`. A Vitis process `exit=0` does not by itself establish qualification.
- Report raw training outcomes, feasibility, within-task qualified HLS nanoseconds and honest heldout comparisons; a flat or negative result must stay visible. Don't use synthetic QoR, a falsified reward curve, hidden test leakage or cross-network absolute latency averages.
- Never commit credentials, vendor licenses, original proprietary models, personal data, private checkpoints or native case folders.

## 7. Pull-request checklist

Include a clear scientific question and scope, changed API/contracts, original-vs-new behavior, touched compiler passes, unit/negative/integration test evidence, any actual licensed native-run evidence (or explicitly **not run**), migration plan, Git SHA and documentation update. A test asserting that no licensed tool was launched is preferable for CPU-only features.

See [research design](rl_environment/RESEARCH_QUESTION.md), [staged Gym protocol](rl_environment/GYM_CURRICULUM_L0_L6.md), [engineering rules](AGENTS.md), [architecture map](docs/RESEARCH_EXTENSION_ARCHITECTURE.md), and [native evidence contract](rl_environment/PAPER_EVIDENCE_SPEC.md).
