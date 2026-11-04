> **Research priority update (2026-11-01):** The [compiler-first architecture-semantic IR paper](../paper/compiler_ir/README.md), its [sprints](../paper/compiler_ir/SPRINTS.md) and [experiment protocol](../paper/compiler_ir/EXPERIMENT_PLAN.md) take priority. RL research is frozen for new development and new experiments; existing source-pinned L0 native evidence may finish and undergo read-only audit. The compiler IR may be *suitable as a typed observation/legal-action interface to future RL agents*; no improvement in RL learning has been demonstrated by that interface. This note changes no existing toolchain or research code.

# Research extension architecture — FPGAI compiler and native-HLS Gym

## Goals and non-goals

This repository has **two complementary but separable research surfaces**:
1. `src/fpgai_research/`: compiler semantics, operator import, architecture legality, lowering, HLS emission and vendor-tool integration.
2. `rl_environment/`: reproducible learning environments consuming compiler-legal candidate architectures and **original Vitis CSim/CSynth** qualification receipts.

Neither side should introduce vendor-specific hardware knowledge in policy code, fabricate QoR, silently change numeric semantics, or run a licensed tool during import, setup, unit tests, dry plans, or reports. Only explicit `--execute-native` with an authorized call budget may run Vitis. Existing source-pinned case folders, checkpoints, and old import paths are immutable historical evidence.

## Architecture map

```text
src/fpgai_research/
  frontend/               ONNX import and attribute parsing
  model_ir/               typed operators, tensor shapes, numerical semantics
  architecture/           per-operator legal hardware requests; global precision constraints
  compiler/               validated architecture -> IR lowering and code generation
  codegen/hls/            HLS C++ emission (not the Vitis process)
  validation/             native-reference sample generation and checks
  toolchains/vitis_hls/   single real Vitis packaging/Tcl/runner adapter
  experiments/           compiler-derived knob catalog used by native Gym

rl_environment/
  agents/                agent Protocol, explicit algorithm registry
  environments/          environment registry and native Gym boundary
  experiments/           explicit named study protocols
  infrastructure/        vendor provenance, storage and source boundary
  evaluation/            independent audit and analysis API boundaries
  curriculum.py          L0..L6 immutable difficulty contracts
  cohort.py, tasks.py    deterministic task distribution and task SHA
  post_hls_env.py        compiler-legal candidate loop + native case receipt
  gym_env.py             Gymnasium observations, masks, history, step semantics
  gym_agent.py           existing shared GNN actor-critic implementation
  gym_study.py           100x10 actual study runner + result journals
  gym_audit.py           independent replay of native source/numeric/HLS evidence
  gym_curriculum_report.py  paired real HLS trained/untrained stage scorecards
  runtime_contract.py    fail-fast source-checkout binding validation
```

The `agents/`, `environments/`, `experiments/` packages define extension interfaces and controlled registries; **older modules are intentionally left at their original paths**. Moving previously source-pinned functions without compatibility and paper-method audits would invalidate reproducibility. This is an incremental migration rather than a large rename.

## New DRL algorithm

Implement a real, tested class in `rl_environment/agents/<name>.py`, conforming to `agents/api.py` (choose, update, save_private, load_private). Register an explicit factory in `agents/registry.py`; the CLI flag `--agent-algorithm` must resolve to actual code, never a placeholder. The algorithm receives **only the provided observation** (model graph, constraints, action mask, previous real HLS history); it must never directly load unpublished native evaluations.

The experimental comparison must control the task cohort, action budget and original HLS source-revision; checkpoint and optimizer restoration must preserve identical algorithm semantics. Add tests for legal masked actions, no vendor calls at init/choose, update using recorded native labels, and checkpoint reloading. New algorithms are **freshly initialized per independent seed** in current L0..L6 protocols; any cross-stage pretraining is a separate explicitly preregistered study.

## New native Gym environment

Use `environments/registry.py` to select a source-pinned environment. An environment must emit a bounded Gymnasium `observation_space` and `action_space`, expose only legal compiler-derived actions, and preserve the call budget, immutable native case paths, exact original HLS source/clock/tool manifest, and strict CSim + numeric error + resource + timing evidence. Synthetic rewards are **not valid** for an original-HLS scientific result. New device/layer families require compiler legality before they are made available as action candidates.

Keep controller logic, observations and compiler contracts separate from tool execution. Implement an independent audit adapter before accepting results for publication.

## New experiment or task family

Implement a deterministic config factory in `experiments/registry.py` or a source-defined stage/task generator. Explicitly record the experiment ID, model/task SHA, random seed, action mask/knob restrictions, clock precision, reward definition, holdout splits, minimum training episodes, algorithm ID, compiler Git SHA and vendor invocation metadata. `gym_curriculum --list-experiments` must not invoke Vitis.

For L0..L6, at least **100 complete original-Vitis training episodes, exactly 10 genuine original HLS calls per episode**, plus independent trained/untrained/control heldout runs are compulsory. The short `--smoke` experiment never counts as completed training. Report negative and inconclusive results without modifying previous source evidence or combining different task reward scales.

## New compiler model/operator/layer

1. Start with the typed operator and shape/broadcast/numeric semantics in `model_ir/` and frontend import in `frontend/`. Every operator must carry correctly shaped tensors and preserve existing graph/temporal contracts rather than be silently lowered to MatMul.
2. Add reference evaluation and numerical tests in `validation/`; qualify precision/rounding/overflow explicitly.
3. Add compiler architecture legality in `architecture/` and coherent lowering/code generation in `compiler/` and `codegen/hls/`. Hardware characteristics belong in hardware/toolchain libraries, never in the GNN or an ONNX frontend shim.
4. Create executable tiny tests spanning import → typed Model IR → reference values → architecture resolution → generated C++/Tcl → original CSim/CSynth evidence. Never claim support based on an operator name in a parser alone.
5. Only after full source/numeric/architecture/HLS implementation should an environment add that operator family to the stage distribution. Preserve original frozen validation/test tasks and their hashes.

The **Morfics five-layer product goal** (System IR, Model IR, Math IR, Circuit Architecture, Physical Implementation) is broader than this research Vitis backend. Contributors must not collapse these layers into a vendor-dependent monolith, force one architecture for all operators, remove system/temporal/multiclock requirements, or require DRL search to compile a design.

## Native environment and stale-install protection

A confirmed host failure had `python -m` import `fpgai_research` from a sibling October research checkout although `pytest` passed because it added `src/` to `sys.path`. `rl_environment/__init__.py` now explicitly selects the sibling checkout `src/`; `runtime_contract.verify_local_compiler_package()` fail-closes on any stale native compiler before original study creation. Run:

```bash
python -m pip install --no-deps -e .
python -m rl_environment.runtime_contract
python -m compileall -q src rl_environment tests
python -m pytest -q
python -m rl_environment.gym_curriculum --list-stages
python -m rl_environment.gym_curriculum --list-experiments
```

If module origins still disagree, inspect `python -c "import sys, fpgai_research; print(sys.path); print(fpgai_research.__file__)"`. **Do not edit existing case files or retry against an existing `build/gym_curriculum_L0_smoke_seed111_v2` directory.**

## Contribution checklist

- Provide a small design note explaining semantics, legal action space, expected evidence and backward compatibility.
- Add pure CPU unit tests and an end-to-end **no-Vitis** integration test, including source path and record hashes.
- Register algorithm/environment/experiment via the matching source-controlled registry; use a new experiment ID/manifest schema for new scientific semantics.
- Preserve old imports, task fingerprinting in opt-out mode, original HLS qualification rules, and any existing source-pinned native data.
- Update `CONTRIBUTING.md`, method notes, experiment registry and research handoff as appropriate.
- Before a native study: all CPU tests pass, the compiler-source preflight passes, the 22-call fresh smoke qualifies and the independent auditor passes.
- State the results as **original Vitis HLS estimates**, never post-route physical timing or guaranteed DRL learning.
