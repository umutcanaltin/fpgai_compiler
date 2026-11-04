# Workstation runbook — native paper campaigns

**State:** Code for source-only planning, explicit L0 smoke, independent replay and strict qualification gating is committed. The maintainer reported 757 passing CPU tests on the preceding source revision. **No new licensed Vitis HLS campaign has been launched by ChatGPT.** All native commands below run only on the workstation where the user types them.

## Scientific boundaries

- The initial executable method is `graph_actor_critic`, the only source-registered policy at the start of this sprint. PPO/A2C/REINFORCE/SAC are not yet native-audited and must not be included in paper data until implementations and algorithm-aware audits are tested.
- Native study directories are create-only; never relabel, modify or merge historical results.
- L0–L6 each require at least 100 **complete** online training tasks with exactly 10 original Vitis evaluations per training episode for a full study; independent heldout controls require additional original calls.
- Rewards come only from strict original source-attested numerical, resource, CSim and Vitis CSynth timing qualifications. These are HLS estimates, **not** measured chip latency or post-route Fmax.
- Smoke is **not** learning evidence. The 22-call L0 smoke engineering gate is exactly 22 cases, independent source replay passed, at least 14 strictly qualified, and zero timing parse exceptions. Any failure stops progression automatically.
- The paper launcher requires smoke and full run to use the **same stage, agent, seed and Git commit**; do not change source between smoke and full. Fresh output roots always.

## 1. Refresh code and preflight

```bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
export PYTHONPATH="$PWD/src:$PWD"
python -m pip install --no-deps -e .
python -m rl_environment.runtime_contract
python -m compileall -q src rl_environment tests
python -m pytest -q
git status --short
command -v vitis_hls
```

Do not proceed if source binding, tests, clean-worktree verification, or Vitis executable availability fail. The launcher repeats the clean Git/source checks.

## 2. Source-only first campaign plan (zero vendor calls)

```bash
python -m rl_environment.paper_campaign --stage L0 --mode smoke --seed 111
python -m rl_environment.paper_campaign --stage L0 --mode full --seed 111
```

Neither command creates a native study. Do not confuse `paper_comparison.py`'s 21 planned stage/seed experiments with actual runs.

## 3. First original-Vitis L0 smoke — intentional foreground execution

The following command **actually invokes up to 22 licensed Vitis HLS evaluations**. Only run after verifying that your PC is ready, and do not choose a previously created output directory.

```bash
python -m rl_environment.paper_campaign \
  --stage L0 --agent graph_actor_critic --mode smoke --seed 111 \
  --execute-native --max-total-vitis-calls 22 \
  --output build/paper_rl/l0_graph_actor_critic_seed111_smoke_v1 \
  --vitis-executable vitis_hls
```

On completion, the launcher independently replays and audits the original cases, runs the read-only qualification funnel, and prints `smoke_gate.status`. **Do not run full training unless it is `SMOKE_ACCEPTED_FOR_FULL_RESEARCH`.**

If the process fails or is interrupted, keep the original directory untouched and use a new version suffix for a subsequent attempt. Preserve terminal logs and native case directories.

## 4. Full L0 — separate explicit decision and 1,090-call maximum

Only after passing the **same-commit** original smoke:

```bash
python -m rl_environment.paper_campaign \
  --stage L0 --agent graph_actor_critic --mode full --seed 111 \
  --execute-native --max-total-vitis-calls 1090 \
  --smoke-study build/paper_rl/l0_graph_actor_critic_seed111_smoke_v1 \
  --output build/paper_rl/l0_graph_actor_critic_seed111_full_v1 \
  --vitis-executable vitis_hls
```

The launcher refuses an incomplete, mismatched or poorly qualified smoke and automatically performs read-only independent audit/funnel on the completed full campaign. A completed run still is not a proof of DRL learning or a physically validated FPGA clock.

## 5. Evidence exports, only after complete audit

```bash
python -m rl_environment.paper_evidence \
  --study build/paper_rl/l0_graph_actor_critic_seed111_full_v1 \
  --output build/paper_rl/l0_graph_actor_critic_seed111_paper_v1

python -m rl_environment.gym_curriculum_report \
  --study L0=build/paper_rl/l0_graph_actor_critic_seed111_full_v1 \
  --output build/paper_rl/l0_graph_actor_critic_seed111_scorecard_v1
```

Report qualified and unqualified outcomes, no-feasible-test cases, separate matched heldout results, costs and honest uncertainty; do not combine incompatible raw latency across network shapes. Do not run more stages or seeds automatically: each needs its own original 22-call audited smoke gate, 100×10+ independently audited campaign, and sufficient disk/runtime budget.

## 6. Follow-up algorithm development

The prospective `agents/candidates/` specifications are **not** executable DRL agents. Each new policy must implement and test the complete agent protocol, checkpointing, source code registry, scientific diagnostics, and **algorithm-aware independent audit** before launching its own source-immutable smoke. The existing audit includes checks specific to the graph actor-critic diagnostic loss and one update per original call; register no algorithm without adjusting/independently testing those invariants. Do not bypass the auditor by pretending a different algorithm uses the old update rule.

The paper matrix and publication method requirements live in [README](README.md) and [EXPERIMENT_MATRIX](EXPERIMENT_MATRIX.md). Use fresh source-bound output paths; no experimental result belongs in Git unless redacted and approved for release.
