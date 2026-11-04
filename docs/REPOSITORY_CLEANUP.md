# Gym research branch cleanup ledger — 31 October 2026

**Branch:** cleanup originally prepared on `feat/gym-graph-rl-research-20261031` and subsequently merged into **main** via PR #4. Historical commits remain recoverable. The full original Git history, old branches and private native evidence are unchanged.

## Paths removed from this branch

| Category | Removed tracked files |
|---|---:|
| Legacy top-level `experiments/` runners, scripts and studies | 116 |
| Dated historical experiment specifications under `docs/` | 60 |
| Old `paper/` and `results/` artifacts | 24 |
| Old `examples/` unrelated to the current compiler | 5 |
| Old reporting/reproduction `scripts/` | 5 |
| Superseded v4–v9 RL source implementations | 12 |
| Historical experiment-only `tests/` modules | 186 |
| Legacy GitHub workflow | 1 |
| Superseded online Post-HLS design v1 | 1 |
| **Total** | **410** |

The initial cleanup retained **46 core test modules**. The user's new Gym test run passed 12 tests, but full collection exposed that `tests/test_cpp_phi_trace.py` depends on the deleted `test_hls_loop_phi_oracle.py`. The exact original Phi-oracle module was restored and `test_test_import_integrity.py` added to guard test-to-test imports. With the new Gym tests there are now **49 active test modules**; net removals from the original cleanup are **409**. Full pytest still needs a post-restoration rerun.

## Post-cleanup dependency correction

- Restored `tests/test_hls_loop_phi_oracle.py` unchanged from the earlier Git revision.
- Added `tests/test_test_import_integrity.py` to fail collection when retained tests import deleted local test modules.
- The table above records the *original* 410 deletions; net removal is 409.

## Deliberately retained

- `src/fpgai_research/`: core Functional/Architectural/Lowered Model IR, HLS emitter, Vitis runner, CSim/testbench, report parser and transitive compatibility dependencies.
- Some internal `src/fpgai_research/experiments/` helpers that remain useful as real compiler/HLS library code. The old RL v4–v9 modules were removed; further internal library removal needs a full dependency graph + regression test.
- `rl_environment/`: our only **active RL research** (existing fixed-clock pilot + proposed Gym/GNN/clock-aware replacement).
- `docs/OPEN_SOURCE_RELEASE_CHECKLIST.md`, `docs/LICENSE_DECISION.md`, release/security notices and basic compiler examples.

## Scientific preservation and release constraints

The prior 16-case actual Vitis study and its independent audit are preserved in previously source-pinned commits and user-owned private directories. Historical studies are **not currently active scientific objectives**. Git does not physically expunge earlier committed code or secrets merely because paths were deleted on a new branch. Do not rewrite Git history or make the repository public without explicit rights-holder approval and complete historical scanning.

## Current research and tests

Read [RESEARCH_QUESTION.md](../rl_environment/RESEARCH_QUESTION.md) for the one and only active optimization experiment: **one shared GNN/DRL agent trained online from scratch in Gym**, full per-layer architecture + requested HLS clock actions, numerical and capacity constraints, single lowest feasible HLS-estimated network latency in nanoseconds, no Vivado.

Run active/core verification after checking out this branch:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q
```

The previous run of **2,156 passing tests belongs to an older different source revision** and does not qualify the new cleaned branch. All Vitis native reruns require explicit opt-in and new, immutable, source-pinned outputs.
