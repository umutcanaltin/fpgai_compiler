> **On hold (2026-11-01):** This is a preserved prospective RL matrix, not an authorization to run new RL experiments. Focus on [compiler-IR experiment plan](../compiler_ir/EXPERIMENT_PLAN.md). The already-started source-pinned L0 full study may complete unchanged and receive read-only audit; no new agent/stage/seed campaign is started during the freeze.

# Planned experiment matrix (not completed experiments)

| Experiment ID | Environment | Agents / controls | Main question | Evidence gate |
| --- | --- | --- | --- | --- |
| E0 | L0 | graph_actor_critic, random, greedy, evolutionary, parzen, untrained graph actor-critic | Can the easiest real-HLS problem produce qualified training/evaluation evidence? | Independent native audit |
| E1 | L1–L4 | same plus **implemented** candidate agents | Does performance change with model depth, action count and numeric options? | Paired heldout native HLS |
| E2 | L5 | implemented graph-policy agents and controls | Joint continuous clock and architecture optimization | Same task/clock legality, native HLS |
| E3 | L6 | independently from-scratch agents | Generalization to unseen deeper model graphs | Disjoint task fingerprints, native HLS |
| E4 | Versioned L0/L3/L5 ablations | graph actor-critic variants | Message passing, history, classifier and clock-search effects | Same budgets and explicit changed protocol |
| E5 | Per-stage algorithm comparison | graph_actor_critic vs graph_ppo/graph_a2c/graph_reinforce when implemented | Sample efficiency and final qualified estimated latency | Matched seeds/tasks, full source audit |

Candidate names do not mean executable implementations. Each new algorithm should receive a separate `agents/<algorithm>.py` implementation, tests, source registry entry and method section. Preserve unqualified-case counts and predeclare how incomplete or interrupted runs are treated.

Suggested preregistered seeds for a **future** multiseed study: 111, 222, 333, 444, 555. These are planning values, not previously executed evidence, and imply substantial native compile cost. Do not run them automatically.
