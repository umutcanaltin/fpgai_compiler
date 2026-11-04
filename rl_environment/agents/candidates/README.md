# Candidate research agents (not yet implemented)

This directory holds **design specifications**, not registered policies. The working algorithm remains `graph_actor_critic` in `../registry.py`. Do not register a policy until it implements the complete `AgentProtocol`, has deterministic CPU tests and an independently verified native-HLS campaign.

## Proposed agents

| ID | Intended method | Reason to compare | Status |
| --- | --- | --- | --- |
| `graph_ppo` | Graph-encoder PPO with masked architecture actor and bounded continuous clock actor | Stable clipped policy optimization | Proposed |
| `graph_a2c` | Graph-encoder synchronous advantage actor-critic | Simple on-policy reference | Proposed |
| `graph_reinforce` | Graph policy gradient with task-level baseline | Minimal learned-policy reference | Proposed |
| `graph_sac` | Hybrid discrete/continuous soft actor-critic, with masked legal actions | Evaluate sample-efficiency under expensive HLS calls | Research/design only; may require an explicitly versioned off-policy protocol |

Create each implementation in `rl_environment/agents/<algorithm>.py` (or a package if it needs several modules), register it explicitly in `../registry.py` and add tests for action legality, clock bounds, no native calls on import/selection, policy gradient/update, checkpoint integrity and fresh initialization. For discrete-only variants, an explicitly versioned clock ablation is required; do not silently omit clock decisions in L5/L6.

Never reuse source-bound results or checkpoints across algorithm revisions. See `../../../paper/rl_experiments/README.md` for the comparative protocol.
