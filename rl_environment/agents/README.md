# DRL agent implementations and research candidates

The active, source-controlled algorithm registry is [registry.py](registry.py). It currently exposes only the algorithms that actually work. The existing GNN actor-critic implementation remains in [../gym_agent.py](../gym_agent.py) for historical import compatibility.

Prospective algorithms are specified in [candidates/README.md](candidates/README.md); **not** registered as executable classes until fully implemented and tested. New contributors should follow [candidates/CONTRIBUTING.md](candidates/CONTRIBUTING.md) and the repository-level [CONTRIBUTING.md](../../CONTRIBUTING.md).

The publication experiment plan and comparisons are in [paper/rl_experiments](../../paper/rl_experiments/README.md). Keep the agent code, experimental protocol, and original Vitis HLS result artifacts separate. No licensed vendor tool may execute during imports, plans, tests, or agent construction.

Run `python -m rl_environment.gym_curriculum --list-experiments` and the full source-only CPU suite before any explicitly authorized original-HLS campaign.
