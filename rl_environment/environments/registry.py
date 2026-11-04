"""Environment factory boundary: keep original vendor HLS execution opt-in."""
from __future__ import annotations


def available_environments() -> tuple[str, ...]:
    return ("graph_native_hls",)


def create_environment(name: str, task, root, **options):
    if name != "graph_native_hls":
        raise ValueError(f"Unsupported environment '{name}'. "
                         "New environments must have a source-pinned Gym/HLS audit adapter.")
    from ..gym_env import GymGraphEnv
    return GymGraphEnv(task, root, **options)
