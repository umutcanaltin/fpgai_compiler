"""CPU-only scientific comparison planner contract."""
import pytest

from rl_environment.paper_comparison import build_comparison_plan


def test_multistage_comparison_is_source_only_and_source_deterministic():
    p = build_comparison_plan(stages=("L0", "L6"), seeds=(111, 222))
    assert p == build_comparison_plan(stages=("L0", "L6"), seeds=(111, 222))
    assert p["vendor_calls_executed"] == 0
    assert p["training_native_calls_required_total"] == 4000
    assert len(p["studies"]) == 4
    assert all(s["results_status"] == "NOT_EXECUTED" for s in p["studies"])
    assert p["studies"][0]["fixed_clock_ns"] == 32.0
    assert p["studies"][2]["fixed_clock_ns"] is None


@pytest.mark.parametrize("kwargs", [
    {"agents": ("graph_ppo",)},
    {"stages": ("L7",)},
    {"stages": ("L0", "L0")},
    {"agents": ("graph_actor_critic", "graph_actor_critic")},
    {"seeds": (111, 111)},
    {"seeds": (-1,)},
    {"training": 99},
    {"calls_per_episode": 9},
    {"stages": ()},
])
def test_invalid_protocols_fail_closed(kwargs):
    with pytest.raises(ValueError):
        build_comparison_plan(**kwargs)
