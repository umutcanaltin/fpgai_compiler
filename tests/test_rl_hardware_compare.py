"""Audited heldout pair comparison; synthetic dictionaries never count as HLS."""
from __future__ import annotations

import copy
import json

import pytest

from rl_environment import hardware_compare as compare


def _run(method, *, latency, qualified_index, seed=111, split="test",
         eval_mode=True, no_mp=False):
    metadata = {
        "method": method, "split": split, "seed": seed,
        "evaluation": eval_mode,
        "message_passing": not no_mp, "trial_history": True,
        "schedule": ["heldout_a", "heldout_b"],
        "episodes": 2, "original_vitis_calls_per_episode": 2,
        "task_definitions": [{"id": "heldout_a"}, {"id": "heldout_b"}],
        "source_revision_sha": "a"*40,
    }
    episodes = []
    trials = {}
    for i, (lat, first) in enumerate(zip(latency, qualified_index)):
        episodes.append({
            "episode": i, "task_id": metadata["schedule"][i],
            "best_qualified_latency_ns": lat,
        })
        trials[i] = [{
            "qualified": first is not None and j >= first,
        } for j in range(1, 3)]
    return {
        "metadata": metadata, "episodes": episodes,
        "trials_by_episode": trials,
    }


def test_hw05_pairwise_censoring_keeps_no_feasible_tasks():
    baseline = _run(
        "random", latency=[1000., None], qualified_index=[2, None]
    )
    learned = _run(
        "ppo", latency=[800., 1600.], qualified_index=[1, 2]
    )
    rows, summary = compare.paired_summary(baseline, learned)
    assert summary["task_episode_pairs"] == 2
    assert summary["categories"]["both_feasible"] == 1
    assert summary["categories"]["candidate_only_feasible"] == 1
    assert summary["baseline_feasible_rate"] == .5
    assert summary["candidate_feasible_rate"] == 1.
    assert summary["conditional_both_feasible_median_ratio"] == .8
    assert summary["conditional_both_feasible_pair_count"] == 1
    assert rows[1]["candidate_over_baseline_latency_ratio"] is None
    assert rows[0]["candidate_first_qualified_call"] == 1
    assert rows[0]["baseline_first_qualified_call"] == 2
    assert summary["paired_significance_claim"] is False


def test_hw05_pairwise_rejects_training_leakage_wrong_budget_and_seed():
    a = _run("random", latency=[None, None], qualified_index=[None, None])
    b = _run("actor_critic", latency=[None, None],
             qualified_index=[None, None])
    for changed in (
        {"evaluation": False},
        {"seed": 222},
        {"original_vitis_calls_per_episode": 3},
        {"source_revision_sha": "b"*40},
        {"schedule": ["heldout_b", "heldout_a"]},
    ):
        altered = copy.deepcopy(b)
        altered["metadata"].update(changed)
        with pytest.raises(ValueError, match="Paired comparison requires"):
            compare.paired_summary(a, altered)


def test_hw05_pairwise_requires_distinct_controls_and_no_proxy_latency():
    a = _run("ppo", latency=[None, None], qualified_index=[None, None])
    with pytest.raises(ValueError, match="labels must differ"):
        compare.paired_summary(a, copy.deepcopy(a))
    b = _run("random", latency=[None, None], qualified_index=[None, None])
    bad = _run("actor_critic", latency=[-1., None],
               qualified_index=[1, None])
    with pytest.raises(ValueError, match="Invalid qualified"):
        compare.paired_summary(b, bad)


def test_hw05_comparison_export_is_create_only(tmp_path):
    baseline = _run("random", latency=[1000., None],
                    qualified_index=[1, None])
    candidate = _run("reinforce", latency=[900., None],
                     qualified_index=[1, None])
    rows, summary = compare.paired_summary(baseline, candidate)
    root = tmp_path / "paper_pairs"
    compare.export_comparison(root, rows, summary)
    assert (root / "comparison.json").is_file()
    assert (root / "paired_native_hls.csv").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        compare.export_comparison(root, rows, summary)
