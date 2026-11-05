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


def test_hw05_comparison_export_requires_independent_original_case_audit(tmp_path):
    # Shape-only fixture dictionaries must never masquerade as paper E3.
    baseline = _run("random", latency=[1000., None],
                    qualified_index=[1, None])
    candidate = _run("reinforce", latency=[900., None],
                     qualified_index=[1, None])
    rows, unverified = compare.paired_summary(baseline, candidate)
    root = tmp_path / "paper_pairs"
    with pytest.raises(ValueError, match="independently audited"):
        compare.export_comparison(root, rows, unverified)
    assert not root.exists()
    # Simulates the status granted by compare_audited_campaigns AFTER full
    # actual native source replay, but never claims this fixture is real.
    stamp = dict(unverified,
                 status="INDEPENDENT_NATIVE_CASES_AUDITED_PAIR_DESCRIPTIVE")
    compare.export_comparison(root, rows, stamp)
    assert (root / "comparison.json").is_file()
    assert (root / "paired_native_hls.csv").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        compare.export_comparison(root, rows, stamp)


def test_hw06_cross_clock_ablation_requires_explicit_opt_in():
    """Source-only shape fixtures do NOT represent original native cases."""
    baseline = _run("ppo", latency=[1000., None], qualified_index=[1, None])
    candidate = _run("ppo", latency=[800., None], qualified_index=[1, None])
    baseline["metadata"].update(clock_regime="fixed", fixed_clock_ns=14.)
    candidate["metadata"].update(clock_regime="search", fixed_clock_ns=None)
    with pytest.raises(ValueError, match="allow-clock-regime-ablation"):
        compare.paired_summary(baseline, candidate)
    rows, report = compare.paired_summary(
        baseline, candidate, allow_clock_regime_ablation=True,
    )
    assert report["cross_clock_action_space_ablation"] is True
    assert report["cross_clock_ablation_explicitly_authorized"] is True
    assert report["baseline"] == "ppo_fixed_clock"
    assert report["candidate"] == "ppo_search_clock"
    assert rows[0]["candidate_over_baseline_latency_ratio"] == .8
    assert report["paired_significance_claim"] is False
    mismatch = copy.deepcopy(candidate)
    mismatch["metadata"]["seed"] = 222
    with pytest.raises(ValueError, match="Paired comparison requires"):
        compare.paired_summary(
            baseline, mismatch, allow_clock_regime_ablation=True,
        )


def test_hw06_general_network_training_provenance_distinguishes_same_method():
    one = _run("ppo", latency=[1000., None], qualified_index=[1, None])
    many = copy.deepcopy(one)
    one["metadata"]["checkpoint_training_network_regime"] = "same"
    one["metadata"]["checkpoint_training_clock_regime"] = "fixed"
    many["metadata"]["checkpoint_training_network_regime"] = "general"
    many["metadata"]["checkpoint_training_clock_regime"] = "fixed"
    rows, report = compare.paired_summary(one, many)
    assert report["baseline_training_network_regime"] == "same"
    assert report["candidate_training_network_regime"] == "general"
    assert "trained_same" in report["baseline"]
    assert "trained_general" in report["candidate"]
    assert report["paired_significance_claim"] is False


def test_hw09_paired_reward_mode_ablation_is_explicit_and_hls_metric_only():
    a = _run("ppo", latency=[9500., None], qualified_index=[1, None])
    b = _run("ppo", latency=[8400., None], qualified_index=[1, None])
    a["metadata"].update(reward_mode="best_improvement", reward_reference_ns=10000.)
    b["metadata"].update(reward_mode="dense_latency", reward_reference_ns=10000.)
    with pytest.raises(ValueError, match="allow-reward-mode-ablation"):
        compare.paired_summary(a, b)
    rows, report = compare.paired_summary(
        a, b, allow_reward_mode_ablation=True,
    )
    assert report["reward_profile_ablation"] is True
    assert report["reward_mode_ablation_explicitly_authorized"] is True
    assert report["baseline"] == "ppo_score_best_improvement"
    assert report["candidate"] == "ppo_score_dense_latency"
    assert report["paired_significance_claim"] is False
    assert rows[0]["candidate_over_baseline_latency_ratio"] == 8400. / 9500.
    assert rows[1]["candidate_over_baseline_latency_ratio"] is None
