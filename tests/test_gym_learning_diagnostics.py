"""No-Vitis, CSV-only controls for scientific learning summaries."""
import csv
from pathlib import Path

import pytest

from rl_environment.gym_learning_diagnostics import diagnose


def _csv(path, names, records):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, names)
        writer.writeheader()
        writer.writerows(records)


def test_learning_diagnostics_preserve_100x10_and_do_not_invent_best_latency(
        tmp_path):
    ep = []
    updates = []
    for episode in range(1, 101):
        feasible = int(episode > 50)
        ep.append({
            "method": "online_train",
            "training_episode_index": episode,
            "recorded_native_calls": 10,
            "qualified_calls": feasible,
            "episode_reward_total": -0.5 if feasible == 0 else -0.455,
            "normalized_improvement_over_first_feasible": "",
        })
        for step in range(10):
            label = step == 0 and feasible == 1
            updates.append({
                "method": "online_train",
                "online_update_index": (episode - 1) * 10 + step + 1,
                "feasibility_prediction_before_hls": 0.0,
                "feasibility_brier_score": 0.0 if not label else 1.0,
                "native_feasibility_label": str(label),
            })
    _csv(tmp_path / "episode_learning_curve.csv", list(ep[0]), ep)
    _csv(tmp_path / "policy_update_curve.csv", list(updates[0]), updates)
    result = diagnose(tmp_path)
    assert result["native_calls_invoked"] == 0
    assert result["original_cases_modified"] == 0
    assert result["all_100_episodes"]["native_calls"] == 1000
    assert result["first_50_episodes"]["feasible_calls"] == 0
    assert result["last_50_episodes"]["feasible_calls"] == 50
    assert result["all_100_episodes"]["episodes_with_positive_within_task_latency_improvement"] == 0
    assert result["last_500_updates"]["mean_pre_hls_brier_score"] == pytest.approx(.1)
    assert result["first_500_updates"]["mean_pre_hls_brier_score"] == 0
    assert result["first_500_updates"]["roc_auc_discrimination"] is None
    assert result["last_500_updates"]["roc_auc_discrimination"] == pytest.approx(.5)
    assert result["last_500_updates"]["mean_analytic_class_weight_corrected_brier"] is not None


def test_learning_diagnostics_reject_incomplete_training(tmp_path):
    ep = [{
        "method": "online_train", "training_episode_index": 1,
        "recorded_native_calls": 10, "qualified_calls": 1,
        "episode_reward_total": -.005,
        "normalized_improvement_over_first_feasible": "",
    }]
    _csv(tmp_path / "episode_learning_curve.csv", list(ep[0]), ep)
    update = [{
        "method": "online_train", "online_update_index": 1,
        "feasibility_brier_score": .1, "native_feasibility_label": "False",
    }]
    _csv(tmp_path / "policy_update_curve.csv", list(update[0]), update)
    with pytest.raises(ValueError, match="all 100"):
        diagnose(tmp_path)



def test_analytic_weighted_bce_calibration_and_auc_without_vendor_calls():
    from rl_environment.gym_learning_diagnostics import (
        _correct_weighted_probability, _auc, _prediction_metrics)
    # An ideal w=8 BCE classifier should report q=8p/(1-p+8p).
    true_probability = .10
    q = 8*true_probability / (1-true_probability + 8*true_probability)
    assert _correct_weighted_probability(q) == pytest.approx(true_probability)
    assert _correct_weighted_probability(0) == 0
    assert _correct_weighted_probability(1) == 1
    assert _auc([(.1, False), (.9, True)]) == 1
    assert _auc([(.9, False), (.1, True)]) == 0
    assert _auc([(.5, False), (.5, True)]) == .5
    assert _auc([(.5, True), (.6, True)]) is None
    rows = [
        {"native_feasibility_label": str(label),
         "feasibility_prediction_before_hls": str(raw),
         "feasibility_brier_score": str((raw - int(label))**2)}
        for label, raw in ((False, .4), (True, .6), (False, .2))
    ]
    stats = _prediction_metrics(rows)
    assert stats["prediction_label_pairs"] == 3
    assert stats["feasibility_prevalence"] == pytest.approx(1/3)
    assert stats["roc_auc_discrimination"] == 1
    assert stats["mean_analytic_class_weight_corrected_brier"] is not None
    assert stats["probability_calibration_is_unvalidated"] is True


def test_learning_metrics_reject_label_probability_mismatch():
    from rl_environment.gym_learning_diagnostics import _prediction_metrics
    with pytest.raises(ValueError, match="drift"):
        _prediction_metrics([{
            "native_feasibility_label": "True",
            "feasibility_prediction_before_hls": ".95",
            "feasibility_brier_score": ".5",
        }])
