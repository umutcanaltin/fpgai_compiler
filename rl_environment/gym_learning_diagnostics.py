"""Read-only, source-grounded 100-episode Gym training diagnostics.

Uses the real original-HLS journal-derived *provisional* paper CSV export.
Never calls Vitis, alters a study, fills missing labels or compares raw
latencies across unrelated neural networks.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import isfinite, log
from pathlib import Path


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    if value in (None, "", "None", "null"):
        return None
    result = float(value)
    return result if isfinite(result) else None


def _summary(rows):
    if not rows:
        return None
    n = len(rows)
    calls = sum(int(r["recorded_native_calls"]) for r in rows)
    feasible = sum(int(r["qualified_calls"]) for r in rows)
    rewards = [_number(r.get("episode_reward_total")) for r in rows]
    improvements = [
        _number(r.get("normalized_improvement_over_first_feasible"))
        for r in rows]
    return {
        "episodes": n, "native_calls": calls,
        "feasible_calls": feasible,
        "feasibility_fraction": feasible / calls if calls else None,
        "episodes_with_a_feasible_candidate": sum(
            int(r["qualified_calls"]) > 0 for r in rows),
        "mean_episode_return": (
            sum(x for x in rewards if x is not None) /
            sum(x is not None for x in rewards)
            if any(x is not None for x in rewards) else None),
        "episodes_with_positive_within_task_latency_improvement": sum(
            x is not None and x > 0 for x in improvements),
    }


def _correct_weighted_probability(raw_probability, *, positive_weight=8.0):
    """Recover a candidate class prevalence from an ideal weighted BCE sigmoid.

    If the true conditional probability is p, the minimizer of class-weighted
    binary cross-entropy is q = w*p/(1-p+w*p). Thus p=q/(w-(w-1)*q).
    This is an *analytical correction*, not a fitted calibration model; real
    classifier misspecification may still require separate held-out calibration.
    """
    if not 0. <= raw_probability <= 1.:
        raise ValueError("Original HLS probability outside [0,1]")
    return raw_probability / (
        positive_weight - (positive_weight - 1.) * raw_probability)


def _auc(pairs):
    """ROC AUC from paired probabilities/labels; None without both classes."""
    yes = [p for p, label in pairs if label]
    no = [p for p, label in pairs if not label]
    if not yes or not no:
        return None
    wins = sum((p > n) + .5 * (p == n) for p in yes for n in no)
    return wins / (len(yes) * len(no))


def _prediction_metrics(subset):
    observed = []
    for row in subset:
        label = row.get("native_feasibility_label")
        raw = _number(row.get("feasibility_prediction_before_hls"))
        recorded = _number(row.get("feasibility_brier_score"))
        if label not in ("True", "False", "true", "false"):
            continue
        if raw is None or recorded is None:
            continue
        y = label.lower() == "true"
        if not 0. <= raw <= 1. or abs(recorded - (raw - float(y)) ** 2) > 1e-6:
            raise ValueError("Source-reported pre-HLS probability/Brier drift")
        observed.append((raw, y))
    def mean(values):
        return sum(values) / len(values) if values else None
    prevalence = mean([float(y) for _, y in observed])
    calibrated = [(_correct_weighted_probability(raw), y)
                  for raw, y in observed]
    def logloss(items):
        return mean([-log(max(1e-12, min(1. - 1e-12, p))) if y
                     else -log(max(1e-12, 1. - p))
                     for p, y in items])
    return {
        "prediction_label_pairs": len(observed),
        "positive_labels": sum(y for _, y in observed),
        "feasibility_prevalence": prevalence,
        "mean_raw_weighted_sigmoid": mean([p for p, _ in observed]),
        "mean_analytic_class_weight_corrected_probability": mean(
            [p for p, _ in calibrated]),
        "mean_pre_hls_brier_score": mean([
            (raw - float(y)) ** 2 for raw, y in observed]),
        "mean_analytic_class_weight_corrected_brier": mean([
            (p - float(y)) ** 2 for p, y in calibrated]),
        "raw_weighted_sigmoid_log_loss": logloss(observed),
        "analytic_class_weight_corrected_log_loss": logloss(calibrated),
        "roc_auc_discrimination": _auc(observed),
        "always_predict_infeasible_brier_reference": prevalence,
        "empirical_constant_prevalence_brier_reference_in_sample": (
            prevalence * (1. - prevalence)
            if prevalence is not None else None),
        "probability_calibration_is_unvalidated": True,
        "positive_weight_of_recorded_classifier": 8.0,
    }


def diagnose(export):
    root = Path(export).resolve(strict=True)
    episodes = _rows(root / "episode_learning_curve.csv")
    policy = _rows(root / "policy_update_curve.csv")
    training = sorted(
        (r for r in episodes if r["method"] == "online_train"),
        key=lambda r: int(r["training_episode_index"]))
    if len(training) != 100 or [
            int(r["training_episode_index"]) for r in training] != list(range(1, 101)):
        raise ValueError("Requires all 100 authentic training-episode summaries")
    updates = [r for r in policy if r["method"] == "online_train"]
    if len(updates) != 1000:
        raise ValueError("Requires all 1000 original online updates")
    updates.sort(key=lambda r: int(r["online_update_index"]))
    if [int(r["online_update_index"]) for r in updates] != list(range(1, 1001)):
        raise ValueError("Nonconsecutive original GNN learning updates")
    result = {
        "status": "DESCRIPTIVE_SOURCE_EXPORTED_TRAINING_DIAGNOSTICS",
        "source_export": str(root),
        "native_calls_invoked": 0,
        "original_cases_modified": 0,
        "first_25_episodes": _summary(training[:25]),
        "last_25_episodes": _summary(training[-25:]),
        "first_50_episodes": _summary(training[:50]),
        "last_50_episodes": _summary(training[-50:]),
        "all_100_episodes": _summary(training),
        "policy_updates": len(updates),
        "caution": ("Different network tasks across episodes; changes in aggregate "
                    "reward and feasibility alone do not prove policy learning. "
                    "One seed, no causal heldout trained-versus-untrained result yet."),
    }
    for name, subset in (("first_250_updates", updates[:250]),
                         ("last_250_updates", updates[-250:]),
                         ("first_500_updates", updates[:500]),
                         ("last_500_updates", updates[-500:])):
        result[name] = {
            "native_updates": len(subset),
            **_prediction_metrics(subset),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--output", help="Optional create-only diagnostic JSON and honest rolling plots")
    args = parser.parse_args(argv)
    result = diagnose(args.export)
    if args.output:
        output = Path(args.output)
        if output.exists() or output.is_symlink():
            raise FileExistsError("Diagnostic output must be create-only")
        output.mkdir(parents=True)
        (output / "learning_diagnostics.json").write_text(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            result["plots_created"] = []
        else:
            src = Path(args.export)
            episodes = sorted(
                (r for r in _rows(src / "episode_learning_curve.csv")
                 if r["method"] == "online_train"),
                key=lambda r: int(r["training_episode_index"]))
            updates = sorted(
                (r for r in _rows(src / "policy_update_curve.csv")
                 if r["method"] == "online_train"),
                key=lambda r: int(r["online_update_index"]))
            figures = []
            def plot_series(y, ylabel, filename, title):
                fig, ax = plt.subplots(figsize=(9, 4))
                x = list(range(1, len(y) + 1))
                ax.plot(x, y, alpha=.3, linewidth=.7,
                        label="Recorded per-episode observation")
                window = 20
                ax.plot(list(range(window, len(y)+1)),
                        [sum(y[i-window:i])/window
                         for i in range(window, len(y)+1)],
                        linewidth=2., label="Trailing 20-episode mean")
                ax.set(xlabel="Training episode (different network each time)",
                       ylabel=ylabel, title=title)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output / filename, dpi=180)
                plt.close(fig)
                figures.append(filename)
            plot_series([float(_number(r["episode_reward_total"])) for r in episodes],
                        "Original HLS reward (unchanged)",
                        "rolling_original_reward.png",
                        "Observed reward; smoothing is descriptive, not proof of learning")
            def brier_rows(subset):
                return _prediction_metrics(subset)
            series = []
            for idx in range(100):
                group = updates[idx*10:(idx+1)*10]
                metrics = brier_rows(group)
                series.append(metrics["mean_analytic_class_weight_corrected_brier"])
            if all(x is not None for x in series):
                plot_series(series, "Class-weight-corrected Brier (lower is better)",
                            "rolling_corrected_brier.png",
                            "Analytical 8:1 BCE correction; not heldout calibration")
            result["plots_created"] = figures
    print(json.dumps(result, sort_keys=True,
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
