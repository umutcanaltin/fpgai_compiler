"""Audited HLS-only experiment tables, paired effects and optional plots.

Generate separately from immutable Vitis cases; no vendor program is called.
Only independently replay-audited observations may be published as HLS
*estimates*, never measured hardware timings.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .gym_audit import audit


def _ci(values, *, seed=20261031, samples=2000):
    if len(values) < 2:
        return None
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(samples, len(data)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def export(study, output):
    passed = audit(study)  # read-only source, timing, reward and provenance audit
    source = Path(study).resolve(strict=True)
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Create-only report folder required")
    summary = json.loads((source / "summary.json").read_text())
    observations = [json.loads(line) for line in
                    (source / "native_steps.jsonl").read_text().splitlines()]
    target.mkdir(parents=True)
    by_task = defaultdict(dict)
    for row in summary["results"]:
        if row["method"] in ("frozen", "frozen_stochastic", "untrained_stochastic", "random", "greedy", "evolutionary",
                             "parzen", "online_adapt") and row["task_id"].startswith("test_"):
            by_task[row["task_sha"]][row["method"]] = row
    paired = []
    methods = ("frozen", "frozen_stochastic", "untrained_stochastic", "online_adapt",
               "greedy", "evolutionary", "parzen")
    for task_sha, arms in by_task.items():
        ref = arms.get("random")
        if ref is None or ref["best_feasible_latency_ns"] is None:
            continue
        random_ns = ref["best_feasible_latency_ns"]
        for method in methods:
            arm = arms.get(method)
            if arm is None or arm["best_feasible_latency_ns"] is None:
                continue
            latency = arm["best_feasible_latency_ns"]
            paired.append({
                "task_sha": task_sha, "method": method,
                "random_estimated_ns": random_ns,
                "method_estimated_ns": latency,
                "relative_improvement_vs_random": (random_ns-latency)/random_ns,
                "original_calls_random": ref["original_vitis_calls"],
                "original_calls_method": arm["original_vitis_calls"],
                "paired_equal_call_count":
                    ref["original_vitis_calls"] == arm["original_vitis_calls"],
            })
    fields = ["task_sha", "method", "random_estimated_ns", "method_estimated_ns",
              "relative_improvement_vs_random", "original_calls_random",
              "original_calls_method", "paired_equal_call_count"]
    with (target / "paired_test.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)
    curves = defaultdict(list)
    for record in observations:
        if record["best_feasible_latency_ns"] is not None:
            curves[(record["task_sha"], record["method"])].append(record)
    with (target / "best_vs_native_calls.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "task_sha", "method", "native_call", "best_feasible_hls_ns"])
        writer.writeheader()
        for (task_sha, method), rows in curves.items():
            for i, row in enumerate(rows, 1):
                writer.writerow({
                    "task_sha": task_sha, "method": method,
                    "native_call": row["native_call_index"] + 1,
                    "best_feasible_hls_ns": row["best_feasible_latency_ns"],
                })
    # Do not turn censored/no-feasible tasks into nonexistent speedup claims.
    # Report success frequency on ALL test tasks, even when zero pairs exist.
    feasibility_counts = {}
    for method in ("random", *methods):
        arms = [mapping.get(method) for mapping in by_task.values()]
        feasibility_counts[method] = {
            "test_tasks_total": len(arms),
            "feasible_test_tasks": sum(
                row is not None and row["best_feasible_latency_ns"] is not None
                for row in arms),
            "test_native_calls": sum(
                row["original_vitis_calls"] for row in arms if row is not None),
            "paired_improvement_not_identifiable_if_random_infeasible": True,
        }
    stats = {}
    for method in methods:
        subset = [r for r in paired if r["method"] == method
                  and r["paired_equal_call_count"]]
        effects = [r["relative_improvement_vs_random"] for r in subset]
        stats[method] = {
            "n_paired_feasible_equal_budget": len(effects),
            "mean_relative_improvement_vs_random":
                float(np.mean(effects)) if effects else None,
            "bootstrap_95pct_ci": _ci(effects),
            "omitted_unmatched_or_infeasible": True,
        }
    chart = False
    if any(curves.values()):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            fig, ax = plt.subplots()
            for method in ("frozen", "frozen_stochastic",
                           "untrained_stochastic", "random", "online_adapt"):
                matched = [rows for (task, m), rows in curves.items() if m == method]
                if not matched:
                    continue
                # Show each independently attested task trajectory, avoiding
                # arbitrary aggregation of heterogeneous absolute latencies.
                for rows in matched:
                    x = [r["native_call_index"]+1 for r in rows]
                    y = [r["best_feasible_latency_ns"] for r in rows]
                    ax.step(x, y, where="post", label=method)
            ax.set_xlabel("Original Vitis HLS evaluations per task")
            ax.set_ylabel("Best feasible HLS-estimated latency (ns)")
            fig.tight_layout()
            fig.savefig(target / "native_hls_latency_curves.png", dpi=160)
            plt.close(fig)
            chart = True
    result = {
        "schema": "fpgai-gym-audited-hls-report/v1",
        "independent_audit": passed,
        "experiment_source_revision": summary["source_revision"],
        "paired_test_effects": stats,
        "all_test_task_feasibility": feasibility_counts,
        "paired_test_rows": len(paired),
        "native_latency_curve_chart": chart,
        "not_physical_or_routed_latency": True,
        "publication_claim_ready": False,
    }
    (target / "report.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    print(json.dumps(export(a.study, a.output), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
