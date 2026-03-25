from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def read_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def to_float(x: Any) -> Optional[float]:
    try:
        if x in (None, "", "None"):
            return None
        return float(x)
    except Exception:
        return None


def to_bool(x: Any) -> Optional[bool]:
    if x in (None, "", "None"):
        return None
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def unique_in_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def filter_ok(rows: List[Dict[str, Any]], require_benchmark: bool = False) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        compile_ok = to_bool(r.get("compile_ok"))
        if compile_ok is False:
            continue
        if require_benchmark:
            passed = r.get("benchmark_passed")
            if passed in ("", None, "None"):
                continue
        out.append(r)
    return out


def group_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[str(r.get(key, ""))].append(r)
    return out


def savefig(path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_error_vs_precision(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    by_model = group_by(rows, "model_name")

    for model_name, model_rows in by_model.items():
        parallel_policies = unique_in_order([str(r["parallel_policy"]) for r in model_rows])
        precision_labels = unique_in_order([str(r["precision_policy"]) for r in model_rows])

        label_to_x = {lab: i for i, lab in enumerate(precision_labels)}

        plt.figure(figsize=(8, 5))
        for pp in parallel_policies:
            xs = []
            ys = []
            for r in model_rows:
                if str(r["parallel_policy"]) != pp:
                    continue
                y = to_float(r.get("rmse"))
                x = label_to_x[str(r["precision_policy"])]
                if y is None:
                    continue
                xs.append(x)
                ys.append(y)
            if xs:
                pairs = sorted(zip(xs, ys), key=lambda t: t[0])
                plt.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=pp)

        plt.xticks(list(label_to_x.values()), list(label_to_x.keys()), rotation=20)
        plt.legend()
        savefig(
            out_dir / f"{model_name}_error_vs_precision.png",
            title=f"{model_name}: RMSE vs Precision Policy",
            xlabel="Precision Policy",
            ylabel="RMSE",
        )


def plot_latency_vs_parallel(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    by_model = group_by(rows, "model_name")

    for model_name, model_rows in by_model.items():
        precision_policies = unique_in_order([str(r["precision_policy"]) for r in model_rows])
        parallel_labels = unique_in_order([str(r["parallel_policy"]) for r in model_rows])

        label_to_x = {lab: i for i, lab in enumerate(parallel_labels)}

        plt.figure(figsize=(8, 5))
        for prec in precision_policies:
            xs = []
            ys = []
            for r in model_rows:
                if str(r["precision_policy"]) != prec:
                    continue
                y = to_float(r.get("latency_seconds_max")) or to_float(r.get("latency_seconds_min"))
                x = label_to_x[str(r["parallel_policy"])]
                if y is None:
                    continue
                xs.append(x)
                ys.append(y * 1e3)  # ms
            if xs:
                pairs = sorted(zip(xs, ys), key=lambda t: t[0])
                plt.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=prec)

        plt.xticks(list(label_to_x.values()), list(label_to_x.keys()), rotation=20)
        plt.legend()
        savefig(
            out_dir / f"{model_name}_latency_vs_parallel.png",
            title=f"{model_name}: Latency vs Parallel Policy",
            xlabel="Parallel Policy",
            ylabel="Latency (ms)",
        )


def plot_resources_vs_precision(rows: List[Dict[str, Any]], out_dir: Path, resource_key: str, resource_label: str) -> None:
    by_model = group_by(rows, "model_name")

    for model_name, model_rows in by_model.items():
        parallel_policies = unique_in_order([str(r["parallel_policy"]) for r in model_rows])
        precision_labels = unique_in_order([str(r["precision_policy"]) for r in model_rows])
        label_to_x = {lab: i for i, lab in enumerate(precision_labels)}

        plt.figure(figsize=(8, 5))
        for pp in parallel_policies:
            xs = []
            ys = []
            for r in model_rows:
                if str(r["parallel_policy"]) != pp:
                    continue
                y = to_float(r.get(resource_key))
                x = label_to_x[str(r["precision_policy"])]
                if y is None:
                    continue
                xs.append(x)
                ys.append(y)
            if xs:
                pairs = sorted(zip(xs, ys), key=lambda t: t[0])
                plt.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=pp)

        plt.xticks(list(label_to_x.values()), list(label_to_x.keys()), rotation=20)
        plt.legend()
        savefig(
            out_dir / f"{model_name}_{resource_key}_vs_precision.png",
            title=f"{model_name}: {resource_label} vs Precision Policy",
            xlabel="Precision Policy",
            ylabel=resource_label,
        )


def plot_pareto(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    by_model = group_by(rows, "model_name")

    for model_name, model_rows in by_model.items():
        plt.figure(figsize=(7, 5))
        for r in model_rows:
            x = to_float(r.get("latency_seconds_max")) or to_float(r.get("latency_seconds_min"))
            y = to_float(r.get("rmse"))
            size = to_float(r.get("dsp")) or 10.0
            if x is None or y is None:
                continue
            label = f'{r["precision_policy"]} | {r["parallel_policy"]}'
            plt.scatter([x * 1e3], [y], s=max(20.0, size * 3.0), label=label)

        handles, labels = plt.gca().get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        if unique:
            plt.legend(unique.values(), unique.keys(), fontsize=7)

        savefig(
            out_dir / f"{model_name}_pareto_latency_rmse.png",
            title=f"{model_name}: Pareto (Latency vs RMSE)",
            xlabel="Latency (ms)",
            ylabel="RMSE",
        )


def plot_heatmap_metric(rows: List[Dict[str, Any]], out_dir: Path, metric_key: str, title_suffix: str) -> None:
    by_model = group_by(rows, "model_name")

    for model_name, model_rows in by_model.items():
        precision_labels = unique_in_order([str(r["precision_policy"]) for r in model_rows])
        parallel_labels = unique_in_order([str(r["parallel_policy"]) for r in model_rows])

        grid = []
        for prec in precision_labels:
            row_vals = []
            for par in parallel_labels:
                match = None
                for r in model_rows:
                    if str(r["precision_policy"]) == prec and str(r["parallel_policy"]) == par:
                        match = r
                        break
                val = to_float(match.get(metric_key)) if match else None
                row_vals.append(float("nan") if val is None else val)
            grid.append(row_vals)

        plt.figure(figsize=(1.3 * len(parallel_labels) + 2, 1.0 * len(precision_labels) + 2))
        plt.imshow(grid, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(parallel_labels)), parallel_labels, rotation=20)
        plt.yticks(range(len(precision_labels)), precision_labels)

        for i in range(len(precision_labels)):
            for j in range(len(parallel_labels)):
                val = grid[i][j]
                txt = "NA" if math.isnan(val) else f"{val:.3g}"
                plt.text(j, i, txt, ha="center", va="center", fontsize=8)

        savefig(
            out_dir / f"{model_name}_{metric_key}_heatmap.png",
            title=f"{model_name}: {title_suffix}",
            xlabel="Parallel Policy",
            ylabel="Precision Policy",
        )


def main() -> None:
    parser = argparse.ArgumentParser("Plot FPGAI policy sweep results")
    parser.add_argument("--csv", required=True, help="CSV produced by run_policy_sweep.py")
    parser.add_argument("--out-dir", default="build/policy_sweeps/plots")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    rows = filter_ok(rows)

    if not rows:
        raise SystemExit("No successful rows found in CSV.")

    plot_error_vs_precision(rows, out_dir)
    plot_latency_vs_parallel(rows, out_dir)
    plot_resources_vs_precision(rows, out_dir, "lut", "LUT")
    plot_resources_vs_precision(rows, out_dir, "dsp", "DSP")
    plot_resources_vs_precision(rows, out_dir, "bram_18k", "BRAM_18K")
    plot_pareto(rows, out_dir)
    plot_heatmap_metric(rows, out_dir, "rmse", "RMSE Heatmap")
    plot_heatmap_metric(rows, out_dir, "latency_seconds_max", "Latency Heatmap (s)")
    plot_heatmap_metric(rows, out_dir, "dsp", "DSP Heatmap")

    print(f"[OK] Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()