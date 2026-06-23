from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
    )


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _pick_latency_seconds(df: pd.DataFrame) -> pd.Series:
    if "latency_seconds_max" in df.columns:
        lat = pd.to_numeric(df["latency_seconds_max"], errors="coerce")
        if "latency_seconds_min" in df.columns:
            lat = lat.fillna(pd.to_numeric(df["latency_seconds_min"], errors="coerce"))
        return lat
    if "latency_seconds_min" in df.columns:
        return pd.to_numeric(df["latency_seconds_min"], errors="coerce")
    raise KeyError("Expected latency_seconds_max or latency_seconds_min in CSV")


def _pick_error_metric(df: pd.DataFrame) -> Optional[pd.Series]:
    for c in ["quant_rmse", "bench_rmse", "quant_mse", "bench_mse"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return None


def load_rows(csv_path: Path, require_pass: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _to_numeric(
        df,
        [
            "dsp",
            "lut",
            "ff",
            "bram_18k",
            "uram",
            "ii",
            "estimated_clock_ns",
            "latency_cycles_min",
            "latency_cycles_max",
            "latency_seconds_min",
            "latency_seconds_max",
            "bench_latency_ms",
            "quant_rmse",
            "quant_mse",
            "quant_mae",
            "quant_cosine",
            "bench_rmse",
            "bench_mse",
            "bench_mae",
            "bench_cosine",
        ],
    )

    if "compile_ok" in df.columns:
        df = df[_to_bool(df["compile_ok"]).fillna(False)]
    if "hls_ok" in df.columns:
        df = df[_to_bool(df["hls_ok"]).fillna(False)]
    if require_pass and "benchmark_passed" in df.columns:
        df = df[_to_bool(df["benchmark_passed"]).fillna(False)]

    df = df.copy()
    df["latency_s"] = _pick_latency_seconds(df)
    df["latency_ms"] = df["latency_s"] * 1e3

    err = _pick_error_metric(df)
    if err is not None:
        df["error_metric"] = err

    needed = ["model_name", "precision_policy", "parallel_policy", "dsp", "latency_ms"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required CSV columns: {missing}")

    df = df.dropna(
        subset=["model_name", "precision_policy", "parallel_policy", "dsp", "latency_ms"]
    )
    return df


def pareto_mask_minimize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            no_worse = (x[j] <= x[i]) and (y[j] <= y[i])
            strictly_better = (x[j] < x[i]) or (y[j] < y[i])
            if no_worse and strictly_better:
                mask[i] = False
                break
    return mask


def select_knee(frontier: pd.DataFrame) -> int:
    if len(frontier) == 1:
        return 0

    pts = frontier[["dsp", "latency_ms"]].to_numpy(dtype=float)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    denom = np.where((maxs - mins) == 0.0, 1.0, (maxs - mins))
    norm = (pts - mins) / denom

    p0 = norm[0]
    p1 = norm[-1]
    line = p1 - p0
    line_norm = np.linalg.norm(line)

    if line_norm == 0.0:
        return int(np.argmin(norm.sum(axis=1)))

    distances = []
    for p in norm:
        # 2D point-to-line distance using cross product magnitude
        dist = abs(np.cross(line, p - p0)) / line_norm
        distances.append(float(dist))

    return int(np.argmax(distances))


def analyze_model(df_model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    working = (
        df_model.sort_values(
            ["dsp", "latency_ms", "precision_policy", "parallel_policy"]
        )
        .reset_index(drop=True)
        .copy()
    )

    mask = pareto_mask_minimize(
        working["dsp"].to_numpy(float),
        working["latency_ms"].to_numpy(float),
    )
    frontier = working[mask].sort_values(["dsp", "latency_ms"]).reset_index(drop=True)

    knee_idx = select_knee(frontier)
    knee_row = frontier.iloc[knee_idx].copy()

    frontier = frontier.copy()
    frontier["is_knee"] = False
    frontier.loc[knee_idx, "is_knee"] = True

    return working, frontier, knee_row


def make_plot(
    df_model: pd.DataFrame,
    frontier: pd.DataFrame,
    knee_row: pd.Series,
    model_name: str,
    out_base: Path,
) -> None:
    fig = plt.figure(figsize=(7.6, 5.2))
    ax = fig.add_subplot(111)

    # all valid points
    ax.scatter(df_model["dsp"], df_model["latency_ms"], alpha=0.75, label="All valid designs")

    # pareto frontier
    ax.plot(
        frontier["dsp"],
        frontier["latency_ms"],
        marker="o",
        linewidth=1.5,
        label="Pareto frontier",
    )

    # knee point
    ax.scatter(
        [knee_row["dsp"]],
        [knee_row["latency_ms"]],
        marker="*",
        s=220,
        label="Knee point",
    )

    for _, r in frontier.iterrows():
        label = f"{r['precision_policy']} / {r['parallel_policy']}"
        ax.annotate(
            label,
            (r["dsp"], r["latency_ms"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_title(f"{model_name}: DSP-Latency Frontier")
    ax.set_xlabel("DSP usage")
    ax.set_ylabel("Inference latency (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_paragraph(model_name: str, frontier: pd.DataFrame, knee: pd.Series) -> str:
    fastest = frontier.sort_values("latency_ms").iloc[0]
    smallest = frontier.sort_values("dsp").iloc[0]

    err_text = ""
    if "error_metric" in knee.index and pd.notna(knee.get("error_metric")):
        err_text = (
            f" The knee-point configuration also records an error metric of "
            f"{float(knee['error_metric']):.4g}."
        )

    return (
        f"For {model_name}, the DSP-latency frontier shows the expected diminishing-returns trend: "
        f"moving from the minimum-resource frontier point ({int(smallest['dsp'])} DSP, "
        f"{smallest['latency_ms']:.3f} ms) toward the fastest frontier point "
        f"({int(fastest['dsp'])} DSP, {fastest['latency_ms']:.3f} ms) reduces latency, "
        f"but the marginal latency gain per additional DSP decreases along the curve. "
        f"Using a maximum-distance-to-chord knee detector on the Pareto frontier, FPGAI identifies "
        f"{knee['precision_policy']} with {knee['parallel_policy']} as the most balanced operating point, "
        f"at {int(knee['dsp'])} DSP and {knee['latency_ms']:.3f} ms.{err_text} "
        f"This point is a practical operating configuration because it avoids both over-provisioned "
        f"high-DSP designs with limited latency benefit and ultra-small designs that incur "
        f"disproportionate latency penalties."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create DSP-vs-latency Pareto frontier plots and tables for FPGAI sweep CSVs."
    )
    ap.add_argument("--csv", required=True, help="CSV produced by scripts/run_policy_sweep.py")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument(
        "--require-pass",
        action="store_true",
        help="Keep only rows with benchmark_passed=True when available",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_rows(csv_path, require_pass=args.require_pass)
    if df.empty:
        raise SystemExit("No valid rows remain after filtering.")

    all_frontiers = []
    all_knees = []
    discussion_lines = []

    for model_name, df_model in df.groupby("model_name"):
        working, frontier, knee = analyze_model(df_model)

        plot_base = outdir / f"{model_name}_dsp_latency_frontier"
        make_plot(working, frontier, knee, model_name, plot_base)

        frontier_out = frontier.copy()
        frontier_out["model_name"] = model_name
        all_frontiers.append(frontier_out)

        knee_out = pd.DataFrame([knee])
        knee_out["model_name"] = model_name
        all_knees.append(knee_out)

        discussion_lines.append(f"[{model_name}]")
        discussion_lines.append(build_paragraph(model_name, frontier, knee))
        discussion_lines.append("")

    frontier_df = pd.concat(all_frontiers, ignore_index=True)
    knee_df = pd.concat(all_knees, ignore_index=True)

    keep_cols = [
        "model_name",
        "precision_policy",
        "parallel_policy",
        "dsp",
        "lut",
        "ff",
        "bram_18k",
        "uram",
        "ii",
        "estimated_clock_ns",
        "latency_cycles_min",
        "latency_cycles_max",
        "latency_ms",
        "error_metric",
        "quant_cosine",
        "quant_mse",
        "quant_mae",
        "quant_rmse",
        "bench_rmse",
        "bench_mae",
        "bench_mse",
        "bench_cosine",
        "is_knee",
        "out_dir",
    ]

    frontier_cols = [c for c in keep_cols if c in frontier_df.columns]
    knee_cols = [c for c in keep_cols if c in knee_df.columns]

    frontier_df[frontier_cols].to_csv(outdir / "frontier_points.csv", index=False)
    knee_df[knee_cols].to_csv(outdir / "frontier_knees.csv", index=False)
    (outdir / "paper_discussion.txt").write_text(
        "\n".join(discussion_lines),
        encoding="utf-8",
    )

    print(f"[OK] Wrote frontier artifacts to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())