#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _precision_order_key(name: str) -> tuple[int, str]:
    n = str(name)
    if n.startswith("Uniform-"):
        try:
            return (0, f"{int(n.split('-')[1]):04d}")
        except Exception:
            return (0, n)
    if n == "Mixed-Conservative":
        return (1, n)
    if n == "Mixed-Aggressive":
        return (2, n)
    return (9, n)


def _parallel_order_key(name: str) -> tuple[int, str]:
    order = {
        "Resource-First": 0,
        "Balanced": 1,
        "Latency-First": 2,
    }
    n = str(name)
    return (order.get(n, 9), n)


def _prepare_precision(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "parallel_policy" in d.columns and "Resource-First" in set(d["parallel_policy"].dropna()):
        d = d[d["parallel_policy"] == "Resource-First"]
    d = d.sort_values(
        by=["model_name", "precision_policy"],
        key=lambda s: s.map(lambda x: _precision_order_key(str(x))) if s.name == "precision_policy" else s,
    )
    d["x_label"] = d["precision_policy"].astype(str)
    return d


def _prepare_parallel(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "precision_policy" in d.columns and "Uniform-12" in set(d["precision_policy"].dropna()):
        d = d[d["precision_policy"] == "Uniform-12"]
    d = d.sort_values(
        by=["model_name", "parallel_policy"],
        key=lambda s: s.map(lambda x: _parallel_order_key(str(x))) if s.name == "parallel_policy" else s,
    )
    d["x_label"] = d["parallel_policy"].astype(str)
    return d


def make_multiplot(df: pd.DataFrame, mode: str, out_base: Path) -> None:
    if mode == "precision":
        d = _prepare_precision(df)
        x_col = "x_label"
        metrics = [
            ("latency_cycles_max", "Latency (cycles)"),
            ("lut", "LUT"),
            ("bram_18k", "BRAM_18K"),
            ("rmse", "RMSE"),
        ]
        title = "FPGAI Precision Sweep Summary"
    else:
        d = _prepare_parallel(df)
        x_col = "x_label"
        metrics = [
            ("latency_cycles_max", "Latency (cycles)"),
            ("dsp", "DSP"),
            ("lut", "LUT"),
            ("ff", "FF"),
        ]
        title = "FPGAI Parallel Policy Sweep Summary"

    models = [m for m in ["mlp_mnist", "cnn_mnist"] if m in set(d["model_name"].dropna())]
    if not models:
        raise ValueError("No expected models found in CSV.")

    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=len(metrics),
        figsize=(14, 5.6 if len(models) == 2 else 3.2),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14)

    for r, model in enumerate(models):
        dm = d[d["model_name"] == model].copy()
        for c, (metric, ylabel) in enumerate(metrics):
            ax = axes[r][c]
            plot_df = dm[[x_col, metric]].dropna()
            if plot_df.empty:
                ax.set_visible(False)
                continue

            ax.bar(plot_df[x_col], plot_df[metric])
            if r == 0:
                ax.set_title(ylabel)
            if c == 0:
                ax.set_ylabel(model)
            ax.tick_params(axis="x", rotation=20)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one compact multi-panel paper plot.")
    parser.add_argument("--csv", required=True, help="Merged CSV path")
    parser.add_argument("--mode", choices=["precision", "parallel"], required=True)
    parser.add_argument("--out", required=True, help="Output base path without extension")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = _to_num(
        df,
        [
            "rmse",
            "latency_cycles_max",
            "lut",
            "ff",
            "dsp",
            "bram_18k",
        ],
    )

    make_multiplot(df, args.mode, Path(args.out))
    print(f"Wrote {Path(args.out).with_suffix('.png')} and .pdf")


if __name__ == "__main__":
    main()