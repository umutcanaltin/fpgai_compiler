from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def to_bool(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
    )


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "dsp", "lut", "ff", "bram_18k", "uram",
        "latency_ms", "latency_seconds_min", "latency_seconds_max",
        "plan_notes_act_bits", "plan_notes_weight_bits",
        "plan_notes_bias_bits", "plan_notes_accum_bits",
        "traincmp_global_grads_mae",
        "traincmp_global_grads_max_abs",
        "traincmp_global_grads_cosine",
        "traincmp_global_grads_relative_l2",
        "traincmp_global_weights_before_mae",
        "traincmp_global_weights_before_max_abs",
        "traincmp_global_weights_before_cosine",
        "traincmp_global_weights_after_mae",
        "traincmp_global_weights_after_max_abs",
        "traincmp_global_weights_after_cosine",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "compile_ok" in df.columns:
        df = df[to_bool(df["compile_ok"])]
    if "hls_ok" in df.columns:
        df = df[to_bool(df["hls_ok"])]

    return df


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "model_name",
        "precision_policy",
        "parallel_policy",
        "plan_notes_act_bits",
        "plan_notes_weight_bits",
        "plan_notes_bias_bits",
        "plan_notes_accum_bits",
        "dsp",
        "lut",
        "ff",
        "bram_18k",
        "uram",
        "latency_ms",
        "traincmp_global_grads_cosine",
        "traincmp_global_grads_mae",
        "traincmp_global_grads_max_abs",
        "traincmp_global_weights_after_cosine",
        "traincmp_global_weights_after_mae",
        "traincmp_global_weights_after_max_abs",
        "out_dir",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    rename_map = {
        "model_name": "model",
        "precision_policy": "precision",
        "parallel_policy": "policy",
        "plan_notes_act_bits": "act_bits",
        "plan_notes_weight_bits": "weight_bits",
        "plan_notes_bias_bits": "bias_bits",
        "plan_notes_accum_bits": "accum_bits",
        "bram_18k": "bram18k",
        "traincmp_global_grads_cosine": "grad_cosine",
        "traincmp_global_grads_mae": "grad_mae",
        "traincmp_global_grads_max_abs": "grad_max_abs",
        "traincmp_global_weights_after_cosine": "w_after_cosine",
        "traincmp_global_weights_after_mae": "w_after_mae",
        "traincmp_global_weights_after_max_abs": "w_after_max_abs",
    }
    out = out.rename(columns=rename_map)
    return out


def plot_latency_vs_precision(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    order = ["Uniform-12", "Mixed-Conservative", "Uniform-16"]
    df = df.copy()
    df["precision_policy"] = pd.Categorical(df["precision_policy"], categories=order, ordered=True)
    df = df.sort_values("precision_policy")

    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(111)
    ax.bar(df["precision_policy"].astype(str), df["latency_ms"])
    ax.set_title("Training latency vs precision")
    ax.set_xlabel("Precision policy")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_latency_vs_precision.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}_latency_vs_precision.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_resources_vs_precision(df: pd.DataFrame, outdir: Path, stem: str, metric: str, ylabel: str) -> None:
    if metric not in df.columns:
        return

    order = ["Uniform-12", "Mixed-Conservative", "Uniform-16"]
    tmp = df.copy()
    tmp["precision_policy"] = pd.Categorical(tmp["precision_policy"], categories=order, ordered=True)
    tmp = tmp.sort_values("precision_policy")

    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(111)
    ax.bar(tmp["precision_policy"].astype(str), tmp[metric])
    ax.set_title(f"{ylabel} vs precision")
    ax.set_xlabel("Precision policy")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_{metric}_vs_precision.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}_{metric}_vs_precision.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training_stability(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    metric = None
    for c in [
        "traincmp_global_weights_after_cosine",
        "traincmp_global_grads_cosine",
        "traincmp_global_weights_after_mae",
        "traincmp_global_grads_mae",
    ]:
        if c in df.columns and df[c].notna().any():
            metric = c
            break

    if metric is None:
        return

    order = ["Uniform-12", "Mixed-Conservative", "Uniform-16"]
    tmp = df.copy()
    tmp["precision_policy"] = pd.Categorical(tmp["precision_policy"], categories=order, ordered=True)
    tmp = tmp.sort_values("precision_policy")

    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(tmp["precision_policy"].astype(str), tmp[metric], marker="o")
    ax.set_title(f"{metric} vs precision")
    ax.set_xlabel("Precision policy")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_{metric}.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}_{metric}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_discussion(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    lines: List[str] = []

    if len(df) == 0:
        (outdir / f"{stem}_discussion.txt").write_text("", encoding="utf-8")
        return

    best_latency = df.sort_values("latency_ms").iloc[0]
    min_dsp = df.sort_values("dsp").iloc[0] if "dsp" in df.columns else None
    min_lut = df.sort_values("lut").iloc[0] if "lut" in df.columns else None

    lines.append(
        f"For training-oriented compilation on {best_latency['model_name']}, "
        f"the precision sweep shows a measurable resource–latency tradeoff. "
        f"The lowest-latency configuration is {best_latency['precision_policy']} "
        f"at {best_latency['latency_ms']:.3f} ms."
    )

    if min_dsp is not None:
        lines.append(
            f"In terms of DSP usage, the smallest design is {min_dsp['precision_policy']} "
            f"with {int(min_dsp['dsp'])} DSP."
        )

    if min_lut is not None:
        lines.append(
            f"In terms of LUT usage, the smallest design is {min_lut['precision_policy']} "
            f"with {int(min_lut['lut'])} LUT."
        )

    if "traincmp_global_weights_after_cosine" in df.columns and df["traincmp_global_weights_after_cosine"].notna().any():
        best_cos = df.sort_values("traincmp_global_weights_after_cosine", ascending=False).iloc[0]
        lines.append(
            f"Training stability remains strong, with the best post-update cosine similarity "
            f"observed for {best_cos['precision_policy']} at "
            f"{best_cos['traincmp_global_weights_after_cosine']:.6f}."
        )

    if "traincmp_global_grads_mae" in df.columns and df["traincmp_global_grads_mae"].notna().any():
        best_mae = df.sort_values("traincmp_global_grads_mae").iloc[0]
        lines.append(
            f"The smallest gradient MAE is obtained by {best_mae['precision_policy']}, "
            f"indicating the closest gradient agreement with the software reference."
        )

    (outdir / f"{stem}_discussion.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate training precision tables and figures for FPGAI paper.")
    ap.add_argument("--csv", required=True, help="Training sweep CSV")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    if df.empty:
        raise SystemExit("No valid rows found in training CSV")

    model_name = str(df["model_name"].iloc[0])
    stem = f"{model_name}_training_precision"

    table = make_summary_table(df)
    table.to_csv(outdir / f"{stem}_table.csv", index=False)

    plot_latency_vs_precision(df, outdir, stem)
    plot_resources_vs_precision(df, outdir, stem, "dsp", "DSP")
    plot_resources_vs_precision(df, outdir, stem, "lut", "LUT")
    plot_resources_vs_precision(df, outdir, stem, "bram_18k", "BRAM_18K")
    plot_training_stability(df, outdir, stem)
    write_discussion(df, outdir, stem)

    print(f"[OK] wrote outputs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())