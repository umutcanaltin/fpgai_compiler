from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


STABLE_PRECISIONS = ["Uniform-12", "Uniform-16", "Mixed-Conservative"]
STABLE_POLICIES = ["Balanced", "Throughput-First", "Latency-First"]
MODEL_ORDER = ["mlp_mnist", "cnn_mnist"]


def sanitize_latex(x: str) -> str:
    return (
        str(x)
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in [
        "compile_ok",
        "hls_ok",
        "benchmark_passed",
        "quant_match",
        "quant_argmax_match",
        "bench_argmax_match",
        "allow_double_buffer",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False}
            ).fillna(df[col])

    return df


def keep_final_rows(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x[x["precision_policy"].isin(STABLE_PRECISIONS)]
    x = x[x["parallel_policy"].isin(STABLE_POLICIES)]
    x = x[x["model_name"].isin(MODEL_ORDER)]
    x = x[x["compile_ok"] == True]
    x = x[x["hls_ok"] == True]
    x = x[x["benchmark_passed"] == True]
    return x


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_table_tex(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    cols = "l" * len(df.columns)
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\hline")
    lines.append(" & ".join(sanitize_latex(c) for c in df.columns) + r" \\")
    lines.append(r"\hline")
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                vals.append("-")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(sanitize_latex(v))
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path.write_text("\n".join(lines), encoding="utf-8")


def table_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model_name",
        "precision_policy",
        "parallel_policy",
        "quant_match",
        "quant_cosine",
        "quant_mse",
        "quant_rmse",
        "quant_max_abs_error",
        "quant_argmax_match",
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    return out.sort_values(["model_name", "precision_policy", "parallel_policy"])


def table_resource_latency(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model_name",
        "precision_policy",
        "parallel_policy",
        "lut",
        "ff",
        "dsp",
        "bram_18k",
        "estimated_clock_ns",
        "latency_cycles_min",
        "hls_latency_ms",
        "latency_per_dsp",
        "latency_per_lut",
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    return out.sort_values(["model_name", "parallel_policy", "precision_policy"])


def table_compile_decisions(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model_name",
        "precision_policy",
        "parallel_policy",
        "compile_clock_mhz",
        "parallel_pe",
        "parallel_simd",
        "parallel_partition_factor",
        "conv_tile_oh",
        "conv_tile_ow",
        "conv_tile_oc",
        "conv_unroll_ic",
        "conv_unroll_oc",
        "dense_tile_in",
        "dense_tile_out",
        "dense_unroll_in",
        "dense_unroll_out",
        "conv_weight_mode",
        "dense_weight_mode",
        "allow_double_buffer",
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    return out.sort_values(["model_name", "parallel_policy", "precision_policy"])


def plot_precision_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    x = df.copy()
    x = x.sort_values(["model_name", "precision_policy", "parallel_policy"])

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for model_name in MODEL_ORDER:
        sub = x[x["model_name"] == model_name]
        if sub.empty:
            continue
        ax.scatter(sub["quant_rmse"], sub["lut"], label=model_name)

    ax.set_xlabel("RMSE")
    ax.set_ylabel("LUT")
    ax.set_title("Precision vs correctness/resource tradeoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_policy_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    x = df.copy()
    x = x.sort_values(["model_name", "parallel_policy", "precision_policy"])

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for pol in STABLE_POLICIES:
        sub = x[x["parallel_policy"] == pol]
        if sub.empty:
            continue
        ax.scatter(sub["hls_latency_ms"], sub["dsp"], label=pol)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("DSP")
    ax.set_title("Parallel policy vs latency/resource tradeoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bottleneck_breakdown(df: pd.DataFrame, out_path: Path) -> None:
    x = (
        df.groupby("model_name")[["lut", "dsp", "bram_18k", "hls_latency_ms"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    width = 0.2
    idx = range(len(x))
    ax.bar([i - 1.5 * width for i in idx], x["lut"], width, label="LUT")
    ax.bar([i - 0.5 * width for i in idx], x["dsp"], width, label="DSP")
    ax.bar([i + 0.5 * width for i in idx], x["bram_18k"], width, label="BRAM18")
    ax.bar([i + 1.5 * width for i in idx], x["hls_latency_ms"], width, label="Latency(ms)")

    ax.set_xticks(list(idx))
    ax.set_xticklabels(x["model_name"].tolist())
    ax.set_title("End-to-end bottleneck breakdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_summary_txt(df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("FPGAI Final Paper Experiment Summary")
    lines.append("=" * 40)
    lines.append(f"Successful runs: {len(df)}")
    lines.append("")

    for model_name in MODEL_ORDER:
        sub = df[df["model_name"] == model_name]
        if sub.empty:
            continue

        lines.append(f"[{model_name}]")
        best_latency = sub.sort_values("hls_latency_ms").iloc[0]
        best_lut = sub.sort_values("lut").iloc[0]
        best_safe = sub.sort_values(["quant_rmse", "hls_latency_ms"]).iloc[0]

        lines.append(
            f"  Best latency: {best_latency['precision_policy']} / "
            f"{best_latency['parallel_policy']} / {best_latency['hls_latency_ms']:.4f} ms"
        )
        lines.append(
            f"  Best LUT: {best_lut['precision_policy']} / "
            f"{best_lut['parallel_policy']} / LUT={int(best_lut['lut'])}"
        )
        lines.append(
            f"  Best balanced-safe: {best_safe['precision_policy']} / "
            f"{best_safe['parallel_policy']} / RMSE={best_safe['quant_rmse']:.6f} / "
            f"Latency={best_safe['hls_latency_ms']:.4f} ms"
        )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="policy_sweep_results.csv")
    ap.add_argument("--outdir", required=True, help="output folder for paper artifacts")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    outdir = Path(args.outdir).resolve()

    ensure_dir(outdir)
    ensure_dir(outdir / "tables")
    ensure_dir(outdir / "figures")

    df = read_csv(csv_path)
    final_df = keep_final_rows(df)

    final_df.to_csv(outdir / "final_successful_runs.csv", index=False)

    acc_df = table_accuracy(final_df)
    res_df = table_resource_latency(final_df)
    plan_df = table_compile_decisions(final_df)

    acc_df.to_csv(outdir / "tables" / "table_accuracy.csv", index=False)
    res_df.to_csv(outdir / "tables" / "table_resource_latency.csv", index=False)
    plan_df.to_csv(outdir / "tables" / "table_compile_decisions.csv", index=False)

    save_table_tex(
        acc_df,
        outdir / "tables" / "table_accuracy.tex",
        "Accuracy and correctness across precision policies.",
        "tab:accuracy",
    )
    save_table_tex(
        res_df,
        outdir / "tables" / "table_resource_latency.tex",
        "Resource utilization and latency across policies.",
        "tab:resource_latency",
    )
    save_table_tex(
        plan_df,
        outdir / "tables" / "table_compile_decisions.tex",
        "Compiler-selected configurations and resulting hardware choices.",
        "tab:compile_decisions",
    )

    if not final_df.empty:
        plot_precision_tradeoff(final_df, outdir / "figures" / "precision_vs_tradeoff.png")
        plot_policy_tradeoff(final_df, outdir / "figures" / "policy_vs_tradeoff.png")
        plot_bottleneck_breakdown(final_df, outdir / "figures" / "bottleneck_breakdown.png")

    write_summary_txt(final_df, outdir / "summary.txt")

    print(f"[OK] Paper artifacts written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())