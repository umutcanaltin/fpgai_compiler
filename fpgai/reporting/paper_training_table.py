from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def to_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
    )


def load_training_csv(path: Path, sweep_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    numeric_cols = [
        "latency_cycles_min",
        "latency_cycles_max",
        "latency_ms",
        "lut",
        "ff",
        "dsp",
        "bram_18k",
        "uram",
        "traincmp_global_grads_cosine",
        "traincmp_global_weights_after_cosine",
        "traincmp_global_weight_after_cosine",
        "traincmp_grad_cosine",
        "traincmp_weight_after_cosine",
        "traincmp_weight_delta_cosine",
        "traincmp_global_grads_mae",
        "traincmp_global_weights_after_mae",
        "traincmp_grad_mae",
        "traincmp_weight_after_mae",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "compile_ok" in df.columns:
        df = df[to_bool(df["compile_ok"])]
    if "hls_ok" in df.columns:
        df = df[to_bool(df["hls_ok"])]

    df = df.copy()
    df["sweep"] = sweep_name
    return df


def normalize_model_name(name: str) -> str:
    s = str(name).strip().lower()
    if "mlp" in s:
        return "MLP-MNIST"
    if "cnn" in s:
        return "CNN-MNIST"
    return str(name)


def normalize_policy_name(row: pd.Series) -> str:
    sweep = str(row["sweep"])
    if sweep == "Precision":
        return str(row["precision_policy"])
    return str(row["parallel_policy"])


def choose_latency_cycles(row: pd.Series) -> int:
    if pd.notna(row.get("latency_cycles_max")):
        return int(row["latency_cycles_max"])
    if pd.notna(row.get("latency_cycles_min")):
        return int(row["latency_cycles_min"])
    return -1


def resolve_metric_column(df: pd.DataFrame, metric_preference: str) -> tuple[Optional[str], str]:
    metric_preference = metric_preference.strip().lower()

    metric_map = {
        "grad_cosine": [
            "traincmp_global_grads_cosine",
            "traincmp_grad_cosine",
        ],
        "weight_after_cosine": [
            "traincmp_global_weights_after_cosine",
            "traincmp_global_weight_after_cosine",
            "traincmp_weight_after_cosine",
        ],
        "weight_delta_cosine": [
            "traincmp_weight_delta_cosine",
        ],
        "grad_mae": [
            "traincmp_global_grads_mae",
            "traincmp_grad_mae",
        ],
        "weight_after_mae": [
            "traincmp_global_weights_after_mae",
            "traincmp_weight_after_mae",
        ],
    }

    label_map = {
        "grad_cosine": "Grad Cos.",
        "weight_after_cosine": "W-After Cos.",
        "weight_delta_cosine": "W-Delta Cos.",
        "grad_mae": "Grad MAE",
        "weight_after_mae": "W-After MAE",
    }

    candidates = metric_map.get(metric_preference, metric_map["grad_cosine"])
    label = label_map.get(metric_preference, "Grad Cos.")

    for c in candidates:
        if c in df.columns and df[c].notna().any():
            return c, label

    return None, label


def build_table_df(
    precision_df: pd.DataFrame,
    parallel_df: pd.DataFrame,
    metric_preference: str,
) -> tuple[pd.DataFrame, str]:
    frames: List[pd.DataFrame] = []
    if precision_df is not None and not precision_df.empty:
        frames.append(precision_df)
    if parallel_df is not None and not parallel_df.empty:
        frames.append(parallel_df)

    if not frames:
        return pd.DataFrame(), "Metric"

    df = pd.concat(frames, ignore_index=True).copy()

    metric_col, metric_label = resolve_metric_column(df, metric_preference)

    df["Model"] = df["model_name"].apply(normalize_model_name)
    df["Sweep"] = df["sweep"]
    df["Policy"] = df.apply(normalize_policy_name, axis=1)
    df["Latency (cycles)"] = df.apply(choose_latency_cycles, axis=1)
    df["LUT"] = df["lut"].fillna(0).astype(int)
    df["FF"] = df["ff"].fillna(0).astype(int)
    df["DSP"] = df["dsp"].fillna(0).astype(int)
    df["BRAM"] = df["bram_18k"].fillna(0).astype(int)

    if metric_col is not None:
        df[metric_label] = df[metric_col]
    else:
        df[metric_label] = None

    order_models = {"MLP-MNIST": 0, "CNN-MNIST": 1}
    order_sweep = {"Precision": 0, "Parallel": 1}

    precision_order = {
        "Uniform-8": 0,
        "Uniform-12": 1,
        "Uniform-16": 2,
        "Mixed-Conservative": 3,
    }
    parallel_order = {
        "Resource-First": 0,
        "Fit-First": 0,
        "Balanced": 1,
        "Throughput-First": 2,
        "Latency-First": 3,
    }

    def policy_order(row: pd.Series) -> int:
        if row["Sweep"] == "Precision":
            return precision_order.get(row["Policy"], 99)
        return parallel_order.get(row["Policy"], 99)

    df["_model_order"] = df["Model"].map(order_models).fillna(99)
    df["_sweep_order"] = df["Sweep"].map(order_sweep).fillna(99)
    df["_policy_order"] = df.apply(policy_order, axis=1)

    df = df.sort_values(["_model_order", "_sweep_order", "_policy_order", "Policy"]).reset_index(drop=True)

    out = df[
        ["Model", "Sweep", "Policy", metric_label, "Latency (cycles)", "LUT", "FF", "DSP", "BRAM"]
    ].copy()

    return out, metric_label


def format_metric_value(v) -> str:
    if pd.isna(v):
        return "-"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def dataframe_to_latex_grouped(df: pd.DataFrame, metric_label: str, caption: str, label: str) -> str:
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\caption{" + caption + "}")
    lines.append("\\label{" + label + "}")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{llcccccccc}")
    lines.append("\\hline")
    lines.append(
        f"Model & Sweep & Policy & {metric_label} & Latency (cycles) & LUT & FF & DSP & BRAM \\\\"
    )
    lines.append("\\hline")

    for i, (_, row) in enumerate(df.iterrows()):
        model = row["Model"]
        sweep = row["Sweep"]

        prev_row = df.iloc[i - 1] if i > 0 else None
        next_row = df.iloc[i + 1] if i + 1 < len(df) else None

        model_text = model if prev_row is None or prev_row["Model"] != model else ""
        sweep_text = sweep if (
            prev_row is None
            or prev_row["Model"] != model
            or prev_row["Sweep"] != sweep
        ) else ""

        lines.append(
            f"{model_text} & {sweep_text} & {row['Policy']} & {format_metric_value(row[metric_label])} & "
            f"{int(row['Latency (cycles)'])} & {int(row['LUT'])} & {int(row['FF'])} & "
            f"{int(row['DSP'])} & {int(row['BRAM'])} \\\\"
        )

        if next_row is not None:
            if next_row["Model"] != model or next_row["Sweep"] != sweep:
                lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate ASAP-style training result table.")
    ap.add_argument("--precision-csvs", nargs="*", default=[], help="Training precision sweep CSVs")
    ap.add_argument("--parallel-csvs", nargs="*", default=[], help="Training parallel sweep CSVs")
    ap.add_argument(
        "--metric",
        default="grad_cosine",
        choices=[
            "grad_cosine",
            "weight_after_cosine",
            "weight_delta_cosine",
            "grad_mae",
            "weight_after_mae",
        ],
        help="Training metric to show instead of pass",
    )
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    precision_frames = [load_training_csv(Path(p), "Precision") for p in args.precision_csvs]
    parallel_frames = [load_training_csv(Path(p), "Parallel") for p in args.parallel_csvs]

    precision_df = pd.concat(precision_frames, ignore_index=True) if precision_frames else pd.DataFrame()
    parallel_df = pd.concat(parallel_frames, ignore_index=True) if parallel_frames else pd.DataFrame()

    table_df, metric_label = build_table_df(precision_df, parallel_df, args.metric)
    if table_df.empty:
        raise SystemExit("No valid training rows found.")

    csv_path = outdir / "training_table_summary.csv"
    tex_path = outdir / "training_table_summary.tex"

    table_df.to_csv(csv_path, index=False)

    latex = dataframe_to_latex_grouped(
        table_df,
        metric_label=metric_label,
        caption="FPGAI training-oriented design-space exploration results on KV260 across precision and parallelization policies.",
        label="tab:training_dse_results",
    )
    tex_path.write_text(latex, encoding="utf-8")

    print(f"[OK] CSV : {csv_path}")
    print(f"[OK] TEX : {tex_path}")
    print(f"[OK] Metric column used: {metric_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())