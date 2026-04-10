#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ------------------------------------------------------------
# Column alias helpers
# ------------------------------------------------------------

ALIASES = {
    "model": [
        "model",
        "model_name",
        "network",
        "net",
    ],
    "mode": [
        "mode",
        "execution_mode",
        "run_mode",
        "flow",
    ],
    "precision": [
        "precision",
        "precision_policy",
        "quant_policy",
        "numeric_policy",
    ],
    "policy": [
        "policy",
        "parallel_policy",
        "schedule_policy",
        "parallelization_policy",
    ],
    "compile_ok": [
        "compile_ok",
        "hls_ok",
        "ok",
    ],
    # Actual latency
    "latency_actual_cycles": [
        "latency_cycles",
        "latency_cycles_max",
        "latency_cycles_min",
        "actual_latency_cycles",
        "csynth_latency_cycles",
        "measured_latency_cycles",
        "real_latency_cycles",
    ],
    # Estimated / predicted latency
    "latency_pred_cycles": [
        "pred_latency_cycles",
        "predicted_latency_cycles",
        "estimated_latency_cycles",
        "est_latency_cycles",
        "latency_est_cycles",
    ],
    # Actual resources
    "lut_actual": [
        "lut",
        "actual_lut",
        "measured_lut",
        "real_lut",
    ],
    "ff_actual": [
        "ff",
        "actual_ff",
        "measured_ff",
        "real_ff",
    ],
    "dsp_actual": [
        "dsp",
        "actual_dsp",
        "measured_dsp",
        "real_dsp",
    ],
    "bram_actual": [
        "bram",
        "bram_18k",
        "actual_bram",
        "measured_bram",
        "real_bram",
    ],
    # Predicted resources
    "lut_pred": [
        "pred_lut",
        "predicted_lut",
        "estimated_lut",
        "est_lut",
        "lut_est",
    ],
    "ff_pred": [
        "pred_ff",
        "predicted_ff",
        "estimated_ff",
        "est_ff",
        "ff_est",
    ],
    "dsp_pred": [
        "pred_dsp",
        "predicted_dsp",
        "estimated_dsp",
        "est_dsp",
        "dsp_est",
    ],
    "bram_pred": [
        "pred_bram",
        "predicted_bram",
        "estimated_bram",
        "est_bram",
        "bram_est",
        "pred_bram_18k",
        "estimated_bram_18k",
    ],
}


def find_col(df: pd.DataFrame, logical_name: str) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for alias in ALIASES.get(logical_name, []):
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
    return None


def require_col(df: pd.DataFrame, logical_name: str) -> str:
    col = find_col(df, logical_name)
    if col is None:
        raise KeyError(
            f"Could not find a column for '{logical_name}'. "
            f"Available columns: {list(df.columns)}"
        )
    return col


# ------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------

def latex_escape(s: object) -> str:
    if pd.isna(s):
        return "-"
    s = str(s)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def fmt_num(v: object, decimals: int = 0) -> str:
    if v is None or pd.isna(v):
        return "-"
    try:
        v = float(v)
    except Exception:
        return latex_escape(v)
    if decimals == 0:
        return str(int(round(v)))
    return f"{v:.{decimals}f}"


def pct_err(pred: object, actual: object) -> Optional[float]:
    if pred is None or actual is None or pd.isna(pred) or pd.isna(actual):
        return None
    try:
        pred = float(pred)
        actual = float(actual)
    except Exception:
        return None
    if actual == 0:
        return None
    return abs(pred - actual) / abs(actual) * 100.0


def choose_latency_actual_value(row: pd.Series, actual_col: str) -> object:
    return row.get(actual_col, None)


def best_model_name(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    s2 = s.lower()
    if "mlp" in s2:
        return "MLP-MNIST"
    if "cnn" in s2:
        return "CNN-MNIST"
    return s


# ------------------------------------------------------------
# Mode inference
# ------------------------------------------------------------

def infer_mode_series(df: pd.DataFrame) -> pd.Series:
    mode_col = find_col(df, "mode")
    if mode_col is not None:
        return df[mode_col].astype(str).str.lower()

    # fallback heuristic from model/path/name fields
    model_col = find_col(df, "model")
    if model_col is not None:
        s = df[model_col].astype(str).str.lower()
        inferred = []
        for x in s:
            if "train" in x or "training" in x:
                inferred.append("training")
            else:
                inferred.append("inference")
        return pd.Series(inferred, index=df.index)

    return pd.Series(["inference"] * len(df), index=df.index)


# ------------------------------------------------------------
# Row selection
# ------------------------------------------------------------

def sort_key_tuple(row: pd.Series, compile_ok_col: Optional[str]) -> Tuple:
    ok = 1
    if compile_ok_col and compile_ok_col in row.index:
        val = str(row[compile_ok_col]).strip().lower()
        ok = 1 if val in {"true", "1", "yes"} else 0
    model = str(row.get("__model_display__", ""))
    policy = str(row.get("__policy_display__", ""))
    precision = str(row.get("__precision_display__", ""))
    return (model, -ok, policy, precision)


def build_summary_rows(df: pd.DataFrame, mode_name: str) -> pd.DataFrame:
    model_col = require_col(df, "model")
    precision_col = require_col(df, "precision")
    policy_col = require_col(df, "policy")

    compile_ok_col = find_col(df, "compile_ok")

    latency_actual_col = require_col(df, "latency_actual_cycles")
    latency_pred_col = require_col(df, "latency_pred_cycles")

    lut_actual_col = require_col(df, "lut_actual")
    lut_pred_col = require_col(df, "lut_pred")
    dsp_actual_col = require_col(df, "dsp_actual")
    dsp_pred_col = require_col(df, "dsp_pred")
    bram_actual_col = require_col(df, "bram_actual")
    bram_pred_col = require_col(df, "bram_pred")

    out_rows: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        model = best_model_name(str(row[model_col]))
        precision = str(row[precision_col])
        policy = str(row[policy_col])

        lat_act = row.get(latency_actual_col, None)
        lat_pred = row.get(latency_pred_col, None)

        lut_act = row.get(lut_actual_col, None)
        lut_pred = row.get(lut_pred_col, None)

        dsp_act = row.get(dsp_actual_col, None)
        dsp_pred = row.get(dsp_pred_col, None)

        bram_act = row.get(bram_actual_col, None)
        bram_pred = row.get(bram_pred_col, None)

        out_rows.append(
            {
                "Model": model,
                "Policy": policy,
                "Precision": precision,
                "Pred. Lat.": lat_pred,
                "Actual Lat.": lat_act,
                "Lat. Err %": pct_err(lat_pred, lat_act),
                "Pred. LUT": lut_pred,
                "Actual LUT": lut_act,
                "LUT Err %": pct_err(lut_pred, lut_act),
                "Pred. DSP": dsp_pred,
                "Actual DSP": dsp_act,
                "DSP Err %": pct_err(dsp_pred, dsp_act),
                "Pred. BRAM": bram_pred,
                "Actual BRAM": bram_act,
                "BRAM Err %": pct_err(bram_pred, bram_act),
                "__model_display__": model,
                "__policy_display__": policy,
                "__precision_display__": precision,
                "__compile_ok__": row.get(compile_ok_col, None) if compile_ok_col else None,
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_df = out_df.sort_values(
        by=["Model", "Policy", "Precision"],
        kind="stable"
    ).reset_index(drop=True)
    return out_df


# ------------------------------------------------------------
# LaTeX emitters
# ------------------------------------------------------------

def dataframe_to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    font_cmd: str = r"\scriptsize",
    table_env: str = "table*",
) -> str:
    visible_cols = [
        "Model",
        "Policy",
        "Precision",
        "Pred. Lat.",
        "Actual Lat.",
        "Lat. Err %",
        "Pred. LUT",
        "Actual LUT",
        "LUT Err %",
        "Pred. DSP",
        "Actual DSP",
        "DSP Err %",
        "Pred. BRAM",
        "Actual BRAM",
        "BRAM Err %",
    ]

    lines = []
    lines.append(rf"\begin{{{table_env}}}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(font_cmd)
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\begin{tabular}{lllccccccccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Model & Policy & Precision & "
        r"Pred. Lat. & Actual Lat. & Lat. Err. (\%) & "
        r"Pred. LUT & Actual LUT & LUT Err. (\%) & "
        r"Pred. DSP & Actual DSP & DSP Err. (\%) & "
        r"Pred. BRAM & Actual BRAM & BRAM Err. (\%) \\"
    )
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        vals = [
            latex_escape(row["Model"]),
            latex_escape(row["Policy"]),
            latex_escape(row["Precision"]),
            fmt_num(row["Pred. Lat."]),
            fmt_num(row["Actual Lat."]),
            fmt_num(row["Lat. Err %"], 2),
            fmt_num(row["Pred. LUT"]),
            fmt_num(row["Actual LUT"]),
            fmt_num(row["LUT Err %"], 2),
            fmt_num(row["Pred. DSP"]),
            fmt_num(row["Actual DSP"]),
            fmt_num(row["DSP Err %"], 2),
            fmt_num(row["Pred. BRAM"]),
            fmt_num(row["Actual BRAM"]),
            fmt_num(row["BRAM Err %"], 2),
        ]
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\end{{{table_env}}}")
    return "\n".join(lines)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX estimator-vs-real comparison tables for inference and training."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV containing predicted and actual resource/latency columns.",
    )
    parser.add_argument(
        "--out-dir",
        default="paper_tables",
        help="Directory to write output .tex files.",
    )
    parser.add_argument(
        "--inference-filter",
        default="inference",
        help="Substring used to identify inference rows in mode column.",
    )
    parser.add_argument(
        "--training-filter",
        default="training",
        help="Substring used to identify training rows in mode column.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    mode_series = infer_mode_series(df)

    inf_mask = mode_series.str.contains(args.inference_filter.lower(), na=False)
    trn_mask = mode_series.str.contains(args.training_filter.lower(), na=False)

    inf_df_raw = df[inf_mask].copy()
    trn_df_raw = df[trn_mask].copy()

    if len(inf_df_raw) == 0:
        print("Warning: no inference rows found.", file=sys.stderr)
    if len(trn_df_raw) == 0:
        print("Warning: no training rows found.", file=sys.stderr)

    if len(inf_df_raw) > 0:
        inf_summary = build_summary_rows(inf_df_raw, "inference")
        inf_tex = dataframe_to_latex_table(
            inf_summary,
            caption="Inference estimator vs. post-synthesis results on KV260.",
            label="tab:inference_estimator_vs_real",
            font_cmd=r"\scriptsize",
            table_env="table*",
        )
        (out_dir / "inference_estimator_vs_real.tex").write_text(inf_tex, encoding="utf-8")
        print(f"Wrote: {out_dir / 'inference_estimator_vs_real.tex'}")

    if len(trn_df_raw) > 0:
        trn_summary = build_summary_rows(trn_df_raw, "training")
        trn_tex = dataframe_to_latex_table(
            trn_summary,
            caption="Training estimator vs. post-synthesis results on KV260.",
            label="tab:training_estimator_vs_real",
            font_cmd=r"\scriptsize",
            table_env="table*",
        )
        (out_dir / "training_estimator_vs_real.tex").write_text(trn_tex, encoding="utf-8")
        print(f"Wrote: {out_dir / 'training_estimator_vs_real.tex'}")

    # Also write CSV summaries for inspection
    if len(inf_df_raw) > 0:
        inf_summary.to_csv(out_dir / "inference_estimator_vs_real.csv", index=False)
    if len(trn_df_raw) > 0:
        trn_summary.to_csv(out_dir / "training_estimator_vs_real.csv", index=False)


if __name__ == "__main__":
    main()