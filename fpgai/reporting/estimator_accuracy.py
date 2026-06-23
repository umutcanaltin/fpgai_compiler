#!/usr/bin/env python3
"""
Collect resource-estimator accuracy reports safely.

Key safety rule:
  Do NOT treat placeholder values such as 0 or 1 as valid resource predictions.

This collector compares Vivado implementation resources against FPGAI estimator
artifacts only when non-placeholder predictions are available. If no usable
predictions exist, it produces an honest report saying so instead of
creating an artificial 100% error table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

RESOURCE_KEYS = {
    "lut": ("lut", "luts", "LUT", "LUTs", "logic_lut", "logic_luts", "estimated_lut", "pred_lut", "predicted_lut"),
    "ff": ("ff", "ffs", "FF", "FFs", "flipflop", "flip_flop", "estimated_ff", "pred_ff", "predicted_ff"),
    "bram": ("bram", "BRAM", "brams", "BRAMs", "bram18", "bram_18k", "estimated_bram", "pred_bram", "predicted_bram"),
    "dsp": ("dsp", "DSP", "dsps", "DSPs", "dsp48", "dsp48e", "estimated_dsp", "pred_dsp", "predicted_dsp"),
}

# Artifacts with these names often contain calibrated coefficients or defaults,
# not full-design resource predictions. They are still listed as candidates, but
# not trusted unless they contain explicit non-placeholder estimate fields.
LOW_CONFIDENCE_FILENAMES = {
    "calibrated_model.json",
    "hls_operator_dataset.json",
}

PLACEHOLDER_VALUES = {0.0, 1.0}


@dataclass
class PredValue:
    value: float | None
    source: str = ""
    key_path: str = ""
    reason: str = ""


def parse_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        if math.isfinite(float(x)):
            return float(x)
        return None
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if not s:
            return None
        try:
            v = float(s)
            if math.isfinite(v):
                return v
        except ValueError:
            return None
    return None


def is_placeholder_prediction(v: float | None) -> bool:
    if v is None:
        return True
    if v in PLACEHOLDER_VALUES:
        return True
    if v < 0:
        return True
    return False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def flatten_json(obj: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten_json(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield from flatten_json(v, p)
    else:
        yield prefix, obj


def looks_like_estimator_file(path: Path) -> bool:
    name = path.name.lower()
    parts = "/".join(path.parts).lower()
    if path.suffix.lower() != ".json":
        return False
    tokens = ["estimate", "estimator", "resource", "tiling", "calibration", "compile_plan", "training_estimate"]
    return any(t in name or t in parts for t in tokens)


def confidence_for_path(path: Path, key_path: str) -> int:
    """Higher is better. Avoid calibrated coefficient/default artifacts."""
    name = path.name
    kp = key_path.lower()
    joined = "/".join(path.parts).lower()

    score = 0
    if "resource" in name.lower() or "resource" in joined:
        score += 4
    if "estimate" in name.lower() or "estimate" in joined:
        score += 4
    if "tiling_resource_estimate" in joined:
        score += 5
    if "training_estimate" in joined:
        score += 3
    if "pred" in kp or "estimate" in kp or "resource" in kp:
        score += 3
    if name in LOW_CONFIDENCE_FILENAMES:
        score -= 10
    if kp.startswith("global."):
        score -= 8
    return score


def find_predictions_for_design(artifact_dir: Path) -> tuple[dict[str, PredValue], list[str]]:
    candidates = [p for p in artifact_dir.rglob("*.json") if looks_like_estimator_file(p)]
    candidate_lines = [str(p) for p in sorted(candidates)]

    best: dict[str, tuple[int, PredValue]] = {}

    for path in candidates:
        obj = load_json(path)
        if obj is None:
            continue
        for key_path, value in flatten_json(obj):
            lower_key = key_path.lower()
            num = parse_float(value)
            if num is None:
                continue
            for resource, aliases in RESOURCE_KEYS.items():
                alias_match = False
                for alias in aliases:
                    # Match complete key components, but also allow common forms
                    a = alias.lower()
                    if re.search(rf"(^|[._\[\]/-]){re.escape(a)}($|[._\[\]/-])", lower_key):
                        alias_match = True
                        break
                    if lower_key.endswith("." + a) or lower_key.endswith("_" + a):
                        alias_match = True
                        break
                if not alias_match:
                    continue

                score = confidence_for_path(path, key_path)
                pv = PredValue(value=num, source=str(path), key_path=key_path)
                old = best.get(resource)
                if old is None or score > old[0]:
                    best[resource] = (score, pv)

    result: dict[str, PredValue] = {}
    for resource in RESOURCE_KEYS:
        if resource not in best:
            result[resource] = PredValue(None, reason="missing")
            continue
        score, pv = best[resource]
        if is_placeholder_prediction(pv.value):
            result[resource] = PredValue(
                None,
                source=pv.source,
                key_path=pv.key_path,
                reason=f"placeholder_prediction_{pv.value:g}",
            )
        elif score < 0:
            result[resource] = PredValue(
                None,
                source=pv.source,
                key_path=pv.key_path,
                reason=f"low_confidence_source_score_{score}",
            )
        else:
            result[resource] = pv
    return result, candidate_lines


def pct_error(pred: float | None, actual: float | None) -> float | None:
    if pred is None or actual is None or actual == 0:
        return None
    return abs(pred - actual) / abs(actual) * 100.0


def fnum(x: Any) -> float | None:
    return parse_float(x)


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", help="Experiment folders to scan")
    ap.add_argument("--vivado-summary", required=True, help="Vivado implementation summary CSV")
    ap.add_argument("--out", default="reports/estimator_accuracy")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vivado_rows = read_csv(Path(args.vivado_summary))
    rows: list[dict[str, Any]] = []
    candidate_files: list[str] = []
    all_errors: dict[str, list[float]] = {"lut": [], "ff": [], "bram": [], "dsp": []}

    for vr in vivado_rows:
        design = vr.get("design", "")
        artifact_dir = Path(vr.get("artifact_dir", ""))
        if not artifact_dir.exists():
            notes = "artifact_dir_missing"
            preds = {r: PredValue(None, reason="artifact_dir_missing") for r in RESOURCE_KEYS}
            cand = []
        else:
            preds, cand = find_predictions_for_design(artifact_dir)
            candidate_files.extend(cand)
            notes = ""

        row: dict[str, Any] = {
            "design": design,
            "status": vr.get("status", ""),
            "estimator_source": "",
            "notes": notes,
        }

        sources = []
        usable_count = 0
        note_bits = []
        for res in ["lut", "ff", "bram", "dsp"]:
            pred = preds[res].value
            actual = fnum(vr.get(res) or vr.get(res.upper()))
            err = pct_error(pred, actual)
            row[f"pred_{res}"] = pred
            row[f"vivado_{res}"] = actual
            row[f"{res}_abs_error_pct"] = err
            row[f"{res}_prediction_source"] = preds[res].source
            row[f"{res}_prediction_key"] = preds[res].key_path
            row[f"{res}_prediction_reason"] = preds[res].reason
            if pred is not None and actual is not None and err is not None:
                usable_count += 1
                all_errors[res].append(err)
                sources.append(f"{res}:{Path(preds[res].source).name}:{preds[res].key_path}")
            elif preds[res].reason:
                note_bits.append(f"{res}:{preds[res].reason}")

        row["usable_prediction_count"] = usable_count
        row["estimator_source"] = ";".join(sources)
        row["notes"] = ";".join([x for x in [notes, *note_bits] if x])
        rows.append(row)

    fields = [
        "design", "status", "usable_prediction_count",
        "pred_lut", "vivado_lut", "lut_abs_error_pct",
        "pred_ff", "vivado_ff", "ff_abs_error_pct",
        "pred_bram", "vivado_bram", "bram_abs_error_pct",
        "pred_dsp", "vivado_dsp", "dsp_abs_error_pct",
        "estimator_source", "notes",
        "lut_prediction_source", "lut_prediction_key", "lut_prediction_reason",
        "ff_prediction_source", "ff_prediction_key", "ff_prediction_reason",
        "bram_prediction_source", "bram_prediction_key", "bram_prediction_reason",
        "dsp_prediction_source", "dsp_prediction_key", "dsp_prediction_reason",
    ]

    write_csv(out / "estimator_accuracy.csv", rows, fields)
    (out / "estimator_accuracy.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "estimator_candidate_files.txt").write_text("\n".join(sorted(set(candidate_files))) + "\n", encoding="utf-8")

    flat_errors = [e for vals in all_errors.values() for e in vals]
    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None
    def median(xs: list[float]) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        n = len(s)
        return s[n//2] if n % 2 else (s[n//2 - 1] + s[n//2]) / 2

    summary = {
        "comparison_rows": len(rows),
        "rows_with_any_usable_prediction": sum(1 for r in rows if int(r["usable_prediction_count"]) > 0),
        "skipped_without_usable_prediction": sum(1 for r in rows if int(r["usable_prediction_count"]) == 0),
        "overall_count": len(flat_errors),
        "overall_mean_abs_percentage_error": mean(flat_errors),
        "overall_median_abs_percentage_error": median(flat_errors),
        "overall_worst_abs_percentage_error": max(flat_errors) if flat_errors else None,
        "by_resource": {
            r: {
                "count": len(vals),
                "mean_abs_percentage_error": mean(vals),
                "median_abs_percentage_error": median(vals),
                "worst_abs_percentage_error": max(vals) if vals else None,
            }
            for r, vals in all_errors.items()
        },
    }
    (out / "estimator_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# Resource Estimator Accuracy\n")
    md.append("This table compares FPGAI resource predictions against Vivado implementation resources only when usable non-placeholder prediction artifacts are found. Values `0` and `1` are treated as placeholders/missing.\n")
    md.append("## Summary\n")
    md.append(f"- comparison rows: {summary['comparison_rows']}")
    md.append(f"- rows with any usable prediction: {summary['rows_with_any_usable_prediction']}")
    md.append(f"- skipped rows without usable prediction: {summary['skipped_without_usable_prediction']}")
    md.append(f"- overall comparison count: {summary['overall_count']}")
    if flat_errors:
        md.append(f"- overall mean absolute percentage error: {fmt(summary['overall_mean_abs_percentage_error'])}%")
        md.append(f"- overall median absolute percentage error: {fmt(summary['overall_median_abs_percentage_error'])}%")
        md.append(f"- overall worst absolute percentage error: {fmt(summary['overall_worst_abs_percentage_error'])}%")
    else:
        md.append("- no usable predicted-vs-Vivado comparisons were found")
    md.append("")
    for r in ["lut", "ff", "bram", "dsp"]:
        s = summary["by_resource"][r]
        md.append(f"- {r.upper()}: count={s['count']}, mean_abs_pct_error={fmt(s['mean_abs_percentage_error'])}, median_abs_pct_error={fmt(s['median_abs_percentage_error'])}, worst_abs_pct_error={fmt(s['worst_abs_percentage_error'])}")
    md.append("\n## Estimator accuracy table\n")
    md.append("| design | status | usable_predictions | pred_lut | vivado_lut | lut_abs_error_pct | pred_ff | vivado_ff | ff_abs_error_pct | pred_bram | vivado_bram | bram_abs_error_pct | pred_dsp | vivado_dsp | dsp_abs_error_pct | notes |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        md.append(
            "| " + " | ".join([
                str(r.get("design", "")), str(r.get("status", "")), fmt(r.get("usable_prediction_count")),
                fmt(r.get("pred_lut")), fmt(r.get("vivado_lut")), fmt(r.get("lut_abs_error_pct")),
                fmt(r.get("pred_ff")), fmt(r.get("vivado_ff")), fmt(r.get("ff_abs_error_pct")),
                fmt(r.get("pred_bram")), fmt(r.get("vivado_bram")), fmt(r.get("bram_abs_error_pct")),
                fmt(r.get("pred_dsp")), fmt(r.get("vivado_dsp")), fmt(r.get("dsp_abs_error_pct")),
                str(r.get("notes", "")).replace("|", "/"),
            ]) + " |"
        )
    md.append("\n## Safe claim\n")
    if flat_errors:
        md.append("FPGAI estimator outputs are compared against Vivado implementation resources where usable non-placeholder prediction artifacts are available.")
    else:
        md.append("No usable non-placeholder full-design resource estimator artifacts were found in the existing experiment folders; estimator accuracy remains an open evidence gap for the arXiv version unless a new estimator export is added.")
    md.append("\n## Limitation\n")
    md.append("This collector summarizes existing estimator artifacts only. It does not rerun estimation. Placeholder values such as 0 or 1 are treated as missing to avoid artificial error tables. Candidate files are listed in `estimator_candidate_files.txt`.")
    (out / "estimator_accuracy.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {out / 'estimator_accuracy.csv'}")
    print(f"Wrote {out / 'estimator_accuracy.json'}")
    print(f"Wrote {out / 'estimator_summary.json'}")
    print(f"Wrote {out / 'estimator_accuracy.md'}")
    print(f"Wrote {out / 'estimator_candidate_files.txt'}")
    print(
        f"rows={summary['comparison_rows']} "
        f"rows_with_any_usable_prediction={summary['rows_with_any_usable_prediction']} "
        f"skipped_without_usable_prediction={summary['skipped_without_usable_prediction']} "
        f"overall_count={summary['overall_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
