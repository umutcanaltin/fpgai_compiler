#!/usr/bin/env python3
"""Summarize HLS calibration validation results into paper-safe claims.

This script reads calibration_validation.json from Sprint 10 and produces a
small JSON/Markdown summary that distinguishes:
  - same-operator held-out validation (leave-one-sample/train-test split), and
  - unseen-operator validation (leave-one-operator-out).

The goal is to prevent overclaiming when overall improvement is strong but
leave-one-operator-out still degrades for some metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


PREFERRED_METRIC_ORDER = ["lut", "ff", "dsp", "bram", "latency_cycles"]
SAME_OPERATOR_MODES = {"leave_one_sample_out", "train_test_split"}
UNSEEN_OPERATOR_MODES = {"leave_one_operator_out"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _ratio(raw_mape: float, calibrated_mape: float) -> float:
    if calibrated_mape == 0.0:
        if raw_mape == 0.0:
            return 1.0
        return math.inf
    return raw_mape / calibrated_mape


def _format_ratio(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}x"


def load_validation(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        mode = str(row.get("mode", ""))
        metric = str(row.get("metric", ""))
        if mode and metric:
            grouped[mode][metric].append(row)

    mode_summaries: Dict[str, Dict[str, Any]] = {}
    for mode, metrics in grouped.items():
        metric_summaries: Dict[str, Dict[str, Any]] = {}
        raw_values: List[float] = []
        calibrated_values: List[float] = []
        improved_metrics = 0
        total_metrics = 0

        ordered_metrics = [m for m in PREFERRED_METRIC_ORDER if m in metrics] + [
            m for m in sorted(metrics) if m not in PREFERRED_METRIC_ORDER
        ]
        for metric in ordered_metrics:
            metric_rows = metrics[metric]
            raw = sum(_safe_float(r.get("raw_mape")) for r in metric_rows) / max(len(metric_rows), 1)
            calibrated = sum(_safe_float(r.get("calibrated_mape")) for r in metric_rows) / max(len(metric_rows), 1)
            improvement = _ratio(raw, calibrated)
            improved = improvement >= 1.0
            metric_summaries[metric] = {
                "raw_mape": raw,
                "calibrated_mape": calibrated,
                "improvement_ratio": improvement,
                "improved": improved,
                "fold_count": len(metric_rows),
            }
            raw_values.append(raw)
            calibrated_values.append(calibrated)
            improved_metrics += 1 if improved else 0
            total_metrics += 1

        raw_mean = sum(raw_values) / max(len(raw_values), 1)
        calibrated_mean = sum(calibrated_values) / max(len(calibrated_values), 1)
        mode_summaries[mode] = {
            "metrics": metric_summaries,
            "raw_mape_mean": raw_mean,
            "calibrated_mape_mean": calibrated_mean,
            "improvement_ratio_mean": _ratio(raw_mean, calibrated_mean),
            "improved_metric_count": improved_metrics,
            "metric_count": total_metrics,
            "all_metrics_improved": improved_metrics == total_metrics and total_metrics > 0,
        }

    same_modes = [m for m in SAME_OPERATOR_MODES if m in mode_summaries]
    unseen_modes = [m for m in UNSEEN_OPERATOR_MODES if m in mode_summaries]

    same_operator_all_improved = bool(same_modes) and all(
        mode_summaries[m]["all_metrics_improved"] for m in same_modes
    )
    unseen_operator_all_improved = bool(unseen_modes) and all(
        mode_summaries[m]["all_metrics_improved"] for m in unseen_modes
    )

    degraded_unseen: List[Dict[str, Any]] = []
    for mode in unseen_modes:
        for metric, metric_summary in mode_summaries[mode]["metrics"].items():
            if not metric_summary["improved"]:
                degraded_unseen.append(
                    {
                        "mode": mode,
                        "metric": metric,
                        "raw_mape": metric_summary["raw_mape"],
                        "calibrated_mape": metric_summary["calibrated_mape"],
                        "improvement_ratio": metric_summary["improvement_ratio"],
                    }
                )

    if same_operator_all_improved and unseen_operator_all_improved:
        claim_level = "strong_generalization"
        recommended_claim = (
            "The calibrated estimator reduces held-out HLS estimation error across same-operator "
            "and unseen-operator validation modes."
        )
    elif same_operator_all_improved:
        claim_level = "same_operator_only"
        recommended_claim = (
            "The calibrated estimator substantially improves same-operator held-out validation, "
            "while unseen-operator generalization remains mixed and should be reported separately."
        )
    else:
        claim_level = "pipeline_only"
        recommended_claim = (
            "FPGAI provides a held-out HLS calibration validation pipeline, but the current "
            "calibration set does not yet support a broad accuracy-improvement claim."
        )

    return {
        "format": "fpgai.hls_calibration_claim_summary.v1",
        "mode_summaries": mode_summaries,
        "same_operator_all_metrics_improved": same_operator_all_improved,
        "unseen_operator_all_metrics_improved": unseen_operator_all_improved,
        "degraded_unseen_operator_metrics": degraded_unseen,
        "claim_level": claim_level,
        "recommended_claim": recommended_claim,
    }


def write_markdown(summary: Mapping[str, Any], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Paper-safe HLS calibration claim summary")
    lines.append("")
    lines.append(f"Claim level: `{summary['claim_level']}`")
    lines.append("")
    lines.append("Recommended claim:")
    lines.append("")
    lines.append(f"> {summary['recommended_claim']}")
    lines.append("")
    lines.append("## Mode summary")
    lines.append("")
    lines.append("| Mode | Mean raw MAPE | Mean calibrated MAPE | Improvement | Improved metrics |")
    lines.append("|---|---:|---:|---:|---:|")
    for mode, mode_summary in sorted(summary["mode_summaries"].items()):
        lines.append(
            "| {mode} | {raw:.2f} | {cal:.2f} | {ratio} | {improved}/{total} |".format(
                mode=mode,
                raw=mode_summary["raw_mape_mean"],
                cal=mode_summary["calibrated_mape_mean"],
                ratio=_format_ratio(mode_summary["improvement_ratio_mean"]),
                improved=mode_summary["improved_metric_count"],
                total=mode_summary["metric_count"],
            )
        )
    lines.append("")
    lines.append("## Per-metric details")
    lines.append("")
    lines.append("| Mode | Metric | Raw MAPE | Calibrated MAPE | Improvement | Status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for mode, mode_summary in sorted(summary["mode_summaries"].items()):
        metrics = mode_summary["metrics"]
        ordered_metrics = [m for m in PREFERRED_METRIC_ORDER if m in metrics] + [
            m for m in sorted(metrics) if m not in PREFERRED_METRIC_ORDER
        ]
        for metric in ordered_metrics:
            metric_summary = metrics[metric]
            status = "improved" if metric_summary["improved"] else "degraded"
            lines.append(
                "| {mode} | {metric} | {raw:.2f} | {cal:.2f} | {ratio} | {status} |".format(
                    mode=mode,
                    metric=metric,
                    raw=metric_summary["raw_mape"],
                    cal=metric_summary["calibrated_mape"],
                    ratio=_format_ratio(metric_summary["improvement_ratio"]),
                    status=status,
                )
            )
    if summary["degraded_unseen_operator_metrics"]:
        lines.append("")
        lines.append("## Caution")
        lines.append("")
        lines.append(
            "Unseen-operator validation has degraded metrics. Do not claim broad unseen-operator "
            "generalization without reporting these metrics."
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True, type=Path, help="Path to calibration_validation.json")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    validation = load_validation(args.validation)
    rows = validation.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"No validation rows found in {args.validation}")

    summary = summarize_rows(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "paper_claim_summary.json"
    md_path = args.out / "paper_claim_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(summary, md_path)

    print(f"[OK] Wrote HLS calibration claim summary: {args.out}")
    print(f"Claim level: {summary['claim_level']}")
    print(f"Recommended claim: {summary['recommended_claim']}")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
