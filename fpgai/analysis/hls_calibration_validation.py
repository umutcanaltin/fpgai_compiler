"""Held-out validation for FPGAI HLS calibration models.

This module evaluates whether the HLS calibration model improves estimates on
samples that were not used to fit the calibration factors.  It intentionally
keeps the data format close to the HLS calibration artifacts:

    {"samples": [{"estimated": {...}, "hls_actual": {...}, ...}]}

The public entry point is :func:`run_calibration_validation`.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from fpgai.analysis.hls_calibration_model import (
    RESOURCE_METRICS,
    apply_calibration_model,
    fit_calibration_model,
    mean_absolute_percentage_error,
)

ValidationRow = dict[str, Any]


def load_calibration_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Calibration dataset has no samples list: {dataset_path}")
    return data


def run_calibration_validation(
    dataset: dict[str, Any],
    *,
    modes: Iterable[str] = ("leave_one_sample_out", "leave_one_operator_out", "train_test_split"),
    test_fraction: float = 0.34,
    seed: int = 0,
) -> dict[str, Any]:
    """Run held-out validation modes on a calibration dataset.

    Supported modes:
      - ``leave_one_sample_out``: train on N-1 samples, test on one sample.
      - ``leave_one_operator_out``: train without each operator, test on that operator.
      - ``train_test_split``: deterministic shuffled split.
    """
    samples = _valid_samples(dataset.get("samples", []))
    if len(samples) < 2:
        raise ValueError("Need at least two usable calibration samples for held-out validation")

    selected_modes = list(modes)
    rows: list[ValidationRow] = []
    warnings: list[dict[str, Any]] = []

    for mode in selected_modes:
        if mode == "leave_one_sample_out":
            mode_rows = _leave_one_sample_out(samples)
        elif mode == "leave_one_operator_out":
            mode_rows = _leave_one_operator_out(samples)
        elif mode == "train_test_split":
            mode_rows = _train_test_split(samples, test_fraction=test_fraction, seed=seed)
        else:
            warnings.append({"mode": mode, "warning": "unknown_validation_mode_skipped"})
            continue
        rows.extend(mode_rows)

    summary = _summarize_rows(rows)
    return {
        "format": "fpgai.hls_calibration_validation.v1",
        "sample_count": len(samples),
        "modes": selected_modes,
        "metrics": list(RESOURCE_METRICS),
        "rows": rows,
        "summary": summary,
        "warnings": warnings,
    }


def write_validation_outputs(report: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    """Write JSON, CSV, TeX, Markdown, and optional PNG validation outputs."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "json": output / "calibration_validation.json",
        "csv": output / "calibration_validation.csv",
        "tex": output / "calibration_validation.tex",
        "markdown": output / "calibration_validation.md",
        "plot": output / "calibration_validation_plot.png",
    }

    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(report, paths["csv"])
    _write_tex(report, paths["tex"])
    _write_markdown(report, paths["markdown"])
    _write_plot(report, paths["plot"])

    return {key: str(value) for key, value in paths.items() if value.exists()}


def _valid_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for sample in samples:
        estimated = sample.get("estimated") or {}
        actual = sample.get("hls_actual") or {}
        if any(_to_float(actual.get(metric, 0.0)) > 0 for metric in RESOURCE_METRICS) and any(
            _to_float(estimated.get(metric, 0.0)) > 0 for metric in RESOURCE_METRICS
        ):
            valid.append(dict(sample))
    return valid


def _leave_one_sample_out(samples: list[dict[str, Any]]) -> list[ValidationRow]:
    rows: list[ValidationRow] = []
    for idx, heldout in enumerate(samples):
        train = [sample for j, sample in enumerate(samples) if j != idx]
        rows.extend(_fit_and_score(train, [heldout], mode="leave_one_sample_out", fold=f"sample_{idx}"))
    return rows


def _leave_one_operator_out(samples: list[dict[str, Any]]) -> list[ValidationRow]:
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_operator[str(sample.get("operator") or "Unknown")].append(sample)

    rows: list[ValidationRow] = []
    for operator in sorted(by_operator):
        test = by_operator[operator]
        train = [sample for sample in samples if str(sample.get("operator") or "Unknown") != operator]
        if not train:
            continue
        rows.extend(_fit_and_score(train, test, mode="leave_one_operator_out", fold=operator))
    return rows


def _train_test_split(samples: list[dict[str, Any]], *, test_fraction: float, seed: int) -> list[ValidationRow]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * test_fraction))))
    test = shuffled[:test_count]
    train = shuffled[test_count:]
    return _fit_and_score(train, test, mode="train_test_split", fold=f"seed_{seed}")


def _fit_and_score(
    train_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    *,
    mode: str,
    fold: str,
) -> list[ValidationRow]:
    model = fit_calibration_model({"samples": train_samples})
    calibrated_dataset = apply_calibration_model({"samples": test_samples}, model)
    heldout = calibrated_dataset.get("samples", [])

    rows: list[ValidationRow] = []
    operators = sorted({str(sample.get("operator") or "Unknown") for sample in test_samples})
    for metric in RESOURCE_METRICS:
        raw_mape = mean_absolute_percentage_error(heldout, metric, calibrated=False)
        calibrated_mape = mean_absolute_percentage_error(heldout, metric, calibrated=True)
        rows.append(
            {
                "mode": mode,
                "fold": fold,
                "metric": metric,
                "train_sample_count": len(train_samples),
                "test_sample_count": len(test_samples),
                "test_operators": ",".join(operators),
                "raw_mape": raw_mape,
                "calibrated_mape": calibrated_mape,
                "improvement_ratio": _improvement_ratio(raw_mape, calibrated_mape),
                "improved": calibrated_mape <= raw_mape,
            }
        )
    return rows


def _summarize_rows(rows: list[ValidationRow]) -> dict[str, Any]:
    summary: dict[str, Any] = {"by_mode_metric": {}, "overall": {}}
    grouped: dict[tuple[str, str], list[ValidationRow]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), str(row["metric"]))].append(row)

    for (mode, metric), group in sorted(grouped.items()):
        key = f"{mode}.{metric}"
        summary["by_mode_metric"][key] = _aggregate_group(group)

    if rows:
        summary["overall"] = _aggregate_group(rows)
    return summary


def _aggregate_group(group: list[ValidationRow]) -> dict[str, float | int]:
    raw_values = [_to_float(row.get("raw_mape")) for row in group]
    cal_values = [_to_float(row.get("calibrated_mape")) for row in group]
    raw_mean = _mean(raw_values)
    cal_mean = _mean(cal_values)
    return {
        "fold_count": len(group),
        "raw_mape": raw_mean,
        "calibrated_mape": cal_mean,
        "improvement_ratio": _improvement_ratio(raw_mean, cal_mean),
        "improved_fold_count": sum(1 for row in group if bool(row.get("improved"))),
    }


def _write_csv(report: dict[str, Any], path: Path) -> None:
    rows = report.get("rows", [])
    fieldnames = [
        "mode",
        "fold",
        "metric",
        "train_sample_count",
        "test_sample_count",
        "test_operators",
        "raw_mape",
        "calibrated_mape",
        "improvement_ratio",
        "improved",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_tex(report: dict[str, Any], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Mode & Metric & Raw MAPE & Calibrated MAPE & Improvement \\",
        r"\midrule",
    ]
    for key, values in sorted((report.get("summary", {}).get("by_mode_metric", {}) or {}).items()):
        mode, metric = key.rsplit(".", 1)
        lines.append(
            f"{_tex_escape(mode)} & {_tex_escape(metric)} & "
            f"{_to_float(values.get('raw_mape')):.2f} & "
            f"{_to_float(values.get('calibrated_mape')):.2f} & "
            f"{_to_float(values.get('improvement_ratio')):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# HLS calibration held-out validation",
        "",
        f"Samples: {report.get('sample_count', 0)}",
        "",
        "| Mode | Metric | Raw MAPE | Calibrated MAPE | Improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, values in sorted((report.get("summary", {}).get("by_mode_metric", {}) or {}).items()):
        mode, metric = key.rsplit(".", 1)
        lines.append(
            f"| {mode} | {metric} | {_to_float(values.get('raw_mape')):.2f} | "
            f"{_to_float(values.get('calibrated_mape')):.2f} | "
            f"{_to_float(values.get('improvement_ratio')):.2f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(report: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    items = list((report.get("summary", {}).get("by_mode_metric", {}) or {}).items())
    if not items:
        return
    labels = [key.replace("leave_one_", "loo_") for key, _ in items]
    raw = [_to_float(values.get("raw_mape")) for _, values in items]
    calibrated = [_to_float(values.get("calibrated_mape")) for _, values in items]

    x = list(range(len(labels)))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 4.5))
    ax.bar([v - width / 2 for v in x], raw, width, label="raw")
    ax.bar([v + width / 2 for v in x], calibrated, width, label="calibrated")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Held-out HLS calibration validation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _improvement_ratio(raw_mape: float, calibrated_mape: float) -> float:
    raw = _to_float(raw_mape)
    calibrated = _to_float(calibrated_mape)
    if calibrated == 0.0:
        return math.inf if raw > 0 else 1.0
    return raw / calibrated


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return 0.0


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_")
