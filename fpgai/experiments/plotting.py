"""Plot generation for FPGAI experiment analyses."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from .analysis import ExperimentRecord


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("matplotlib is required for plot generation") from exc
    return plt


def _short_labels(records: Sequence[ExperimentRecord]) -> List[str]:
    return [r.design_name.replace("precision_", "").replace("policy_", "") for r in records]


def plot_duration_by_design(records: Sequence[ExperimentRecord], out_path: str | Path) -> Path:
    plt = _require_matplotlib()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = _short_labels(records)
    values = [r.duration_sec or 0.0 for r in records]
    fig, ax = plt.subplots(figsize=(max(6, len(records) * 1.2), 4))
    ax.bar(range(len(values)), values)
    ax.set_ylabel("Duration (s)")
    ax.set_xlabel("Design")
    ax.set_title("Benchmark duration by design")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_raw_vs_calibrated_mape(records: Sequence[ExperimentRecord], out_path: str | Path) -> Path:
    plt = _require_matplotlib()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metric_pairs = [
        ("lut", "LUT"),
        ("ff", "FF"),
        ("dsp", "DSP"),
        ("bram", "BRAM"),
        ("latency_cycles", "Latency"),
    ]
    raw_values = []
    cal_values = []
    labels = []
    for suffix, label in metric_pairs:
        raw = [r.metrics.get(f"raw_mape.{suffix}") for r in records if f"raw_mape.{suffix}" in r.metrics]
        cal = [r.metrics.get(f"cal_mape.{suffix}") for r in records if f"cal_mape.{suffix}" in r.metrics]
        if raw or cal:
            labels.append(label)
            raw_values.append(sum(raw) / len(raw) if raw else 0.0)
            cal_values.append(sum(cal) / len(cal) if cal else 0.0)
    x = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.0), 4))
    ax.bar([i - width / 2 for i in x], raw_values, width, label="Raw")
    ax.bar([i + width / 2 for i in x], cal_values, width, label="Calibrated")
    ax.set_ylabel("Mean absolute percentage error (%)")
    ax.set_xlabel("Metric")
    ax.set_title("Raw vs calibrated HLS estimate error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def write_plots(records: Sequence[ExperimentRecord], out_dir: str | Path) -> List[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        paths.append(plot_duration_by_design(records, out / "duration_by_design.png"))
        paths.append(plot_raw_vs_calibrated_mape(records, out / "raw_vs_calibrated_mape.png"))
    except RuntimeError:
        # Keep analysis script usable in environments without matplotlib.
        (out / "PLOTS_SKIPPED.txt").write_text(
            "matplotlib is not installed, so PNG plot generation was skipped.\n",
            encoding="utf-8",
        )
    return paths
