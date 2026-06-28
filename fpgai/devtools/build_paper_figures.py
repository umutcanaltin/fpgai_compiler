"""Build paper-ready figures from parsed FPGAI numeric artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _num(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", "None", "NA"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _short_name(design: str) -> str:
    name = design
    for prefix in ("kv260_", "pynq_z2_", "kr260_", "training_kv260_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", "\n")


def _save_bar(path: Path, labels: list[str], values: list[float], ylabel: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 0.45), 4.2))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_grouped_bar(
    path: Path,
    labels: list[str],
    left: list[float],
    right: list[float],
    left_label: str,
    right_label: str,
    ylabel: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 0.5), 4.4))
    x = list(range(len(labels)))
    width = 0.38
    ax.bar([v - width / 2 for v in x], left, width, label=left_label)
    ax.bar([v + width / 2 for v in x], right, width, label=right_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _prediction_vs_hls(rows: list[dict[str, str]], out: Path, resource: str) -> None:
    pred_key = f"prediction_{resource}"
    hls_key = f"hls_{resource}"
    labels: list[str] = []
    pred: list[float] = []
    hls: list[float] = []

    for r in rows:
        p = _num(r, pred_key)
        h = _num(r, hls_key)
        if p is None or h is None:
            continue
        labels.append(_short_name(r["design"]))
        pred.append(p)
        hls.append(h)

    _save_grouped_bar(
        out / f"prediction_vs_hls_{resource}.pdf",
        labels,
        pred,
        hls,
        "Prediction",
        "Vitis HLS",
        resource.upper(),
        f"Prediction vs Vitis HLS {resource.upper()}",
    )


def _knob_latency(rows: list[dict[str, str]], out: Path) -> None:
    labels: list[str] = []
    values: list[float] = []

    for r in rows:
        if r.get("board") != "kv260" or r.get("mode") != "inference":
            continue
        lat = _num(r, "hls_latency_worst_cycles")
        if lat is None:
            continue
        labels.append(_short_name(r["design"]))
        values.append(lat)

    _save_bar(
        out / "knob_latency_effect.pdf",
        labels,
        values,
        "Worst-case HLS latency cycles",
        "KV260 inference design knob latency effect",
    )


def _hls_vs_vivado_lut(rows: list[dict[str, str]], out: Path) -> None:
    labels: list[str] = []
    hls: list[float] = []
    vivado: list[float] = []

    for r in rows:
        h = _num(r, "hls_lut")
        v = _num(r, "vivado_lut")
        if h is None or v is None:
            continue
        labels.append(_short_name(r["design"]))
        hls.append(h)
        vivado.append(v)

    _save_grouped_bar(
        out / "hls_vs_vivado_lut.pdf",
        labels,
        hls,
        vivado,
        "Vitis HLS",
        "Vivado impl",
        "LUTs",
        "HLS vs Vivado implementation LUTs",
    )


def _artifact_coverage(rows: list[dict[str, str]], out: Path) -> None:
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["paper_status"]] = statuses.get(r["paper_status"], 0) + 1

    labels = [
        "prediction",
        "HLS",
        "Vivado\nbitstream",
        "Vivado\ncapacity\nreject",
        "HLS-only",
    ]
    values = [
        sum(bool(r.get("prediction_lut")) for r in rows),
        sum(bool(r.get("hls_lut")) for r in rows),
        statuses.get("vivado_impl_bitstream_ready", 0),
        statuses.get("vivado_board_capacity_rejected", 0),
        statuses.get("hls_only", 0),
    ]

    _save_bar(
        out / "artifact_coverage.pdf",
        labels,
        [float(v) for v in values],
        "Design count",
        "Paper artifact coverage",
    )


def _training_capacity(rows: list[dict[str, str]], out: Path) -> None:
    labels: list[str] = []
    hls_lut: list[float] = []

    for r in rows:
        if not r["design"].startswith("training_"):
            continue
        v = _num(r, "hls_lut")
        if v is None:
            continue
        labels.append(_short_name(r["design"]))
        hls_lut.append(v)

    _save_bar(
        out / "training_capacity.pdf",
        labels,
        hls_lut,
        "Vitis HLS LUT estimate",
        "Training designs and KV260 capacity pressure",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="paper_results/parsed/paper_numeric_joined.csv")
    ap.add_argument("--out", default="paper_results/figures")
    args = ap.parse_args()

    rows = _read_rows(Path(args.input))
    out = Path(args.out)

    _prediction_vs_hls(rows, out, "lut")
    _prediction_vs_hls(rows, out, "dsp")
    _prediction_vs_hls(rows, out, "bram18")
    _knob_latency(rows, out)
    _hls_vs_vivado_lut(rows, out)
    _artifact_coverage(rows, out)
    _training_capacity(rows, out)

    figures = sorted(out.glob("*.pdf"))
    for fig in figures:
        print(f"[OK] wrote {fig}")

    print("[SUMMARY]")
    print(f"input_rows={len(rows)}")
    print(f"figure_count={len(figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
