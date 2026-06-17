"""LaTeX table helpers for FPGAI experiment analysis.

These helpers intentionally avoid external dependencies so experiment analysis
can run in minimal CI environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence


_LATEX_ESCAPE_MAP = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def escape_latex(value: object) -> str:
    """Escape a value for safe use in a LaTeX tabular cell."""
    text = "" if value is None else str(value)
    return "".join(_LATEX_ESCAPE_MAP.get(ch, ch) for ch in text)


def _metric(record: object, key: str, default: object = None) -> object:
    metrics = getattr(record, "metrics", None)
    if isinstance(metrics, Mapping):
        return metrics.get(key, default)
    if isinstance(record, Mapping):
        metrics = record.get("metrics", {})
        if isinstance(metrics, Mapping):
            return metrics.get(key, default)
    return default


def _attr(record: object, key: str, default: object = None) -> object:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _fmt_num(value: object, digits: int = 2) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return escape_latex(value)


def _tabular(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    column_spec = "l" * len(headers)
    lines = [
        r"\begin{tabular}{" + column_spec + "}",
        r"\hline",
        " & ".join(escape_latex(h) for h in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(escape_latex(cell) for cell in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def design_summary_table(records: Sequence[object]) -> str:
    """Return a LaTeX table summarizing one row per design point."""
    rows = []
    for rec in records:
        raw_lut = _metric(rec, "raw_mape.lut")
        cal_lut = _metric(rec, "cal_mape.lut")
        rows.append(
            [
                _attr(rec, "design_index", ""),
                _attr(rec, "design_name", ""),
                _attr(rec, "status", ""),
                _fmt_num(_attr(rec, "duration_sec", None), 1),
                _fmt_num(raw_lut, 2),
                _fmt_num(cal_lut, 2),
            ]
        )
    return _tabular(
        ["#", "Design", "Status", "Duration (s)", "Raw LUT MAPE", "Cal. LUT MAPE"],
        rows,
    )


def _average_metric(records: Sequence[object], key: str) -> float | None:
    values = []
    for rec in records:
        val = _metric(rec, key)
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            pass
    return sum(values) / len(values) if values else None


def resource_error_table(records: Sequence[object]) -> str:
    """Return a compact LaTeX table of average raw/calibrated MAPE metrics.

    The row labels are intentionally paper-friendly (for example, ``Raw LUT``
    and ``Cal. LUT``) so the generated table can be pasted into a manuscript
    without renaming metric keys such as ``raw_mape.lut``.
    """
    metric_labels = [
        ("lut", "LUT"),
        ("ff", "FF"),
        ("dsp", "DSP"),
        ("bram", "BRAM"),
        ("latency_cycles", "Latency cycles"),
    ]
    rows = []
    for metric_key, label in metric_labels:
        raw_avg = _average_metric(records, f"raw_mape.{metric_key}")
        cal_avg = _average_metric(records, f"cal_mape.{metric_key}")
        rows.append([f"Raw {label}", _fmt_num(raw_avg, 2)])
        rows.append([f"Cal. {label}", _fmt_num(cal_avg, 2)])
    return _tabular(["Metric", "MAPE (\\%)"], rows)


def write_latex_tables(records: Sequence[object], out_dir: str | Path) -> dict[str, Path]:
    """Write paper-ready LaTeX tables and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary_by_design.tex"
    resource_path = out / "resource_error_table.tex"
    summary_path.write_text(design_summary_table(records), encoding="utf-8")
    resource_path.write_text(resource_error_table(records), encoding="utf-8")
    return {
        "summary_by_design_tex": summary_path,
        "resource_error_table_tex": resource_path,
    }
