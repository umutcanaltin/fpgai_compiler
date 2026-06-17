"""Cross-experiment comparison utilities for FPGAI experiment outputs."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class ExperimentComparisonRow:
    experiment: str
    experiment_dir: str
    total_records: int
    passed: int
    failed: int
    boards: str
    single_commit: bool
    commit_hashes: str
    mean_duration_sec: float | None
    raw_mape_lut: float | None
    cal_mape_lut: float | None
    raw_mape_ff: float | None
    cal_mape_ff: float | None
    raw_mape_dsp: float | None
    cal_mape_dsp: float | None
    raw_mape_bram: float | None
    cal_mape_bram: float | None
    raw_mape_latency_cycles: float | None
    cal_mape_latency_cycles: float | None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[Any]) -> float | None:
    xs = [_as_float(v) for v in values]
    nums = [x for x in xs if x is not None]
    if not nums:
        return None
    return float(mean(nums))


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def load_results(experiment_dir: str | Path) -> Dict[str, Any]:
    experiment_dir = Path(experiment_dir)
    path = experiment_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing results.json in {experiment_dir}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"Invalid FPGAI results.json: {path}")
    return payload


def summarize_experiment(experiment_dir: str | Path, name: str | None = None) -> ExperimentComparisonRow:
    experiment_dir = Path(experiment_dir)
    payload = load_results(experiment_dir)
    results: List[Mapping[str, Any]] = list(payload.get("results") or [])
    passed_rows = [r for r in results if r.get("status") == "passed"]
    failed_rows = [r for r in results if r.get("status") == "failed"]
    boards = sorted({str(r.get("board")) for r in results if r.get("board")})
    commits = sorted({str(r.get("commit_hash")) for r in results if r.get("commit_hash")})

    def metric_avg(metric: str) -> float | None:
        return _mean((r.get("metrics") or {}).get(metric) for r in passed_rows)

    return ExperimentComparisonRow(
        experiment=name or experiment_dir.name,
        experiment_dir=str(experiment_dir),
        total_records=len(results),
        passed=len(passed_rows),
        failed=len(failed_rows),
        boards=", ".join(boards),
        single_commit=len(commits) <= 1,
        commit_hashes=", ".join(commits),
        mean_duration_sec=_mean(r.get("duration_sec") for r in passed_rows),
        raw_mape_lut=metric_avg("raw_mape.lut"),
        cal_mape_lut=metric_avg("cal_mape.lut"),
        raw_mape_ff=metric_avg("raw_mape.ff"),
        cal_mape_ff=metric_avg("cal_mape.ff"),
        raw_mape_dsp=metric_avg("raw_mape.dsp"),
        cal_mape_dsp=metric_avg("cal_mape.dsp"),
        raw_mape_bram=metric_avg("raw_mape.bram"),
        cal_mape_bram=metric_avg("cal_mape.bram"),
        raw_mape_latency_cycles=metric_avg("raw_mape.latency_cycles"),
        cal_mape_latency_cycles=metric_avg("cal_mape.latency_cycles"),
    )


def compare_experiments(experiment_dirs: Sequence[str | Path]) -> List[ExperimentComparisonRow]:
    return [summarize_experiment(p) for p in experiment_dirs]


def write_comparison_csv(rows: Sequence[ExperimentComparisonRow], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ExperimentComparisonRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: getattr(row, k) for k in fields})
    return path


def escape_latex(text: Any) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def comparison_latex_table(rows: Sequence[ExperimentComparisonRow]) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Experiment & Records & Passed & Mean duration (s) & LUT MAPE (\%) \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{escape_latex(row.experiment)} & {row.total_records} & {row.passed} & "
            f"{_fmt(row.mean_duration_sec)} & {_fmt(row.raw_mape_lut)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_comparison_tex(rows: Sequence[ExperimentComparisonRow], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(comparison_latex_table(rows), encoding="utf-8")
    return path


def write_comparison_overview(rows: Sequence[ExperimentComparisonRow], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FPGAI Experiment Comparison",
        "",
        "| Experiment | Records | Passed | Failed | Boards | Single commit | Mean duration (s) | LUT MAPE | Cal. LUT MAPE |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.experiment}` | {row.total_records} | {row.passed} | {row.failed} | "
            f"{row.boards or '-'} | {'yes' if row.single_commit else 'no'} | "
            f"{_fmt(row.mean_duration_sec)} | {_fmt(row.raw_mape_lut)} | {_fmt(row.cal_mape_lut)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_comparison_json(rows: Sequence[ExperimentComparisonRow], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "experiment_count": len(rows), "experiments": [row.__dict__ for row in rows]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_comparison_plots(rows: Sequence[ExperimentComparisonRow], out_dir: str | Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    paths: List[Path] = []
    names = [row.experiment for row in rows]

    durations = [row.mean_duration_sec or 0.0 for row in rows]
    fig = plt.figure(figsize=(max(6, len(names) * 1.5), 4))
    plt.bar(names, durations)
    plt.ylabel("Mean duration (s)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = out_dir / "mean_duration_by_experiment.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    raw = [row.raw_mape_lut or 0.0 for row in rows]
    cal = [row.cal_mape_lut or 0.0 for row in rows]
    x = list(range(len(names)))
    width = 0.35
    fig = plt.figure(figsize=(max(6, len(names) * 1.5), 4))
    plt.bar([i - width / 2 for i in x], raw, width, label="LUT MAPE")
    plt.bar([i + width / 2 for i in x], cal, width, label="Cal. LUT MAPE")
    plt.ylabel("MAPE (%)")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    p = out_dir / "mape_by_experiment.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    return paths


def write_comparison_outputs(rows: Sequence[ExperimentComparisonRow], out_dir: str | Path, plots: bool = True) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    outputs["comparison_summary_csv"] = str(write_comparison_csv(rows, out_dir / "comparison_summary.csv"))
    outputs["comparison_summary_tex"] = str(write_comparison_tex(rows, out_dir / "comparison_summary.tex"))
    outputs["comparison_overview_md"] = str(write_comparison_overview(rows, out_dir / "comparison_overview.md"))
    outputs["comparison_summary_json"] = str(write_comparison_json(rows, out_dir / "comparison_summary.json"))
    if plots:
        for p in write_comparison_plots(rows, out_dir):
            outputs[p.stem] = str(p)
    return outputs
