#!/usr/bin/env python3
"""Analyze FPGAI experiment outputs and generate paper-ready artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from fpgai.experiments.analysis import load_experiment, write_metrics_overview, write_summary_csv
from fpgai.experiments.latex_tables import write_latex_tables
from fpgai.experiments.plotting import write_plots


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze FPGAI experiment results")
    parser.add_argument("--experiment", required=True, help="Experiment directory containing results.json")
    parser.add_argument("--out", default=None, help="Output analysis directory; default: <experiment>/analysis")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plot generation")
    return parser


def analyze_experiment(experiment: str | Path, out: str | Path | None = None, *, no_plots: bool = False) -> Dict[str, Any]:
    analysis = load_experiment(experiment)
    out_dir = Path(out) if out is not None else analysis.experiment_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: List[str] = []
    artifacts.append(str(write_summary_csv(analysis.records, out_dir / "summary_by_design.csv")))
    artifacts.append(str(write_metrics_overview(analysis, out_dir / "metrics_overview.md")))
    artifacts.extend(str(p) for p in write_latex_tables(analysis.records, out_dir))
    if not no_plots:
        artifacts.extend(str(p) for p in write_plots(analysis.records, out_dir))

    summary_path = out_dir / "analysis_summary.json"
    payload = {
        "experiment_dir": str(analysis.experiment_dir),
        "analysis_dir": str(out_dir),
        "summary": analysis.summary,
        "artifacts": artifacts,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    artifacts.append(str(summary_path))
    payload["artifacts"] = artifacts
    return payload


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = analyze_experiment(args.experiment, args.out, no_plots=args.no_plots)
    print(f"[OK] Analysis completed: {payload['analysis_dir']}")
    for artifact in payload["artifacts"]:
        print(f" - {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
