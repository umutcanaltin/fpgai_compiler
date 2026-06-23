#!/usr/bin/env python3
"""Compare multiple FPGAI experiment result directories."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from fpgai.experiments.comparison import compare_experiments, write_comparison_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare FPGAI experiment outputs")
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment directories containing results.json")
    parser.add_argument("--out", required=True, help="Output directory for comparison artifacts")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate PNG plots")
    return parser


def run_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    rows = compare_experiments([Path(p) for p in args.experiments])
    outputs = write_comparison_outputs(rows, Path(args.out), plots=not args.no_plots)
    return {"rows": rows, "outputs": outputs}


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = run_comparison(args)
    print(f"[OK] Comparison completed: {args.out}")
    for path in payload["outputs"].values():
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
