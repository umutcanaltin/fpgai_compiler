#!/usr/bin/env python3
"""Run FPGAI experiment sweeps.

Sprint 6 experiment automation CLI.
This script intentionally keeps path handling explicit:
relative --sweep and --out paths are resolved against --repo-root.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Allow running directly from the repository root without installation.
_REPO_GUESS = Path(__file__).resolve().parents[1]
if str(_REPO_GUESS) not in sys.path:
    sys.path.insert(0, str(_REPO_GUESS))

from fpgai.experiments.design_matrix import expand_design_matrix, load_sweep_config  # noqa: E402
from fpgai.experiments.sweep_runner import SweepRunner  # noqa: E402


def _resolve_under_repo(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an FPGAI experiment sweep")
    parser.add_argument("--sweep", required=True, help="Path to sweep YAML, relative to repo root unless absolute")
    parser.add_argument("--out", required=True, help="Output experiment directory, relative to repo root unless absolute")
    parser.add_argument("--max-design-points", type=int, default=None, help="Limit number of design points")
    parser.add_argument("--dry-run", action="store_true", help="Record commands without executing them")
    parser.add_argument("--timeout-sec", type=int, default=None, help="Per-design timeout in seconds")
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving relative paths")
    return parser


def run_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    sweep_path = _resolve_under_repo(repo_root, args.sweep)
    out_dir = _resolve_under_repo(repo_root, args.out)

    config = load_sweep_config(sweep_path)
    # Preserve source path relative to repo root where possible.
    try:
        source_path = str(sweep_path.relative_to(repo_root))
    except Exception:
        source_path = str(sweep_path)
    config.setdefault("source_path", source_path)

    points = expand_design_matrix(config, limit=args.max_design_points)

    runner = SweepRunner(
        out_dir,
        repo_root=repo_root,
        dry_run=bool(args.dry_run),
        timeout_sec=args.timeout_sec,
        materialize_configs=config.get("materialize_configs"),
        command_template=config.get("command_template"),
    )
    payload = runner.run_points(points)
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_sweep(args)

    results = payload.get("results", []) if isinstance(payload.get("results", []), list) else []
    failed = int(payload.get("failed_count", 0) or 0)
    if failed == 0 and results:
        failed = sum(1 for row in results if row.get("status") == "failed")
    total = int(payload.get("result_count", len(results)) or 0)
    results_path = Path(payload.get("experiment_dir", args.out)) / "results.json"
    if failed:
        print(f"[WARN] Sweep completed: {total} records, {failed} failed")
        print(f"Results: {results_path}")
        return 0
    print(f"[OK] Sweep completed: {total} records, 0 failed")
    print(f"Results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
