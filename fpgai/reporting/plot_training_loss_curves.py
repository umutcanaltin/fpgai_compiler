#!/usr/bin/env python3
"""
Plot Sprint 16B training loss curves/table summary.

Input:
  evidence/sprint16b_training_convergence/training_convergence.csv

Output:
  evidence/sprint16b_training_convergence/loss_curve.png

If only initial/final loss are available, this plots a two-point loss line per design.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def as_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--out", default="evidence/sprint16b_training_convergence/loss_curve.png")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    out = Path(args.out)
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    usable = []
    for r in rows:
        initial = as_float(r.get("initial_loss", ""))
        final = as_float(r.get("final_loss", ""))
        if initial is not None and final is not None:
            usable.append((r.get("design", "design"), initial, final))

    if not usable:
        raise SystemExit("No rows with both initial_loss and final_loss found; cannot plot.")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for design, initial, final in usable:
        ax.plot([0, 1], [initial, final], marker="o", label=design)

    ax.set_title("FPGAI Training Convergence Smoke Evidence")
    ax.set_xlabel("Evaluation point")
    ax.set_ylabel("Loss")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["initial", "final"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
