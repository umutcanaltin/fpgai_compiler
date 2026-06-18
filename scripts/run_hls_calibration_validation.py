#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from fpgai.analysis.hls_calibration_validation import (
    load_calibration_dataset,
    run_calibration_validation,
    write_validation_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run held-out HLS calibration validation.")
    parser.add_argument("--dataset", required=True, help="Path to estimate_vs_hls.json or calibration dataset JSON")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["leave_one_sample_out", "leave_one_operator_out", "train_test_split"],
        help="Validation modes to run",
    )
    parser.add_argument("--test-fraction", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = load_calibration_dataset(args.dataset)
    report = run_calibration_validation(
        dataset,
        modes=args.modes,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    paths = write_validation_outputs(report, Path(args.out))

    print(f"[OK] HLS calibration validation completed: {args.out}")
    print(f"Samples: {report['sample_count']}")
    for key in ["json", "csv", "tex", "markdown", "plot"]:
        if key in paths:
            print(f"{key}: {paths[key]}")
    overall = report.get("summary", {}).get("overall", {})
    if overall:
        print(
            "Overall raw MAPE={:.2f} calibrated MAPE={:.2f} improvement={:.2f}x".format(
                float(overall.get("raw_mape", 0.0)),
                float(overall.get("calibrated_mape", 0.0)),
                float(overall.get("improvement_ratio", 0.0)),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
