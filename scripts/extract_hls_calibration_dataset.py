#!/usr/bin/env python3
"""Extract an FPGAI HLS calibration dataset and reports from an existing build.

Example:
    python scripts/extract_hls_calibration_dataset.py \
      --build-dir build/my_project \
      --compile-plan build/my_project/compile_plan.json \
      --out build/my_project/calibration
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

# Works both from repo root and from direct script execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fpgai.analysis.hls_calibration_dataset import build_calibration_dataset
from fpgai.analysis.hls_calibration_model import fit_calibration_model
from fpgai.analysis.hls_estimate_report import write_estimate_vs_hls_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, help="FPGAI build directory containing HLS reports")
    parser.add_argument("--compile-plan", default=None, help="Compile plan/manifest JSON with FPGAI estimates")
    parser.add_argument("--out", default=None, help="Output calibration directory")
    parser.add_argument("--fail-on-empty", action="store_true", help="Return non-zero if no samples are extracted")
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    out_dir = Path(args.out) if args.out else build_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    compile_plan = Path(args.compile_plan) if args.compile_plan else _find_compile_plan(build_dir)
    if compile_plan is None:
        print(f"[FPGAI] No compile plan JSON found under {build_dir}", file=sys.stderr)
        return 2 if args.fail_on_empty else 0

    dataset_path = out_dir / "hls_operator_dataset.json"
    model_path = out_dir / "calibrated_model.json"
    estimate_vs_hls_path = out_dir / "estimate_vs_hls.json"
    summary_path = out_dir / "summary.txt"

    dataset = build_calibration_dataset(
        compile_plan_path=compile_plan,
        hls_report_dir=build_dir,
        output_path=dataset_path,
    )
    model = fit_calibration_model(dataset)
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True))
    write_estimate_vs_hls_report(dataset, model, estimate_vs_hls_path, summary_path)

    sample_count = len(dataset.get("samples", []))
    print(f"[FPGAI] HLS calibration samples: {sample_count}")
    print(f"[FPGAI] Wrote {dataset_path}")
    print(f"[FPGAI] Wrote {model_path}")
    print(f"[FPGAI] Wrote {estimate_vs_hls_path}")
    print(f"[FPGAI] Wrote {summary_path}")

    if args.fail_on_empty and sample_count == 0:
        return 3
    return 0


def _find_compile_plan(build_dir: Path) -> Path | None:
    names = (
        "compile_plan.json",
        "plan.json",
        "manifest.json",
        "summary.json",
        "benchmark_summary.json",
    )
    for name in names:
        matches = list(build_dir.rglob(name))
        if matches:
            return matches[0]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
