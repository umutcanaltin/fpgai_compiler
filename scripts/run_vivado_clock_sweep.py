from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from fpgai.analysis.clock_sweep import point_from_characterization, summarize_clock_sweep


def _set_clock(raw: dict, mhz: float) -> None:
    clocks = raw.setdefault("targets", {}).setdefault("platform", {}).setdefault("clocks", [])
    if clocks:
        clocks[0]["target_mhz"] = float(mhz)
    else:
        clocks.append({"name": "pl_clk0", "target_mhz": float(mhz)})


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run constraint-verified Vivado implementation clock sweep through FPGAI")
    ap.add_argument("--config", required=True)
    ap.add_argument("--clocks", nargs="+", type=float, required=True)
    ap.add_argument("--out", default="build/clock_sweeps")
    ap.add_argument("--continue-on-failure", action="store_true")
    args = ap.parse_args()

    base_path = Path(args.config).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    sweep_root = Path(args.out).resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)
    points = []

    for mhz in args.clocks:
        raw = copy.deepcopy(base)
        point_dir = sweep_root / f"{mhz:g}mhz"
        raw.setdefault("project", {})["out_dir"] = str(point_dir)
        raw.setdefault("project", {})["clean"] = True
        _set_clock(raw, mhz)
        cfg = sweep_root / f"config_{mhz:g}mhz.yml"
        cfg.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        proc = subprocess.run([sys.executable, "-m", "fpgai.cli", "compile", "--config", str(cfg)])

        report = point_dir / "reports" / "vivado_implementation_characterization.json"
        payload = _load_json(report) if report.is_file() else {
            "status": "tool_failed",
            "clock": {"timing_met": None, "wns_ns": None},
        }
        hls_payload = _load_json(point_dir / "reports" / "hls_synthesis_characterization.json")
        timing_report = point_dir / "vivado_bridge" / "reports" / "timing_impl.rpt"
        point = point_from_characterization(
            mhz,
            payload,
            out_dir=str(point_dir),
            timing_report=timing_report,
            hls_payload=hls_payload,
        )
        points.append(point)
        if not point.constraint_verified:
            print(
                f"[CLOCK-SWEEP] {mhz:g} MHz constraint NOT verified: "
                f"requested={point.requested_period_ns:.6g} ns, "
                f"Vivado={point.vivado_constraint_period_ns} ns",
                file=sys.stderr,
            )
        if proc.returncode and not args.continue_on_failure:
            break

    summary = summarize_clock_sweep(points)
    out_json = sweep_root / "clock_sweep.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()
