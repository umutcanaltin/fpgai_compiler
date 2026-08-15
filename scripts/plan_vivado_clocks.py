from __future__ import annotations

import argparse
from pathlib import Path

from fpgai.analysis.board_clock_planner import run_clock_probe, write_clock_probe_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe board-realizable PL clocks without running HLS or implementation")
    ap.add_argument("--board", default="kv260")
    ap.add_argument("--requests", nargs="+", type=float, required=True)
    ap.add_argument("--out", default="build/clock_plans")
    ap.add_argument("--vivado", default="vivado")
    args = ap.parse_args()

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    points = []
    for mhz in args.requests:
        point = run_clock_probe(
            board_name=args.board,
            requested_mhz=mhz,
            out_dir=root / f"probe_{mhz:g}mhz",
            vivado_executable=args.vivado,
        )
        points.append(point)
        print(f"{mhz:g} MHz -> {point.actual_mhz if point.actual_mhz is not None else 'unresolved'} MHz ({point.status})")
    report = write_clock_probe_report(points, root / "board_clock_probe.json")
    print(report)


if __name__ == "__main__":
    main()
