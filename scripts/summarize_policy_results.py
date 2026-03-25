from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def to_float(x: Any) -> Optional[float]:
    try:
        if x in (None, "", "None"):
            return None
        return float(x)
    except Exception:
        return None


def to_bool(x: Any) -> Optional[bool]:
    if x in (None, "", "None"):
        return None
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
            return False
    return None


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser("Summarize FPGAI policy sweep CSV")
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    rows = read_rows(Path(args.csv).resolve())
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_model[str(r.get("model_name", ""))].append(r)

    for model_name, model_rows in by_model.items():
        print("=" * 72)
        print(model_name)

        valid = [r for r in model_rows if to_bool(r.get("compile_ok")) is not False]
        print(f"rows={len(model_rows)} valid={len(valid)}")

        best_rmse = None
        best_latency = None
        best_balanced = None

        for r in valid:
            rmse = to_float(r.get("rmse"))
            lat = to_float(r.get("latency_seconds_max")) or to_float(r.get("latency_seconds_min"))
            dsp = to_float(r.get("dsp")) or 0.0

            if rmse is not None:
                if best_rmse is None or rmse < to_float(best_rmse["rmse"]):
                    best_rmse = r

            if lat is not None:
                cur = to_float(best_latency.get("latency_seconds_max")) if best_latency else None
                if best_latency is None or (cur is None) or lat < cur:
                    best_latency = r

            if rmse is not None and lat is not None:
                score = lat * (1.0 + rmse) * (1.0 + 0.005 * dsp)
                if best_balanced is None:
                    best_balanced = dict(r)
                    best_balanced["_score"] = score
                elif score < best_balanced["_score"]:
                    best_balanced = dict(r)
                    best_balanced["_score"] = score

        def show(tag: str, r: Optional[Dict[str, Any]]) -> None:
            if not r:
                print(f"{tag}: none")
                return
            print(
                f"{tag}: "
                f'precision={r.get("precision_policy")} '
                f'parallel={r.get("parallel_policy")} '
                f'rmse={r.get("rmse")} '
                f'latency_s={r.get("latency_seconds_max") or r.get("latency_seconds_min")} '
                f'dsp={r.get("dsp")} lut={r.get("lut")} bram={r.get("bram_18k")}'
            )

        show("best_rmse", best_rmse)
        show("best_latency", best_latency)
        show("best_balanced", best_balanced)


if __name__ == "__main__":
    main()