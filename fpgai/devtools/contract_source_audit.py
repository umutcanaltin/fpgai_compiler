from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("paper_experiments/full_pipeline_gate/sprint25_e2e_audit")
OUT_JSON = ROOT / "contract_source_audit.json"
OUT_MD = ROOT / "contract_source_audit.md"


def _read_texts(case_dir: Path) -> str:
    chunks: list[str] = []
    for rel in [
        "hls/src/deeplearn.cpp",
        "hls/src/fpgai_params.cpp",
        "hls/include/fpgai_types.h",
        "hls/run_hls.tcl",
    ]:
        p = case_dir / rel
        if p.exists():
            chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _contract_rows(case_dir: Path) -> dict[str, dict[str, Any]]:
    contract = _load_json(case_dir / "reports" / "hardware_knob_contract.json")
    rows: dict[str, dict[str, Any]] = {}

    if isinstance(contract.get("rows"), dict):
        for key, value in contract["rows"].items():
            if isinstance(value, dict):
                rows[str(key)] = value
        return rows

    knobs = contract.get("knobs", [])
    if isinstance(knobs, list):
        for row in knobs:
            if isinstance(row, dict) and row.get("path"):
                rows[str(row["path"])] = row

    return rows


def _find_row(rows: dict[str, dict[str, Any]], *paths: str) -> tuple[str | None, dict[str, Any] | None]:
    for path in paths:
        if path in rows:
            return path, rows[path]
    return None, None


def _effective(row: dict[str, Any] | None) -> Any:
    return row.get("effective") if isinstance(row, dict) else None


def _add(checks: list[dict[str, Any]], knob: str, expected: Any, passed: bool, detail: str) -> None:
    checks.append(
        {
            "knob": knob,
            "expected": expected,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def _check_case(case_dir: Path) -> dict[str, Any]:
    rows = _contract_rows(case_dir)
    source = _read_texts(case_dir)
    low = source.lower()
    checks: list[dict[str, Any]] = []

    path, row = _find_row(rows, "memory.weight_storage")
    weight = str(_effective(row) or "").lower()
    if path and weight in {"bram", "uram", "lutram"}:
        _add(
            checks,
            path,
            weight,
            (
                f"impl={weight}" in low
                or f"requested for {weight}" in low
                or f"-> {weight}" in low
            ),
            f"{weight.upper()} contract must appear in canonical HLS source or params trace.",
        )

    path, row = _find_row(rows, "optimization.pipeline.ii", "pipeline.ii")
    ii = _effective(row)
    if path and ii is not None:
        _add(
            checks,
            path,
            ii,
            "#pragma hls pipeline" in low or "pipeline ii=" in low or "ii=" in low,
            "Pipeline II contract must appear as a pipeline pragma or metadata.",
        )

    path, row = _find_row(
        rows,
        "optimization.parallel.partition_factor",
        "parallel.partition_factor",
    )
    partition = _effective(row)
    if path and partition is not None:
        _add(
            checks,
            path,
            partition,
            "#pragma hls array_partition" in low,
            "Partition factor contract must appear as ARRAY_PARTITION pragma.",
        )

    path, row = _find_row(
        rows,
        "optimization.parallel.unroll_factor",
        "parallel.unroll_factor",
    )
    unroll = _effective(row)
    if path and unroll is not None:
        _add(
            checks,
            path,
            unroll,
            "#pragma hls unroll" in low,
            "Unroll factor contract must appear as UNROLL pragma.",
        )

    if not checks:
        _add(
            checks,
            "contract.rows",
            "non-empty",
            False,
            "No supported contract rows were parsed. Empty PASS is not allowed.",
        )

    return {
        "case": case_dir.name,
        "passed": all(check["passed"] for check in checks),
        "check_count": len(checks),
        "checks": checks,
    }


def main() -> int:
    if not ROOT.exists():
        raise SystemExit(f"Missing audit root: {ROOT}")

    case_dirs = [
        p for p in sorted(ROOT.iterdir())
        if p.is_dir() and (p / "reports" / "hardware_knob_contract.json").exists()
    ]

    results = [_check_case(case_dir) for case_dir in case_dirs]
    overall = bool(results) and all(result["passed"] for result in results)

    OUT_JSON.write_text(
        json.dumps(
            {
                "root": str(ROOT),
                "passed": overall,
                "case_count": len(results),
                "cases": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Contract/source audit",
        "",
        f"Overall: {'PASS' if overall else 'FAIL'}",
        f"Cases: {len(results)}",
        "",
    ]

    for result in results:
        lines.extend(
            [
                f"## {result['case']}",
                "",
                f"Status: {'PASS' if result['passed'] else 'FAIL'}",
                f"Checks: {result['check_count']}",
                "",
                "| Knob | Expected | Status | Detail |",
                "|---|---:|---|---|",
            ]
        )
        for check in result["checks"]:
            lines.append(
                f"| `{check['knob']}` | `{check['expected']}` | "
                f"{'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |"
            )
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] wrote {OUT_JSON}")
    print(f"[OK] wrote {OUT_MD}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
