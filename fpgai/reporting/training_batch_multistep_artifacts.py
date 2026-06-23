from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_one(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def _rel(path: Path | None, base: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def _summary_for_design(design_dir: Path) -> Dict[str, Any]:
    build = design_dir / "build"
    manifest = _load_json(build / "manifest.json")
    multi_summary_path = _find_one(build, "training_multistep_summary.json")
    multi_summary = _load_json(multi_summary_path) if multi_summary_path else {}

    training_compare = manifest.get("training_compare") or {}
    training_plan = manifest.get("training_plan") or {}
    cfg = manifest.get("configuration") or {}
    requested = cfg.get("requested") or {}
    effective = cfg.get("effective") or {}

    weights_before = _find_one(build, "weights_before.bin")
    grads = _find_one(build, "grads.bin")
    weights_after = _find_one(build, "weights_after.bin")

    row = {
        "design": design_dir.name,
        "status": "unknown",
        "model": manifest.get("model_path", ""),
        "weights_mode": manifest.get("weights_mode") or effective.get("weights_mode") or requested.get("weights_mode") or "",
        "hls_ok": manifest.get("hls_ok"),
        "training_plan": bool(training_plan),
        "training_reference": bool(manifest.get("training_reference")),
        "training_compare": bool(training_compare),
        "has_multistep_summary": multi_summary_path is not None,
        "multistep_summary_path": _rel(multi_summary_path, build),
        "train_steps": multi_summary.get("train_steps", ""),
        "batch_size": multi_summary.get("batch_size", ""),
        "total_train_calls": multi_summary.get("total_train_calls", ""),
        "weight_words": multi_summary.get("weight_words", ""),
        "hls_weights_before_bin": weights_before is not None,
        "hls_grads_bin": grads is not None,
        "hls_weights_after_bin": weights_after is not None,
        "grad_cosine": training_compare.get("grad_cosine", ""),
        "weight_after_cosine": training_compare.get("weight_after_cosine", ""),
        "weight_delta_cosine": training_compare.get("weight_delta_cosine", ""),
    }
    return row


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m fpgai.reporting.training_batch_multistep_artifacts <experiment_dir>", file=sys.stderr)
        return 2

    exp = Path(sys.argv[1])
    results_path = exp / "results.json"
    results = _load_json(results_path)
    statuses = {}
    for item in results.get("results", []):
        if isinstance(item, dict):
            statuses[item.get("design_name", "")] = item.get("status", "")

    artifacts = exp / "artifacts"
    rows: List[Dict[str, Any]] = []
    for design_dir in sorted(artifacts.iterdir() if artifacts.exists() else []):
        if not design_dir.is_dir():
            continue
        row = _summary_for_design(design_dir)
        row["status"] = statuses.get(row["design"], row["status"])
        rows.append(row)

    out_dir = exp / "training_batch_multistep_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "training_batch_multistep_artifacts.json"
    csv_path = out_dir / "training_batch_multistep_artifacts.csv"
    md_path = out_dir / "training_batch_multistep_artifacts.md"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fields = [
        "design",
        "status",
        "weights_mode",
        "hls_ok",
        "training_plan",
        "training_reference",
        "training_compare",
        "has_multistep_summary",
        "train_steps",
        "batch_size",
        "total_train_calls",
        "weight_words",
        "hls_weights_before_bin",
        "hls_grads_bin",
        "hls_weights_after_bin",
        "grad_cosine",
        "weight_after_cosine",
        "weight_delta_cosine",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    lines = ["# Training batch/multi-step artifacts", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md_path.read_text(encoding="utf-8"))
    print(f"[OK] Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
